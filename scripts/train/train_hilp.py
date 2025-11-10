"""
Training Hilbert Representation Model phi(s)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import wandb
import datetime as dt
import numpy as np
import glob
from tqdm import tqdm
import click
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from model.hilp import HilbertRepresentation
from dataset.dataset import GoalDataset
from utils.seed_utils import seed_all
from utils.utils import l2_expectile_loss
from utils.logger import JsonLogger, _stats_dict
from utils.multiprocessing import _worker_init_fn


def eval(
    model, 
    target_model, 
    dataloader, 
    epoch,
    total_epoch,
    gamma,
    expectile_tau,
    device
    ):
    model.eval()
    target_model.eval()
    
    total_loss = 0.0
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        desc=f"[Validation] {epoch}/{total_epoch}", 
        leave=True, 
        ncols=100
        )
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for i, (state, next_state, goal, is_goal_now) in pbar:
            state = state.to(device, non_blocking=True)
            next_state = next_state.to(device, non_blocking=True)
            goal = goal.to(device, non_blocking=True)
            is_goal_now = is_goal_now.to(device, non_blocking=True).float()
            
            B, C, H, W = state.shape

            sg = torch.cat([state, goal], dim=0)            # (2*B, C, H, W)
            ng = torch.cat([next_state, goal], dim=0)       # (2*B, C, H, W)
            
            phi = model(sg)
            phi_s, phi_g = phi[:B], phi[B:]
            
            phi_next = target_model(ng)
            phi_next_s, phi_next_g = phi_next[:B], phi_next[B:]
            
            temporal_dist = torch.norm(phi_s - phi_g, dim=-1)       # (B, )
            reward = -1.0 * (1.0 - is_goal_now)
            target_dist = gamma * torch.norm(phi_next_s - phi_next_g, dim=-1)
            delta = reward - target_dist + temporal_dist

            loss = l2_expectile_loss(delta, expectile_tau)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(
    model, 
    target_model, 
    train_dataloader, 
    val_dataloader,
    scaler,
    wb,
    ckpt,
    checkpoint_dir,
    cfg,
    logger
    ):
    
    model = model.to(cfg.device)
    target_model = target_model.to(cfg.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs*len(train_dataloader))
    
    if ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        del ckpt
    
    print(f"[Torch] {cfg.device} is used.")
    print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")
    
    best_loss = float(np.inf)
    best_train_loss = float(np.inf)
    global_step = 0
    early_stop_counter = 0
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader), 
            desc=f"[Train] Epoch {epoch}/{cfg.num_epochs}", 
            leave=True, 
            ncols=100
            )
        model.train()
        
        for i, (state, next_state, goal, is_goal_now) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            next_state = next_state.to(cfg.device, non_blocking=True)
            goal =  goal.to(cfg.device, non_blocking=True)
            is_goal_now = is_goal_now.to(cfg.device, non_blocking=True).float()
            
            B, C, H, W = state.shape
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # phi(s), phi(g)
                sg = torch.cat([state, goal], dim=0)
                phi = model(sg)
                phi_s, phi_g = phi[:B], phi[B:]
                
                with torch.inference_mode():
                    ng = torch.cat([next_state, goal], dim=0)
                    phi_next = target_model(ng)
                    # target_phi(s'), target_phi(g)
                    target_phi_next_s, target_phi_g = phi_next[:B], phi_next[B:]
                
                temporal_dist = torch.norm(phi_s - phi_g, dim=-1)       # (B, )
                
                reward = -1.0 * (1.0 - is_goal_now)
                
                target_dist = cfg.gamma * torch.norm(target_phi_next_s - target_phi_g, dim=-1)
                delta = reward - target_dist + temporal_dist
                
                loss = l2_expectile_loss(delta, cfg.expectile_tau)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            lr_scheduler.step()
            scaler.update()

            if (i % cfg.target_update_frequency) == 0:
                with torch.no_grad():
                    tau = cfg.tau
                    for param, target_param in zip(model.parameters(), target_model.parameters()):
                        target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            if (i % cfg.log_frequency) == 0:
                step_log = {"train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/loss": loss.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "debug/temporal_dist_mean": float(temporal_dist.mean()),
                            "debug/target_dist_mean": float(target_dist.mean()),
                            "debug/delta_abs_mean": float(delta.abs().mean())
                            }
                step_log.update(_stats_dict("debug/state", state))
                step_log.update(_stats_dict("debug/next_state", next_state))
                step_log.update(_stats_dict("debug/goal", goal))
                step_log.update({"debug/has_nan_input": int(torch.isnan(state).any().item() or torch.isnan(next_state).any().item() or torch.isnan(goal).any().item())})
                
                logger.log(step_log)
                if wb:
                    wandb.log(step_log, step=global_step)
            
            global_step += 1
            
        avg_loss = total_loss / len(train_dataloader)
        
        if avg_loss <= best_train_loss:
            best_train_loss = avg_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= cfg.patience:
            print(f"[Train] Early stopped at epoch {epoch}")
            break
        
        step_log = {"train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/loss": avg_loss,
                    "train/lr": lr_scheduler.get_last_lr()[0]
                    }
        
        logger.log(step_log)
        if wb:
            wandb.log(step_log, step=global_step)
         
        if epoch % cfg.eval_frequency == 0:
            eval_loss = eval(model=model, 
                            target_model=target_model, 
                            dataloader=val_dataloader, 
                            epoch=epoch,
                            total_epoch=cfg.num_epochs,
                            gamma=cfg.gamma,
                            expectile_tau=cfg.expectile_tau,
                            device=cfg.device)
            
            print(f"[Validation] Loss: {eval_loss:.4f}")
            
            eval_log = {'eval/epoch': epoch,
                        'eval/global_step': global_step,
                        'eval/loss': eval_loss}
            
            logger.log(eval_log)
            if wb:
                wandb.log(eval_log, step=global_step)
            
            if eval_loss <= best_loss:
                import gc
                gc.collect()
                
                print(f"Save best model of epoch {epoch} (loss={eval_loss:.4f})")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_loss_{eval_loss:.3f}.pt")
                
                torch.save({
                    "epoch": epoch,
                    "hilbert_representation_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": avg_loss,
                    "eval_loss": eval_loss
                }, checkpoint_path)
                best_loss = eval_loss
                
    print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")



@click.command()
@click.option("-c", "--config", default="train_hilp", type=str, help="Config to use")
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f"config/{config}.yaml")
    cfg = OmegaConf.load(CONFIG_FILE)

    if cfg.resume:
        resum_conf = OmegaConf.load(os.path.join(os.getcwd(), f"data/outputs/hilp/{cfg.resume_ckpt_dir}/{config}.yaml"))
        cfg.data = resum_conf.data
        cfg.model = resum_conf.model
        cfg.train = resum_conf.train
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
        
    seed_all(train_cfg.seed)
    
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    model = HilbertRepresentation(model_cfg)
    target_model = HilbertRepresentation(model_cfg)
    target_model.load_state_dict(model.state_dict())
    
    for p in target_model.parameters():
        p.requires_grad_(False)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    ckpt = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"data/outputs/hilp/{cfg.resume_ckpt_dir}/*.pt")))
        ckpt = torch.load(ckpts[-1])
        print(f"Resume from {ckpts[-1]}")
        model.load_state_dict(ckpt["hilbert_representation_state_dict"])
        target_model.load_state_dict(ckpt["hilbert_representation_state_dict"])
        
    dataset = GoalDataset(train_cfg.seed, data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=data_cfg.train_batch_size, 
        shuffle=True, 
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        pin_memory_device=train_cfg.device,
        persistent_workers=(data_cfg.num_workers>0),
        worker_init_fn=_worker_init_fn,
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=data_cfg.val_batch_size, 
        shuffle=False, 
        num_workers=data_cfg.num_workers, 
        pin_memory=data_cfg.pin_memory,
        pin_memory_device=train_cfg.device,
        persistent_workers=(data_cfg.num_workers>0),
        worker_init_fn=_worker_init_fn,
        )
    
    checkpoint_dir = os.path.join(os.getcwd(), f"data/outputs/hilp/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
    
    logger = JsonLogger(path=os.path.join(checkpoint_dir, "log.json"))
    logger.start()
    if cfg.wb:
        wandb.init(project=cfg.wandb_project,
                   config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = cfg.wandb_tag
        wandb.run.name = f"{cfg.wandb_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        train(
            model=model, 
            target_model=target_model, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
            scaler=scaler,
            ckpt=ckpt, 
            wb=cfg.wb,
            checkpoint_dir=checkpoint_dir,
            cfg=train_cfg,
            logger=logger,
            )
    finally:
        logger.stop()


if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()