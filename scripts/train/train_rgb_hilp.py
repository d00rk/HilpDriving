"""
Training Hilbert Representation Model phi(s)
Example: torchrun --nproc_per_node=2 scripts/train/train_rgb_hilp.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import wandb
import datetime as dt
import numpy as np
import glob
from tqdm import tqdm
import click
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torchvision.utils import save_image

from model.hilp import ViewAwareHilbertRepresentation
from dataset.rgb_dataset import GoalRGBDataset
from utils.seed_utils import seed_all
from utils.utils import ensure_chw, l2_expectile_loss, update_exponential_moving_average
from utils.logger import JsonLogger, _stats_dict
from utils import ddp as ddp_utils


def eval(
    model, 
    target_model, 
    dataloader, 
    epoch,
    ckpt_dir,
    cfg
    ):
    model.eval()
    target_model.eval()
    
    eval_loss = {}
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        desc=f"[Validation] {epoch}/{cfg.num_epochs}", 
        leave=True, 
        ncols=100,
        disable=not ddp_utils.is_main_process()
        )
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=cfg.use_amp):
        total_loss = 0.0
        total_loss_hilp = 0.0
        total_loss_bev = 0.0
        num_batches = 0
        for i, (state, next_state, goal, bev, is_goal_now) in pbar:
            for k, v in state.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                state[k] = v
            
            for k, v in next_state.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                next_state[k] = v
            
            for k, v in goal.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                goal[k] = v
            
            bev = bev.to(cfg.device, non_blocking=True)
            bev = ensure_chw(bev)
            
            is_goal_now = is_goal_now.to(cfg.device, non_blocking=True).float()
            
            z_t, bev_pred_t = model(state)
            z_g, _ = model(goal)
            
            z_next, _ = target_model(next_state)
            z_g_target, _ = target_model(goal)

            temporal_dist = torch.norm(z_t - z_g, dim=-1)       # (B, )
            target_dist_val = torch.norm(z_next - z_g_target, dim=-1)
            
            mask = (1.0 - is_goal_now)
            reward = -1.0 * mask
            target_dist = cfg.gamma * mask * target_dist_val
            delta = reward - target_dist + temporal_dist

            loss_hilp = l2_expectile_loss(delta, cfg.expectile_tau)
            loss_bev = F.mse_loss(bev_pred_t, bev)
            loss = loss_hilp + (cfg.lambda_bev * loss_bev)
            
            total_loss += loss.item()
            total_loss_hilp += loss_hilp.item()
            total_loss_bev += loss_bev.item()
            num_batches += 1
            
            if ddp_utils.is_main_process() and i == 0:
                comparison = torch.cat([bev[0], bev_pred_t[0]], dim=2)
                save_path = os.path.join(ckpt_dir, "bev", f"epoch_{epoch:03d}.png")
                save_image(comparison, save_path)
        
        # Reduce validation losses across processes so all ranks share the same metrics
        loss_tensor = torch.tensor(
            [total_loss, total_loss_hilp, total_loss_bev, num_batches],
            device=cfg.device
        )
        if dist.is_initialized():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_loss_hilp, total_loss_bev, num_batches = loss_tensor.tolist()
        num_batches = max(1, int(num_batches))
        eval_loss['eval/loss'] = total_loss / num_batches
        eval_loss['eval/loss_hilp'] = total_loss_hilp / num_batches
        eval_loss['eval/loss_bev'] = total_loss_bev  / num_batches
    return eval_loss


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
    
    # Track whether this is the main process for logging/checkpointing
    main_process = ddp_utils.is_main_process()
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs*len(train_dataloader))
    
    if ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        del ckpt
    
    if main_process:
        print(f"[Train] {cfg.device} is used.")
        print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")
    
    best_loss = float(np.inf)
    best_train_loss = float(np.inf)
    global_step = 0
    early_stop_counter = 0
    for epoch in range(cfg.num_epochs):
        sampler = getattr(train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_loss_hilp = 0.0
        total_loss_bev = 0.0
        
        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader), 
            desc=f"[Train] Epoch {epoch}/{cfg.num_epochs}", 
            leave=True, 
            ncols=100,
            disable=not main_process
            )
        device_type = 'cuda' if 'cuda' in str(cfg.device) else 'cpu'
        
        model.train()
        
        for i, (state, next_state, goal, bev, is_goal_now) in pbar:
            for k, v in state.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                state[k] = v
            
            for k, v in next_state.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                next_state[k] = v
            
            for k, v in goal.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                goal[k] = v
            
            bev = bev.to(cfg.device, non_blocking=True)
            bev = ensure_chw(bev)
            is_goal_now = is_goal_now.to(cfg.device, non_blocking=True).float()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                # phi(s), phi(g)
                z_t, bev_pred_t = model(state)
                z_g, _ = model(goal)
                
                with torch.no_grad():
                    # target_phi(s'), target_phi(g)
                    z_next, _ = target_model(next_state)
                    z_g_target, _ = target_model(goal)
                
                temporal_dist = torch.norm(z_t - z_g, dim=-1)       # (B, )
                
                mask = (1.0 - is_goal_now)
                reward = -1.0 * mask
                
                target_dist_val = torch.norm(z_next - z_g_target, dim=-1)
                target_dist = cfg.gamma * mask * target_dist_val
                delta = reward - target_dist + temporal_dist
                
                loss_hilp = l2_expectile_loss(delta, cfg.expectile_tau)
                loss_bev = F.mse_loss(bev_pred_t, bev)
                loss = loss_hilp + (cfg.lambda_bev * loss_bev)
            
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            lr_scheduler.step()
            scaler.update()

            if (i % cfg.target_update_frequency) == 0:
                update_exponential_moving_average(target_model, model, cfg.tau)
            
            total_loss += loss.item()
            total_loss_hilp += loss_hilp.item()
            total_loss_bev += loss_bev.item()
            if main_process:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            if main_process and (i % cfg.log_frequency) == 0:
                step_log = {"train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/loss": loss.item(),
                            "train/loss_hilp": loss_hilp.item(),
                            "train/loss_bev": loss_bev.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "debug/temporal_dist_mean": float(temporal_dist.mean()),
                            "debug/target_dist_mean": float(target_dist.mean()),
                            "debug/delta_abs_mean": float(delta.abs().mean())
                            }
                step_log.update(_stats_dict("debug/state", state))
                step_log.update(_stats_dict("debug/next_state", next_state))
                step_log.update(_stats_dict("debug/goal", goal))
                step_log.update({"debug/has_nan_input": int(torch.isnan(state).any().item() or torch.isnan(next_state).any().item() or torch.isnan(goal).any().item())})
                
                if logger:
                    logger.log(step_log)
                if wb:
                    wandb.log(step_log, step=global_step)
            
            global_step += 1
        
        avg_loss = total_loss / len(train_dataloader)
        avg_loss_hilp = total_loss_hilp / len(train_dataloader)
        avg_loss_bev = total_loss_bev / len(train_dataloader)
        
        if main_process:
            step_log = {"train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/loss": avg_loss,
                        "train/loss_hilp": avg_loss_hilp,
                        "train/loss_bev": avg_loss_bev,
                        "train/lr": lr_scheduler.get_last_lr()[0]
                        }
            
            if logger:
                logger.log(step_log)
            if wb:
                wandb.log(step_log, step=global_step)
         
        if epoch % cfg.eval_frequency == 0:
            eval_loss_dict = eval(
                model=model, 
                target_model=target_model, 
                dataloader=val_dataloader, 
                epoch=epoch,
                ckpt_dir=checkpoint_dir,
                cfg=cfg,
                )
            
            eval_loss = eval_loss_dict['eval/loss']
            if main_process:
                print(f"[Validation] Loss: {eval_loss:.4f}")
                
                if logger:
                    logger.log(eval_loss_dict)
                if wb:
                    wandb.log(eval_loss_dict, step=global_step)
            
            if eval_loss <= best_loss:
                import gc
                gc.collect()
                
                if main_process:
                    print(f"Save best model of epoch {epoch} (loss={eval_loss:.4f})")
                    
                    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_loss_{eval_loss:.3f}.pt")
                    model_to_save = model.module if isinstance(model, DDP) else model
                    torch.save({
                        "epoch": epoch,
                        "hilbert_representation_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "loss": avg_loss,
                        "eval_loss": eval_loss
                    }, checkpoint_path)
                best_loss = eval_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
        
        if early_stop_counter > cfg.patience:
            if main_process:
                print(f"Early stopped at epoch {epoch} (best loss={best_loss:.4f})")
            break
    
    if main_process:
        print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")



@click.command()
@click.option("-c", "--config", default="train_rgb_hilp", type=str, help="Config to use")
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f"config/{config}.yaml")
    cfg = OmegaConf.load(CONFIG_FILE)
    
    # Initialize torch.distributed using torchrun environment variables if available
    ddp_enabled = ddp_utils.setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if cfg.resume:
        resum_conf = OmegaConf.load(os.path.join(os.getcwd(), f"data/outputs/rgb_hilp/{cfg.resume_ckpt_dir}/{config}.yaml"))
        cfg.data = resum_conf.data
        cfg.model = resum_conf.model
        cfg.train = resum_conf.train
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    seed_all(train_cfg.seed)
    # Use LOCAL_RANK to pick the GPU for this process
    device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    train_cfg.device = device_str
    
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    model = ViewAwareHilbertRepresentation(model_cfg)
    target_model = ViewAwareHilbertRepresentation(model_cfg)
    target_model.load_state_dict(model.state_dict())
    for p in target_model.parameters():
        p.requires_grad_(False)
    
    scaler = torch.amp.GradScaler(enabled=train_cfg.use_amp)
    
    ckpt = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"data/outputs/rgb_hilp/{cfg.resume_ckpt_dir}/*.pt")))
        ckpt = torch.load(ckpts[-1])
        model.load_state_dict(ckpt["hilbert_representation_state_dict"])
        target_model.load_state_dict(ckpt["hilbert_representation_state_dict"])
        if ddp_utils.is_main_process():
            print(f"Resume from {ckpts[-1]}")
    
    # Move models to the local GPU determined by LOCAL_RANK
    model = model.to(train_cfg.device)
    target_model = target_model.to(train_cfg.device)
    
    # Wrap the model with DistributedDataParallel for multi-GPU training
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    dataset = GoalRGBDataset(train_cfg.seed, data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    # Use DistributedSampler to shard data across processes
    train_sampler = DistributedSampler(train_dataset) if ddp_enabled else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp_enabled else None
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=data_cfg.train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=data_cfg.val_batch_size,
        shuffle=val_sampler is None,
        sampler=val_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory
    )
    
    checkpoint_dir = os.path.join(os.getcwd(), f"data/outputs/rgb_hilp/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
    if ddp_utils.is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
        os.makedirs(os.path.join(checkpoint_dir, "bev"), exist_ok=True)
    # Make sure all processes wait for checkpoint directory setup
    if ddp_enabled:
        dist.barrier()
    
    logger = JsonLogger(path=os.path.join(checkpoint_dir, "log.json")) if ddp_utils.is_main_process() else None
    if logger:
        logger.start()
    wb_enabled = cfg.wb and ddp_utils.is_main_process()
    if wb_enabled:
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
            wb=wb_enabled,
            checkpoint_dir=checkpoint_dir,
            cfg=train_cfg,
            logger=logger,
            )
    finally:
        if logger:
            logger.stop()
        # Clean up process group so torchrun exits cleanly
        ddp_utils.cleanup_distributed()


if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()
