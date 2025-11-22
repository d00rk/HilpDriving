"""
Training High-Level Policy Goal-conditioned pi(z|s, g) with pretrained hilbert representation model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Prevent h5py from aborting with file-lock errors when used in DataLoader workers
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import wandb
import datetime as dt
import click
from tqdm import tqdm
import copy
import glob
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.distributions import Normal

from model.hilp import HilbertRepresentation
from model.value_function import TwinQ, ValueFunction
from model.policy import GoalConditionedGaussianPolicy
from dataset.dataset import LatentGoalDataset
from utils.utils import asymmetric_l2_loss, update_exponential_moving_average, cosine_align_loss, get_lambda, compute_z, ensure_chw
from utils.seed_utils import seed_all
from utils.logger import JsonLogger, _stats_dict


EXP_ADV_MAX = 100.


def eval(
    hilbert,
    q_function,
    target_q_function,
    v_function,
    policy,
    dataloader,
    epoch,
    cfg,
    max_step
    ):
    
    q_function.eval()
    policy.eval()
    if v_function is not None:
        v_function.eval()
    
    eval_loss = {}
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        desc=f"[Validation] Epoch {epoch}/{cfg.num_epochs}", 
        leave=True, 
        ncols=100
        )
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
        total_loss = 0.0
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_policy_loss = 0.0
        
        lambda_align = get_lambda(cfg, epoch)
        
        for i, (state, z, next_state, goal, reward, terminal, _) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            z = z.to(cfg.device, non_blocking=True)
            next_state = next_state.to(cfg.device, non_blocking=True)
            goal = goal.to(cfg.device, non_blocking=True)
            # Reward/terminal come in as (B, 1); squeeze to (B,) to avoid broadcasting with 1D value outputs
            reward = reward.to(cfg.device, non_blocking=True).squeeze(-1)
            terminal = terminal.to(cfg.device, non_blocking=True).squeeze(-1)
            
            state = ensure_chw(state)
            next_state = ensure_chw(next_state)
            goal = ensure_chw(goal)
            
            q_target = target_q_function(state, z)
            next_v = v_function(next_state)
            v = v_function(state)
            adv = q_target - v
            v_loss = asymmetric_l2_loss(adv, cfg.tau)
            
            targets = reward + (1.0 - terminal) * cfg.discount * next_v.detach()
            q1, q2 = q_function.both(state, z)
            q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
            
            exp_adv = torch.exp(cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
            z_mu, z_logstd = policy(state, goal)
            
            dist = Normal(z_mu, z_logstd.exp())
            log_prob = dist.log_prob(z).sum(dim=-1)
            policy_loss = -(exp_adv * log_prob).mean()
            
            align_loss = 0.0
            if lambda_align > 0.0:
                z_star = compute_z(hilbert, state, goal)
                align_loss = cosine_align_loss(z_mu, z_star) * lambda_align
                policy_loss = policy_loss + align_loss
            
            loss = v_loss.item() + q_loss.item() + policy_loss.item()
            total_loss += loss
            total_q_loss += q_loss.item()
            total_v_loss += v_loss.item()
            total_policy_loss += policy_loss.item()
            
            if max_step is not None and (i >= max_step):
                break

        avg_loss = total_loss / len(dataloader)
        avg_q_loss = total_q_loss / len(dataloader)
        avg_v_loss = total_v_loss / len(dataloader)
        avg_policy_loss = total_policy_loss / len(dataloader)
        
        eval_loss["eval/loss"] = avg_loss
        eval_loss["eval/q_loss"] = avg_q_loss
        eval_loss["eval/policy_loss"] = avg_policy_loss
        eval_loss["eval/v_loss"] = avg_v_loss
        
        return eval_loss


def train(
    hilbert,
    q_function,
    v_function,
    policy, 
    scaler,
    train_dataloader, 
    val_dataloader,
    wb,
    ckpt,
    checkpoint_dir,
    cfg,
    logger,
    debug
    ):
    train_max_step = None
    eval_max_step = None
    if debug:
        train_max_step = 5
        eval_max_step = 5
        cfg.num_epochs = 2
    
    q_function = q_function.to(cfg.device)
    policy = policy.to(cfg.device)
    target_q_function = copy.deepcopy(q_function).requires_grad_(False).to(cfg.device)
    v_function = v_function.to(cfg.device)
    
    q_optimizer = torch.optim.AdamW(q_function.parameters(), lr=cfg.q_lr)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.policy_lr)
    v_optimizer = torch.optim.AdamW(v_function.parameters(), lr=cfg.v_lr)
    policy_lr_scheduler = CosineAnnealingLR(policy_optimizer, T_max=cfg.num_epochs*len(train_dataloader))
    
    if ckpt is not None:
        q_optimizer.load_state_dict(ckpt['q_optimizer_state_dict'])
        policy_optimizer.load_state_dict(ckpt['policy_optimizer_state_dict'])
        policy_lr_scheduler.load_state_dict(ckpt['policy_lr_scheduler_state_dict'])
        v_optimizer.load_state_dict(ckpt['v_optimizer_state_dict'])
        del ckpt
    
    print(f"[Train] {cfg.device} is used.")
    print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")
    
    best_loss = float(np.inf)
    global_step = 0
    early_stop_counter = 0
    for epoch in range(cfg.num_epochs):
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_policy_loss = 0.0
        total_loss = 0.0
        
        q_function.train()
        policy.train()
        v_function.train()
        
        lambda_align = get_lambda(cfg, epoch)
        
        pbar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f"[Epoch {epoch}/{cfg.num_epochs}]", 
            leave=True, 
            ncols=100
            )
        
        device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
        for i, (state, z, next_state, goal, reward, terminal, _) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            z = z.to(cfg.device, non_blocking=True)
            next_state = next_state.to(cfg.device, non_blocking=True)
            goal = goal.to(cfg.device, non_blocking=True)
            reward = reward.to(cfg.device, non_blocking=True).squeeze(-1)
            terminal = terminal.to(cfg.device, non_blocking=True).squeeze(-1)
            
            state = ensure_chw(state)
            next_state = ensure_chw(next_state)
            goal = ensure_chw(goal)

            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                with torch.no_grad():
                    q_target = target_q_function(state, z)
                    next_v = v_function(next_state)
                v = v_function(state)
                adv = q_target - v
                v_loss = asymmetric_l2_loss(adv, cfg.tau)
            v_optimizer.zero_grad(set_to_none=True)
            scaler.scale(v_loss).backward()
            nn.utils.clip_grad_norm_(v_function.parameters(), max_norm=5.0)
            scaler.step(v_optimizer)
            
            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                targets = reward + (1.0 - terminal) * cfg.discount * next_v.detach()
                q1, q2 = q_function.both(state, z)
                q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
            q_optimizer.zero_grad(set_to_none=True)
            scaler.scale(q_loss).backward()
            nn.utils.clip_grad_norm_(q_function.parameters(), max_norm=5.0)
            scaler.step(q_optimizer)
            
            if (i % cfg.target_update_frequency) == 0:
                update_exponential_moving_average(target_q_function, q_function, cfg.alpha)
            
            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                exp_adv = torch.exp(cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
                z_mu, z_logstd = policy(state, goal)
                dist = Normal(z_mu, z_logstd.exp())
                log_prob = dist.log_prob(z).sum(dim=-1)
                policy_loss = -(exp_adv * log_prob).mean()

                align_loss = 0.0
                if lambda_align > 0.0:
                    z_star = compute_z(hilbert, state, goal)
                    align_loss = cosine_align_loss(z_mu, z_star) * lambda_align
                    policy_loss = policy_loss + align_loss
            
            policy_optimizer.zero_grad(set_to_none=True)
            scaler.scale(policy_loss).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            scaler.step(policy_optimizer)
            policy_lr_scheduler.step()
            
            scaler.update()
            
            loss = v_loss.item() + q_loss.item() + policy_loss.item()
            total_loss += loss
            total_q_loss += q_loss.item()
            total_v_loss += v_loss.item()
            total_policy_loss += policy_loss.item()
            
            if (i % cfg.log_frequency) == 0:
                step_log = {
                    'train/epoch': epoch,
                    'train/global_step': global_step,
                    'train/loss': loss,
                    'train/q_loss': q_loss.item(),
                    'train/v_loss': v_loss.item(),
                    'train/policy_loss': policy_loss.item(),
                    'train/lr': policy_lr_scheduler.get_last_lr()[0],
                    'debug/has_nan_input': int(
                        torch.isnan(state).any().item() or torch.isnan(z).any().item() or torch.isnan(next_state).any().item() or torch.isnan(goal).any().item()
                    ),
                }
                step_log.update(_stats_dict("debug/state", state))
                step_log.update(_stats_dict("debug/next_state", next_state))
                step_log.update(_stats_dict("debug/z", z))
                
                logger.log(step_log)
                if wb:
                    wandb.log(step_log, step=global_step)
                
            global_step += 1
            pbar.set_postfix({"Total Loss": f"{loss:.4f}"})
            
            if (train_max_step is not None) and (i >= train_max_step):
                break

        avg_q_loss = total_q_loss / len(train_dataloader)
        avg_v_loss = total_v_loss / len(train_dataloader)
        avg_policy_loss = total_policy_loss / len(train_dataloader)
        avg_loss = total_loss / len(train_dataloader)
        
        log_dict = {"train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/loss": avg_loss,
                    "train/q_loss": avg_q_loss,
                    "train/v_loss": avg_v_loss,
                    "train/policy_loss": avg_policy_loss,
                    "train/lr": policy_lr_scheduler.get_last_lr()[0]}
        
        logger.log(log_dict)
        if wb:
            wandb.log(log_dict, step=global_step)
        
        if cfg.save_latest_ckpt:
            checkpoint_path = os.path.join(checkpoint_dir, f"latest.pt")
            torch.save({
                'epoch': epoch,
                'q_state_dict': q_function.state_dict(),
                'v_state_dict': v_function.state_dict(),
                'policy_state_dict': policy.state_dict(),
                'q_optimizer_state_dict': q_optimizer.state_dict(),
                'v_optimizer_state_dict': v_optimizer.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'policy_lr_scheduler_state_dict': policy_lr_scheduler.state_dict(),
            }, checkpoint_path)
        
        if epoch % cfg.eval_frequency == 0:
            eval_loss_dict = eval(
                hilbert=hilbert,
                q_function=q_function,
                target_q_function=target_q_function,
                v_function=v_function,
                policy=policy,
                dataloader=val_dataloader, 
                epoch=epoch,
                cfg=cfg,
                max_step=eval_max_step
                )
            
            print(f"[Validation] Loss: {eval_loss_dict['eval/loss']:.4f}")
            
            logger.log(eval_loss_dict)
            if wb:
                wandb.log(eval_loss_dict, step=global_step)
            
            eval_loss = eval_loss_dict['eval/loss']
            if eval_loss <= best_loss:
                import gc
                gc.collect()
                
                print(f"Save best model of epoch {epoch} (loss={eval_loss:.4f})")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_loss_{eval_loss:.3f}.pt")

                torch.save({
                    'epoch': epoch,
                    'q_state_dict': q_function.state_dict(),
                    'v_state_dict': v_function.state_dict(),
                    'policy_state_dict': policy.state_dict(),
                    'q_optimizer_state_dict': q_optimizer.state_dict(),
                    'v_optimizer_state_dict': v_optimizer.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                    'policy_lr_scheduler_state_dict': policy_lr_scheduler.state_dict(),
                    'loss': avg_loss,
                    'eval_loss': eval_loss 
                }, checkpoint_path)
                early_stop_counter = 0
                best_loss = eval_loss
            else:
                early_stop_counter += 1
        
        if early_stop_counter > cfg.patience:
            print(f"Early Stopped at epoch {epoch:02d} (best loss={best_loss:.4f})")
            break
    
    print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")


@click.command()
@click.option("-c", "--config", type=str, default='train_high_level_policy', required=True, help="config file name")
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    debug = cfg.debug
    
    if cfg.resume:
        resume_conf = OmegaConf.load(os.path.join(os.getcwd(), f'data/outputs/hilp_high_level/{cfg.resume_ckpt_dir}/{config}.yaml'))
        cfg.data = resume_conf.data
        cfg.model = resume_conf.model
        cfg.train = resume_conf.train
        del resume_conf
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    seed_all(train_cfg.seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # pretrained hilbert representation model
    HILP_DICT_PATH = os.path.join(os.getcwd(), f'data/outputs/hilp/{train_cfg.hilp_dir}/{train_cfg.hilp_dict_name}.pt')
    hilbert_representation = HilbertRepresentation(model_cfg)
    ckpt = torch.load(HILP_DICT_PATH)
    hilbert_representation.load_state_dict(ckpt['hilbert_representation_state_dict'])
    hilbert_representation.eval()
    hilbert_representation.to(train_cfg.device)
    for p in hilbert_representation.parameters():
        p.requires_grad_(False)
    # Use a CPU copy for data loading so DataLoader workers never touch CUDA.
    hilbert_representation_cpu = copy.deepcopy(hilbert_representation).cpu()
    hilbert_representation_cpu.eval()
    for p in hilbert_representation_cpu.parameters():
        p.requires_grad_(False)
    
    scaler = torch.amp.GradScaler(enabled=train_cfg.use_amp)
    
    dataset = LatentGoalDataset(train_cfg.seed, train_cfg.discount, hilbert_representation_cpu, data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()

    # In debug mode keep data loading single-threaded to avoid h5py worker aborts
    if debug:
        data_cfg.num_workers = 0
        data_cfg.pin_memory = False

    # Only create a multiprocessing context when workers are used
    mp_ctx = mp.get_context("spawn") if data_cfg.num_workers > 0 else None
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=data_cfg.train_batch_size, 
        shuffle=True, 
        num_workers=data_cfg.num_workers, 
        pin_memory=data_cfg.pin_memory,
        multiprocessing_context=mp_ctx,
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=data_cfg.val_batch_size, 
        shuffle=False, 
        num_workers=data_cfg.num_workers, 
        pin_memory=data_cfg.pin_memory,
        multiprocessing_context=mp_ctx,
        )
    
    # Create model
    q_func = TwinQ(model_cfg)
    v_func = ValueFunction(model_cfg)
    policy = GoalConditionedGaussianPolicy(model_cfg)
    q_func.initialize()
    v_func.initialize()
    policy.initialize()
    
    ckpt = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"data/outputs/hilp_high_level/{cfg.resume_ckpt_dir}/*.pt")))
        ckpt = torch.load(ckpts[-1])
        q_func.load_state_dict(ckpt['q_state_dict'])
        v_func.load_state_dict(ckpt['v_state_dict'])
        policy.load_state_dict(ckpt['policy_state_dict'])
        print(f"Resume from {ckpts[-1]}")
        
    checkpoint_dir = os.path.join(os.getcwd(), f"data/outputs/hilp_high_level/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Created output directory: {checkpoint_dir}.")
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
            hilbert=hilbert_representation,
            q_function=q_func, 
            v_function=v_func, 
            policy=policy, 
            scaler=scaler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            wb=cfg.wb,
            ckpt=ckpt,
            checkpoint_dir=checkpoint_dir,
            cfg=train_cfg,
            logger=logger,
            debug=debug
            )
    finally:
        logger.stop()

if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()
