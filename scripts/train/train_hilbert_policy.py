"""
Training Hilbert Foundation Policy (Low-Level Policy) pi(a|s, z) with pretrained hilbert representation model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import wandb
import datetime as dt
import numpy as np
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
from model.value_function import TwinQforHilbert, ValueFunctionforHilbert
from model.policy import ConditionedGaussianPolicy
from dataset.dataset import SubTrajDataset
from utils.utils import asymmetric_l2_loss, update_exponential_moving_average, ensure_chw
from utils.seed_utils import seed_all
from utils.sampler import sample_latent_vectors
from utils.logger import JsonLogger, _stats_dict


EXP_ADV_MAX = 100.


def eval(
    model, 
    q_function,
    target_q_function,
    v_function,
    policy, 
    dataloader,
    latent_dim,
    epoch,
    cfg
    ):
    torch.cuda.empty_cache()
    
    q_function.eval()
    policy.eval()
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
    with torch.inference_mode(), torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
        total_loss = 0.0
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_policy_loss = 0.0
        for i, (state, action, next_state, _, terminal, _) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            action = action.to(cfg.device, non_blocking=True) 
            next_state = next_state.to(cfg.device, non_blocking=True) 
            terminal = terminal.to(cfg.device, non_blocking=True)
            
            # Keep state/action/next_state lengths in sync; trim to the shortest to avoid shape mismatches
            seq_len = min(state.shape[1], action.shape[1], next_state.shape[1], terminal.shape[1])
            state = state[:, :seq_len].contiguous()
            action = action[:, :seq_len].contiguous()
            next_state = next_state[:, :seq_len].contiguous()
            terminal = terminal[:, :seq_len].contiguous()
            state = ensure_chw(state)
            next_state = ensure_chw(next_state)
            
            B, S, C, H, W = state.shape
            
            z = sample_latent_vectors(batch_size=B, latent_dim=latent_dim)
            z_expand = z.unsqueeze(1).expand(B, S, latent_dim).contiguous().view(B*S, latent_dim)
            z_expand = z_expand.to(cfg.device)
            
            state = state.view(B*S, C, H, W)
            action = action.view(B*S, -1)
            next_state = next_state.view(B*S, C, H, W)
            terminal = terminal.view(B*S)
            
            phi_s = model(state)
            phi_next_s = model(next_state)
            intrinsic_reward = ((phi_next_s - phi_s) * z_expand).sum(dim=-1)
            
            q_target = target_q_function(state, z_expand, action)
            next_v = v_function(next_state, z_expand)
            v = v_function(state, z_expand)
            adv = q_target - v
            v_loss = asymmetric_l2_loss(adv, cfg.tau)
            
            targets = intrinsic_reward + cfg.skill_discount * (1.0 - terminal) * next_v.detach()
            q1, q2 = q_function.both(state, z_expand, action)
            q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
            
            exp_adv = torch.exp(cfg.skill_temperature * adv.detach()).clamp(max=EXP_ADV_MAX)
            a_mu, a_logstd = policy(state, z_expand)
            dist = Normal(a_mu, a_logstd.exp())
            log_prob = dist.log_prob(action).sum(dim=-1)
            policy_loss = -(exp_adv * log_prob).mean()
            
            loss = v_loss.item() + q_loss.item() + policy_loss.item()
            total_loss += loss
            total_q_loss += q_loss.item()
            total_v_loss += v_loss.item()
            total_policy_loss += policy_loss.item()
        
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
    model, 
    q_function,
    v_function,
    policy,
    scaler,
    train_dataloader, 
    val_dataloader,  
    latent_dim,
    wb,
    ckpt,
    checkpoint_dir,
    cfg,
    logger
    ):
    
    model = model.to(cfg.device)
    model.eval()                # hilbert representation
    
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
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    for epoch in range(cfg.num_epochs):
        sampler = getattr(train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        
        pbar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f"[Train] Epoch {epoch}/{cfg.num_epochs}", 
            leave=True, 
            ncols=100
            )
        
        q_function.train()
        policy.train()
        v_function.train()
        
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_policy_loss = 0.0
        total_loss = 0.0
        
        for i, (state, action, next_state, _, terminal, _) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            action = action.to(cfg.device, non_blocking=True) 
            next_state = next_state.to(cfg.device, non_blocking=True) 
            terminal = terminal.to(cfg.device, non_blocking=True)
            
            # Keep state/action/next_state lengths in sync; trim to the shortest to avoid shape mismatches
            seq_len = min(state.shape[1], action.shape[1], next_state.shape[1], terminal.shape[1])
            state = state[:, :seq_len].contiguous()
            action = action[:, :seq_len].contiguous()
            next_state = next_state[:, :seq_len].contiguous()
            terminal = terminal[:, :seq_len].contiguous()
            state = ensure_chw(state)
            next_state = ensure_chw(next_state)
            
            B, S, C, H, W = state.shape
            
            z = sample_latent_vectors(batch_size=B, latent_dim=latent_dim)
            z_expand = z.unsqueeze(1).expand(B, S, latent_dim).contiguous().view(B*S, latent_dim)
            z_expand = z_expand.to(cfg.device)
            
            state = state.view(B*S, C, H, W)
            action = action.view(B*S, -1)
            next_state = next_state.view(B*S, C, H, W)
            terminal = terminal.view(B*S)
            
            with torch.no_grad():
                phi_s = model(state)
                phi_next_s = model(next_state)
            dist = phi_next_s - phi_s
            intrinsic_reward = (dist * z_expand).sum(dim=-1)

            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                with torch.no_grad():
                    q_target = target_q_function(state, z_expand, action)
                    next_v = v_function(next_state, z_expand)
                v = v_function(state, z_expand)
                adv = q_target - v
                v_loss = asymmetric_l2_loss(adv, cfg.tau)
            
            v_optimizer.zero_grad(set_to_none=True)
            scaler.scale(v_loss).backward()
            nn.utils.clip_grad_norm_(v_function.parameters(), max_norm=5.0)
            scaler.step(v_optimizer)
            
            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                targets = intrinsic_reward + cfg.skill_discount * (1.0 - terminal) * next_v.detach()
                q1, q2 = q_function.both(state, z_expand, action)
                q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
            q_optimizer.zero_grad(set_to_none=True)
            scaler.scale(q_loss).backward()
            nn.utils.clip_grad_norm_(q_function.parameters(), max_norm=5.0)
            scaler.step(q_optimizer)
            
            if ( i % cfg.target_update_frequency) == 0:
                update_exponential_moving_average(target_q_function, q_function, cfg.alpha)
            
            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                exp_adv = torch.exp(cfg.skill_temperature * adv.detach()).clamp(max=EXP_ADV_MAX)
                a_mu, a_logstd = policy(state, z_expand)
                dist = Normal(a_mu, a_logstd.exp())
                log_prob = dist.log_prob(action).sum(dim=-1)
                policy_loss = -(exp_adv * log_prob).mean()
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
                        torch.isnan(state).any().item() or torch.isnan(next_state).any().item()
                    ),
                    'debug/disp': dist.mean.detach().mean().item(),
                    'debug/intrinsic_reward': float(intrinsic_reward.mean())
                }
                step_log.update(_stats_dict("debug/state", state))
                step_log.update(_stats_dict("debug/next_state", next_state))
                
                logger.log(step_log)
                if wb:
                    wandb.log(step_log, step=global_step)
            
            global_step += 1
            pbar.set_postfix({"Total Loss": f"{loss:.4f}"})

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
                'hilbert_representation_state_dict': model.state_dict(),
                'q_optimizer_state_dict': q_optimizer.state_dict(),
                'v_optimizer_state_dict': v_optimizer.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'policy_lr_scheduler_state_dict': policy_lr_scheduler.state_dict(),
            }, checkpoint_path)
        
        if epoch % cfg.eval_frequency == 0:
            eval_loss_dict = eval(model=model,
                             q_function=q_function,
                             target_q_function=target_q_function,
                             v_function=v_function,
                             policy=policy,
                             dataloader=val_dataloader, 
                             latent_dim=latent_dim,
                             epoch=epoch,
                             cfg=cfg)
            
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
                    'hilbert_representation_state_dict': model.state_dict(),
                    'q_optimizer_state_dict': q_optimizer.state_dict(),
                    'v_optimizer_state_dict': v_optimizer.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                    'policy_lr_scheduler_state_dict': policy_lr_scheduler.state_dict(),
                    'loss': avg_loss,
                    'eval_loss': eval_loss  
                }, checkpoint_path)
                best_loss = eval_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
        
        if early_stop_counter > cfg.patience:
            print(f"Early Stopped at epoch {epoch:02d} (best loss={best_loss:.4f})")
            break
    
    print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")



@click.command()
@click.option("-c", "--config", type=str, default='train_hilbert_policy', required=True, help="config file name")
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_conf = OmegaConf.load(os.path.join(os.getcwd(), f"data/outputs/hilbert_policy/{cfg.resume_ckpt_dir}/{config}.yaml"))
        cfg.data = resume_conf.data
        cfg.model = resume_conf.model
        cfg.train = resume_conf.train
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    seed_all(train_cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # pretrained hilbert representation model
    HILP_DICT_PATH = os.path.join(os.getcwd(), f"data/outputs/hilp/{train_cfg.hilp_dir}/{train_cfg.hilp_dict_name}.pt")
    hilbert_representation = HilbertRepresentation(model_cfg)
    ckpt = torch.load(HILP_DICT_PATH)
    hilbert_representation.load_state_dict(ckpt['hilbert_representation_state_dict'])
    hilbert_representation.eval()
    for p in hilbert_representation.parameters():
        p.requires_grad_(False)
    
    scaler = torch.amp.GradScaler(enabled=train_cfg.use_amp)
    
    dataset = SubTrajDataset(train_cfg.seed, data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    use_cuda = train_cfg.device.startswith("cuda") and torch.cuda.is_available()
    pin_memory = bool(data_cfg.pin_memory and use_cuda)  # pinning on CPU-only runs raises CUDA errors
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=data_cfg.train_batch_size, 
        shuffle=True, 
        num_workers=data_cfg.num_workers, 
        pin_memory=pin_memory
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=data_cfg.val_batch_size, 
        shuffle=False, 
        num_workers=data_cfg.num_workers, 
        pin_memory=pin_memory,
        )
    
    q_func = TwinQforHilbert(model_cfg)
    policy = ConditionedGaussianPolicy(model_cfg)
    v_func = ValueFunctionforHilbert(model_cfg)
    q_func.initialize()
    policy.initialize()
    v_func.initialize()
    
    ckpt = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"data/outputs/hilbert_policy/{cfg.resume_ckpt_dir}/*.pt")))
        ckpt = torch.load(ckpts[-1])
        q_func.load_state_dict(ckpt['q_state_dict'])
        v_func.load_state_dict(ckpt['v_state_dict'])
        policy.load_state_dict(ckpt['policy_state_dict'])
    
    checkpoint_dir = os.path.join(os.getcwd(), f"data/outputs/hilbert_policy/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
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
            model=hilbert_representation, 
            q_function=q_func, 
            v_function=v_func, 
            policy=policy,
            scaler=scaler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            latent_dim=model_cfg.latent_dim,
            wb=cfg.wb,
            ckpt=ckpt,
            checkpoint_dir=checkpoint_dir,
            cfg=train_cfg,
            logger=logger
            )
    finally:
        logger.stop()
    

if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()
