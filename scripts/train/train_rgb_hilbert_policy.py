"""
Training Hilbert Foundation Policy (Low-Level Policy) pi(a|s, z) 
with pretrained view-aware hilbert representation model.
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.hilp import ViewAwareHilbertRepresentation
from model.value_function import MLPTwinQforHilbert, MLPValueFunctionforHilbert
from model.policy import MLPConditionedGaussianPolicy
from dataset.rgb_dataset import SubTrajRGBDataset
from utils.utils import asymmetric_l2_loss, update_exponential_moving_average, ensure_chw
from utils.seed_utils import seed_all
from utils.sampler import sample_latent_vectors
from utils.logger import JsonLogger, _stats_dict


EXP_ADV_MAX = 100.


class _NoOpLogger:
    def start(self):
        return None

    def stop(self):
        return None

    def log(self, *args, **kwargs):
        return None


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


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
    q_function.eval()
    policy.eval()
    v_function.eval()
    if _is_distributed() and dist.get_rank() != 0:
        return {}
    
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
            for k, v in state.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                state[k] = v
            
            for k, v in next_state.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                next_state[k] = v
                
            action = action.to(cfg.device, non_blocking=True)  
            terminal = terminal.to(cfg.device, non_blocking=True)
            
            B, S = action.shape[:2]
            
            state_flat = {k: v.view(-1, *v.shape[2:]) for k, v in state.items()}
            next_state_flat = {k: v.view(-1, *v.shape[2:]) for k, v in next_state.items()}
            
            phi_s, _ = model(state)
            phi_next_s, _ = model(next_state)
            
            action = action.view(B*S, -1)
            terminal = terminal.view(B*S)
            
            z = sample_latent_vectors(batch_size=B*S, latent_dim=latent_dim).to(cfg.device)
            
            dist = phi_next_s - phi_s
            intrinsic_reward = (dist * z).sum(dim=-1)
            
            q_target = target_q_function(phi_s, z, action)
            next_v = v_function(phi_next_s, z)
            v = v_function(phi_s, z)
            adv = q_target - v
            v_loss = asymmetric_l2_loss(adv, cfg.tau)
            
            targets = intrinsic_reward + cfg.skill_discount * (1.0 - terminal) * next_v.detach()
            q1, q2 = q_function.both(phi_s, z, action)
            q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
            
            exp_adv = torch.exp(cfg.skill_temperature * adv.detach()).clamp(max=EXP_ADV_MAX)
            a_mu, a_logstd = policy(phi_s, z)
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
    logger,
    is_distributed=False,
    is_main_process=True,
    local_rank=0,
    world_size=1
    ):
    
    model = model.to(cfg.device)
    model.eval()                # hilbert representation
    
    q_function = q_function.to(cfg.device)
    policy = policy.to(cfg.device)
    v_function = v_function.to(cfg.device)
    target_q_function = copy.deepcopy(q_function).requires_grad_(False).to(cfg.device)
    
    if is_distributed:
        q_function = DDP(q_function, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        v_function = DDP(v_function, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
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
    
    if is_main_process:
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
            leave=is_main_process, 
            ncols=100,
            disable=not is_main_process
            )
        
        q_function.train()
        policy.train()
        v_function.train()
        
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_policy_loss = 0.0
        total_loss = 0.0
        
        for i, (state, action, next_state, _, terminal, _) in pbar:
            for k, v in state.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                state[k] = v
            for k, v in next_state.items():
                v = v.to(cfg.device, non_blocking=True)
                v = ensure_chw(v)
                next_state[k] = v
            
            action = action.to(cfg.device, non_blocking=True) 
            terminal = terminal.to(cfg.device, non_blocking=True)
            
            B, S = action.shape[2:]
            action = action.view(B*S, -1)
            terminal = terminal.view(B*S)
            
            state_flat = {k: v.view(-1, *v.shape[2:]) for k, v in state.items()}
            next_state_flat = {k: v.view(-1, *v.shape[2:]) for k, v in next_state.items()}
            
            with torch.no_grad():
                phi_s, _ = model(state_flat)
                phi_next_s, _ = model(next_state_flat)
            
            z = sample_latent_vectors(batch_size=B, latent_dim=latent_dim).to(cfg.device)
            
            dist = phi_next_s - phi_s
            intrinsic_reward = (dist * z).sum(dim=-1)

            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                with torch.no_grad():
                    q_target = target_q_function(phi_s, z, action)
                    next_v = v_function(phi_next_s, z)
                v = v_function(phi_s, z)
                adv = q_target - v
                v_loss = asymmetric_l2_loss(adv, cfg.tau)
            
            v_optimizer.zero_grad(set_to_none=True)
            scaler.scale(v_loss).backward()
            nn.utils.clip_grad_norm_(v_function.parameters(), max_norm=5.0)
            scaler.step(v_optimizer)
            
            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                targets = intrinsic_reward + cfg.skill_discount * (1.0 - terminal) * next_v.detach()
                q1, q2 = q_function.both(phi_s, z, action)
                q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
            q_optimizer.zero_grad(set_to_none=True)
            scaler.scale(q_loss).backward()
            nn.utils.clip_grad_norm_(q_function.parameters(), max_norm=5.0)
            scaler.step(q_optimizer)
            
            if (i % cfg.target_update_frequency) == 0:
                update_exponential_moving_average(target_q_function, q_function, cfg.alpha)
            
            with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
                exp_adv = torch.exp(cfg.skill_temperature * adv.detach()).clamp(max=EXP_ADV_MAX)
                a_mu, a_logstd = policy(phi_s, z)
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
            
            if (i % cfg.log_frequency) == 0 and is_main_process:
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
        if is_distributed:
            metric_tensor = torch.tensor([avg_loss, avg_q_loss, avg_v_loss, avg_policy_loss], device=cfg.device)
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            metric_tensor /= world_size
            avg_loss, avg_q_loss, avg_v_loss, avg_policy_loss = metric_tensor.tolist()
        
        log_dict = {"train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/loss": avg_loss,
                    "train/q_loss": avg_q_loss,
                    "train/v_loss": avg_v_loss,
                    "train/policy_loss": avg_policy_loss,
                    "train/lr": policy_lr_scheduler.get_last_lr()[0]}
        
        logger.log(log_dict) if is_main_process else None
        if wb and is_main_process:
            wandb.log(log_dict, step=global_step)
        
        if cfg.save_latest_ckpt and is_main_process:
            checkpoint_path = os.path.join(checkpoint_dir, f"latest.pt")
            torch.save({
                'epoch': epoch,
                'q_state_dict': q_function.module.state_dict() if is_distributed else q_function.state_dict(),
                'v_state_dict': v_function.module.state_dict() if is_distributed else v_function.state_dict(),
                'policy_state_dict': policy.module.state_dict() if is_distributed else policy.state_dict(),
                'hilbert_representation_state_dict': model.state_dict(),
                'q_optimizer_state_dict': q_optimizer.state_dict(),
                'v_optimizer_state_dict': v_optimizer.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'policy_lr_scheduler_state_dict': policy_lr_scheduler.state_dict(),
            }, checkpoint_path)
        
        if epoch % cfg.eval_frequency == 0:
            eval_loss_dict = None
            if (not is_distributed) or is_main_process:
                eval_loss_dict = eval(model=model,
                                 q_function=q_function.module if is_distributed else q_function,
                                 target_q_function=target_q_function,
                                 v_function=v_function.module if is_distributed else v_function,
                                 policy=policy.module if is_distributed else policy,
                                 dataloader=val_dataloader, 
                                 latent_dim=latent_dim,
                                 epoch=epoch,
                                 cfg=cfg)
            
            if is_distributed:
                eval_tensor = torch.tensor(
                    [eval_loss_dict.get('eval/loss', 0.0) if eval_loss_dict else 0.0],
                    device=cfg.device,
                    dtype=torch.float32,
                )
                dist.broadcast(eval_tensor, src=0)
                eval_loss = eval_tensor.item()
            else:
                eval_loss = eval_loss_dict['eval/loss']

            if is_main_process and eval_loss_dict:
                print(f"[Validation] Loss: {eval_loss_dict['eval/loss']:.4f}")
                logger.log(eval_loss_dict)
                if wb:
                    wandb.log(eval_loss_dict, step=global_step)
            
            if eval_loss <= best_loss and is_main_process:
                import gc
                gc.collect()
                
                print(f"Save best model of epoch {epoch} (loss={eval_loss:.4f})")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_loss_{eval_loss:.3f}.pt")
                torch.save({
                    'epoch': epoch,
                    'q_state_dict': q_function.module.state_dict() if is_distributed else q_function.state_dict(),
                    'v_state_dict': v_function.module.state_dict() if is_distributed else v_function.state_dict(),
                    'policy_state_dict': policy.module.state_dict() if is_distributed else policy.state_dict(),
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
            elif is_main_process:
                early_stop_counter += 1

            if is_distributed:
                best_loss_tensor = torch.tensor([best_loss], device=cfg.device, dtype=torch.float32)
                counter_tensor = torch.tensor([early_stop_counter], device=cfg.device, dtype=torch.long)
                dist.broadcast(best_loss_tensor, src=0)
                dist.broadcast(counter_tensor, src=0)
                best_loss = best_loss_tensor.item()
                early_stop_counter = int(counter_tensor.item())
                dist.barrier()
        
        stop_training = early_stop_counter > cfg.patience
        if is_distributed:
            stop_tensor = torch.tensor([int(stop_training)], device=cfg.device, dtype=torch.long)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            if is_main_process:
                print(f"Early Stopped at epoch {epoch:02d} (best loss={best_loss:.4f})")
            break
    
    if is_main_process:
        print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")



@click.command()
@click.option("-c", "--config", type=str, default='train_rgb_hilbert_policy', required=True, help="config file name")
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_conf = OmegaConf.load(os.path.join(os.getcwd(), f"data/outputs/rgb_hilbert_policy/{cfg.resume_ckpt_dir}/{config}.yaml"))
        cfg.data = resume_conf.data
        cfg.model = resume_conf.model
        cfg.train = resume_conf.train
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # pretrained hilbert representation model
    HILP_DICT_PATH = os.path.join(os.getcwd(), f"data/outputs/rgb_hilp/{train_cfg.hilp_dir}/{train_cfg.hilp_dict_name}.pt")
    hilbert_representation = ViewAwareHilbertRepresentation(model_cfg)
    ckpt = torch.load(HILP_DICT_PATH)
    hilbert_representation.load_state_dict(ckpt['hilbert_representation_state_dict'])
    hilbert_representation.eval()
    for p in hilbert_representation.parameters():
        p.requires_grad_(False)
    
    scaler = torch.amp.GradScaler(enabled=train_cfg.use_amp)
    
    dataset = SubTrajRGBDataset(train_cfg.seed, data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        is_distributed = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        train_cfg.device = f"cuda:{local_rank}"
    
    seed_all(train_cfg.seed + rank)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=data_cfg.train_batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=data_cfg.num_workers, 
        pin_memory=data_cfg.pin_memory
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=data_cfg.val_batch_size, 
        shuffle=False, 
        num_workers=data_cfg.num_workers, 
        pin_memory=data_cfg.pin_memory
        )
    
    q_func = MLPTwinQforHilbert(model_cfg)
    policy = MLPConditionedGaussianPolicy(model_cfg)
    v_func = MLPValueFunctionforHilbert(model_cfg)
    q_func.initialize()
    policy.initialize()
    v_func.initialize()
    
    ckpt = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"data/outputs/rgb_hilbert_policy/{cfg.resume_ckpt_dir}/*.pt")))
        ckpt = torch.load(ckpts[-1])
        q_func.load_state_dict(ckpt['q_state_dict'])
        v_func.load_state_dict(ckpt['v_state_dict'])
        policy.load_state_dict(ckpt['policy_state_dict'])
    
    checkpoint_dir = os.path.join(os.getcwd(), f"data/outputs/rgb_hilbert_policy/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created output directory: {checkpoint_dir}.")
        OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
    
    logger = JsonLogger(path=os.path.join(checkpoint_dir, "log.json")) if rank == 0 else _NoOpLogger()
    logger.start() if rank == 0 else None
    if cfg.wb and rank == 0:
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
            logger=logger,
            is_distributed=is_distributed,
            is_main_process=(rank == 0),
            local_rank=local_rank,
            world_size=world_size
            )
    finally:
        logger.stop() if rank == 0 else None
        if is_distributed:
            dist.destroy_process_group()
    

if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()
