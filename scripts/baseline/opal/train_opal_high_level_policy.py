import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Normal
import torch.multiprocessing as mp
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import wandb
import datetime as dt
import click
from tqdm import tqdm
import copy
import glob
from omegaconf import OmegaConf

from model.opal import Encoder
from model.value_function import TwinQ, ValueFunction
from model.policy import GaussianPolicy
from dataset.dataset import LatentDataset
from utils.seed_utils import seed_all
from utils.utils import asymmetric_l2_loss, update_exponential_moving_average, log_sum_exp, get_reward_statics
from utils.logger import JsonLogger
from utils.ddp import is_dist_avail_and_initialized, get_rank, is_main_process, setup_distributed, cleanup_distributed, unwrap
from utils.multiprocessing import _worker_init_fn


EXP_ADV_MAX = 100.0


def eval(
    q_function,
    target_q_function,
    v_function,
    policy,
    dataloader,
    epoch,
    cfg,
    ):
    q_function = unwrap(q_function)
    policy = unwrap(policy)
    if v_function is not None:
        v_function = unwrap(v_function)
    
    q_function.eval()
    policy.eval()
    if v_function is not None:
        v_function.eval()
    
    total_loss = 0.0
    total_q_loss = 0.0
    total_policy_loss = 0.0
    total_v_loss = 0.0 if cfg.algo == 'iql' else None
    num_batches = 0
    
    eval_loss = {}
    pbar = enumerate(dataloader)
    if is_main_process():
        pbar = tqdm(
            pbar, 
            total=len(dataloader), 
            desc=f"[Validation] Epoch {epoch}/{cfg.num_epochs}", 
            leave=False, 
            ncols=100
            )
    
    for i, (state, z, next_state, reward, terminal, _) in pbar:
        state = state.to(cfg.device, non_blocking=True)
        z = z.to(cfg.device, non_blocking=True)
        next_state = next_state.to(cfg.device, non_blocking=True)
        reward = reward.to(cfg.device, non_blocking=True)
        terminal = terminal.to(cfg.device, non_blocking=True)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.use_amp):
            if cfg.algo == 'iql':
                q_target = target_q_function(state, z)
                next_v = v_function(next_state)
                v = v_function(state)
                adv = q_target - v
                v_loss = asymmetric_l2_loss(adv, cfg.tau)
                
                targets = reward (1.0 - terminal) * cfg.discount * next_v.detach()
                q1, q2 = q_function.both(state, z)
                q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
                
                exp_adv = torch.exp(cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
                z_mu, z_logstd = policy(state)
                dist = Normal(z_mu, z_logstd.exp())
                log_prob = dist.log_prob(z).sum(dim=-1)
                policy_loss = - (exp_adv * log_prob).mean()
                
                loss = v_loss.item() + q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += q_loss.item()
                total_v_loss += v_loss.item()
                total_policy_loss += policy_loss.item()
                num_batches += 1
            elif cfg.algo == 'cql':
                z_next_mu, z_next_logstd = policy(next_state)
                dist_next = Normal(z_next_mu, z_next_logstd.exp())
                z_next_pred = dist_next.rsample()
                logp_next = dist_next.log_prob(z_next_pred).sum(dim=-1)
                
                q_next_target = target_q_function(next_state, z_next_pred)
                target_q = reward + (1.0 - terminal) * cfg.discount * (q_next_target - cfg.alpha_entropy * logp_next)
                
                q1, q2 = q_function.both(state, z)
                Q_loss = sum(F.mse_loss(q, target_q) for q in [q1, q2]) / 2
                
                B, Z = z.shape
                random_z = torch.rand(B, cfg.num_random, Z, device=cfg.device)
                random_z = random_z * (cfg.high - cfg.low) + cfg.low
                
                obs_expand = state.unsqueeze(1).repeat(1, cfg.num_random, 1, 1, 1)
                obs_expand = obs_expand.view(B*cfg.num_random, *state.shape[1:])
                random_z = random_z.view(B*cfg.num_random, Z)
                
                q1_random, q2_random = q_function.both(obs_expand, random_z)
                q1_random = q1_random.view(B, cfg.num_random)
                q2_random = q2_random.view(B, cfg.num_random)
                
                z_mu, z_logstd = policy(state)
                dist = Normal(z_mu, z_logstd.exp())
                z_pi = dist.rsample()
                q1_policy, q2_policy = q_function.both(state, z_pi)
                q_new = torch.min(q1_policy, q2_policy)
                
                q1_cat = torch.cat([q1_random, q1_policy.unsqueeze(1)], dim=1)
                q2_cat = torch.cat([q2_random, q2_policy.unsqueeze(1)], dim=1)
                
                lse_q1 = log_sum_exp(q1_cat / cfg.temperature, dim=1)
                cql_q1 = (lse_q1 * cfg.temperature - q1).mean()
                lse_q2 = log_sum_exp(q2_cat / cfg.temperature, dim=1)
                cql_q2 = (lse_q2 * cfg.temperature - q2).mean()
                
                cql_loss_total = cfg.cql_alpha * (cql_q1 + cql_q2)
                q_loss = Q_loss + cql_loss_total
                
                policy_loss = (cfg.alpha_entropy * log_prob - q_new).mean()
                
                loss = q_loss + policy_loss
                total_loss += loss
                total_q_loss += q_loss.item()
                total_policy_loss += policy_loss.item()
                num_batches += 1
            
        device0 = cfg.device
        if cfg.algo == 'iql':
            t = torch.tensor([total_loss, total_q_loss, total_v_loss, total_policy_loss, num_batches], dtype=torch.float64, device=device0)
        elif cfg.algo == 'cql':
            t = torch.tensor([total_loss, total_q_loss, total_policy_loss, num_batches], dtype=torch.float64, device=device0)
            
        if is_dist_avail_and_initialized():
            distributed.all_reduce(t, op=distributed.ReduceOp.SUM)
        
        if cfg.algo == 'iql':
            sum_loss, sum_q, sum_v, sum_policy, sum_batches = t.tolist()
            if sum_batches == 0:
                eval_loss['eval/loss'] = 0.0
                eval_loss['eval/q_loss'] = 0.0
                eval_loss['eval/policy_loss'] = 0.0
                eval_loss['eval/v_loss'] = 0.0
                return eval_loss
            eval_loss['eval/loss'] = sum_loss/sum_batches
            eval_loss['eval/q_loss'] = sum_q/sum_batches
            eval_loss['eval/v_loss'] = sum_v/sum_batches
            eval_loss['eval/policy_loss'] = sum_policy/sum_batches
            return eval_loss
        elif cfg.algo == 'cql':
            sum_loss, sum_q, sum_policy, sum_batches = t.tolist()
            if sum_batches == 0:
                eval_loss['eval/loss'] = 0.0
                eval_loss['eval/q_loss'] = 0.0
                eval_loss['eval/policy_loss'] = 0.0
                return eval_loss
            eval_loss['eval/loss'] = sum_loss/sum_batches
            eval_loss['eval/q_loss'] = sum_q/sum_batches
            eval_loss['eval/policy_loss'] = sum_policy/sum_batches
            return eval_loss


def train(
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
    ):
    device = cfg.device
    
    target_q_function = copy.deepcopy(q_function).requires_grad_(False).to(device)
    
    if is_dist_avail_and_initialized():
        q_function = DDP(q_function.to(device), device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), find_unused_parameters=False)
        policy = DDP(policy.to(device), device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), find_unused_parameters=False)
        if v_function is not None:
            v_function = DDP(v_function.to(device), device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), find_unused_parameters=False)
    else:
        q_function = q_function.to(device)
        policy = policy.to(device)
        if v_function is not None:
            v_function = v_function.to(device)
    
    q_optimizer = torch.optim.AdamW(q_function.parameters(), lr=cfg.q_lr)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.policy_lr)
    v_optimizer = torch.optim.AdamW(v_function.parameters(), lr=cfg.v_lr) if v_function is not None else None
    policy_lr_scheduler = CosineAnnealingLR(policy_optimizer, T_max=cfg.num_epochs*len(train_dataloader))
    
    if ckpt is not None:
        q_optimizer.load_state_dict(ckpt['q_optimizer_state_dict'])
        policy_optimizer.load_state_dict(ckpt['policy_optimizer_state_dict'])
        policy_lr_scheduler.load_state_dict(ckpt['policy_lr_scheduler_state_dict'])
        if v_optimizer is not None:
            v_optimizer.load_state_dict(ckpt['v_optimizer_state_dict'])
        del ckpt
    
    if is_main_process():
        print(f"[Torch] {device} is used.")
        print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")
    
    best_loss = float(np.inf)
    global_step = 0
    
    train_sampler = train_dataloader.sampler if isinstance(train_dataloader.sampler, DistributedSampler) else None
    val_sampler = val_dataloader.sampler if isinstance(val_dataloader.sampler, DistributedSampler) else None
    
    for epoch in range(cfg.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_q_loss = 0.0
        total_v_loss = 0.0 if cfg.algo == 'iql' else None
        total_policy_loss = 0.0

        q_function.train()
        policy.train()
        if v_function is not None:
            v_function.train()
        
        pbar = enumerate(train_dataloader)
        if is_main_process():
            pbar = tqdm(
                pbar, 
                total=len(train_dataloader), 
                desc=f"[Epoch {epoch}/{cfg.num_epochs}]", 
                leave=True, 
                ncols=100
                )
        
        for i, (state, z, next_state, reward, terminal, _) in pbar:
            state = state.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            next_state = next_state.to(device, non_blocking=True)
            reward = reward.to(device, non_blocking=True)
            terminal = terminal.to(device, non_blocking=True)
            
            if cfg.algo == 'iql':
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    with torch.no_grad():
                        q_target = target_q_function(state, z)
                        next_v = v_function(next_state)
                    v = v_function(state)
                    adv = q_target - v
                    v_loss = asymmetric_l2_loss(adv, cfg.tau)
                v_optimizer.zero_grad(set_to_none=True)
                scaler.scale(v_loss).backward()
                scaler.step(v_optimizer)
                
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    targets = reward + (1.0 - terminal) * cfg.discount * next_v.detach()
                    q1, q2 = q_function.both(state, z)
                    q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
                q_optimizer.zero_grad(set_to_none=True)
                scaler.scale(q_loss).backward()
                scaler.step(q_optimizer)
                
                update_exponential_moving_average(target_q_function, q_function, cfg.alpha)
                
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    exp_adv = torch.exp(cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
                    z_mu, z_logstd = policy(state)
                    dist = Normal(z_mu, z_logstd.exp())
                    log_prob = dist.log_prob(z).sum(dim=-1)
                    policy_loss = -(exp_adv * log_prob).mean()
                policy_optimizer.zero_grad(set_to_none=True)
                scaler.scale(policy_loss).backward()
                scaler.step(policy_optimizer)
                policy_lr_scheduler.step()
                
                scaler.update()
                
                loss = v_loss.item() + q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += q_loss.item()
                total_v_loss += v_loss.item()
                total_policy_loss += policy_loss.item()
            
            elif cfg.algo == 'cql':
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    with torch.no_grad():
                        z_next_mu, z_next_logstd = policy(next_state)
                        dist_next = Normal(z_next_mu, z_next_logstd.exp())
                        z_next_pred = dist_next.rsample()
                        logp_next = dist_next.log_prob(z_next_pred).sum(dim=-1)
                        q_next_target = target_q_function(next_state, z_next_pred)
                        target_q = reward + (1.0 - terminal) * cfg.discount * (q_next_target - cfg.alpha_entropy * logp_next)
                    
                    q1, q2 = q_function.both(state, z)
                    Q_loss = sum(F.mse_loss(q, target_q) for q in [q1, q2]) / 2
                    
                    # conservative term: random + policy + next-policy
                    batch_size, z_dim = z.shape
                    random_z = torch.rand(batch_size, cfg.num_random, z_dim, device=cfg.device)
                    random_z = random_z * (cfg.high - cfg.low) + cfg.low
                    
                    obs_expand = state.unsqueeze(1).repeat(1, cfg.num_random, 1, 1, 1)
                    obs_expand = obs_expand.view(batch_size * cfg.num_random, *state.shape[1:])
                    random_z = random_z.view(batch_size * cfg.num_random, z_dim)
                    
                    q1_random, q2_random = q_function.both(obs_expand, random_z)
                    q1_random = q1_random.view(batch_size, cfg.num_random)
                    q2_random = q2_random.view(batch_size, cfg.num_random)
                    
                    z_mu, z_logstd = policy(state)
                    dist = Normal(z_mu, z_logstd.exp())
                    z_pi = dist.rsample()
                    log_prob = dist.log_prob(z_pi).sum(dim=-1)
                    q1_policy, q2_policy = q_function.both(state, z_pi)
                    q_new = torch.min(q1_policy, q2_policy)
                    
                    q1_cat = torch.cat([q1_random, q1_policy.unsqueeze(1)], dim=1)
                    q2_cat = torch.cat([q2_random, q2_policy.unsqueeze(1)], dim=1)
                    
                    lse_q1 = log_sum_exp(q1_cat / cfg.temperature, dim=1)
                    cql_q1 = (lse_q1 * cfg.temperature - q1).mean()
                    lse_q2 = log_sum_exp(q2_cat/cfg.temperature, dim=1)
                    cql_q2 = (lse_q2 * cfg.temperature - q2).mean()
                    
                    cql_loss_total = cfg.cql_alpha * (cql_q1 + cql_q2)
                    q_loss = Q_loss + cql_loss_total

                # q function update
                q_optimizer.zero_grad(set_to_none=True)
                scaler.scale(q_loss).backward()
                scaler.step(q_optimizer)
                
                update_exponential_moving_average(target_q_function, q_function, cfg.alpha)
                
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    # policy update           
                    policy_loss = (cfg.alpha_entropy * log_prob - q_new).mean()
                policy_optimizer.zero_grad(set_to_none=True)
                scaler.scale(policy_loss).backward()
                scaler.step(policy_optimizer)
                policy_lr_scheduler.step()
                
                scaler.update()
                
                loss = q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += q_loss.item()
                total_policy_loss += policy_loss.item()
            else:
                raise ValueError(f"Invalid algorithm: {cfg.algo}")
            
            if is_main_process() and (i % cfg.log_frequency == 0):
                step_log = {
                    'train/epoch': epoch,
                    'train/global_step': global_step,
                    'train/loss': loss,
                    'train/q_loss': q_loss.item(),
                    'train/policy_loss': policy_loss.item(),
                    'train/lr': policy_lr_scheduler.get_last_lr()[0]
                }
                if v_function is not None:
                    step_log['train/v_loss'] = v_loss.item()
                logger.log(step_log)
                if wb:
                    wandb.log(step_log, step=global_step)
                if isinstance(pbar, tqdm):
                    pbar.set_postfix({"Total Loss": f"{loss:.4f}"})
            global_step += 1
                
        t = torch.tensor([total_loss, total_q_loss, total_policy_loss, len(train_dataloader)], dtype=torch.float64, device=device)
        
        if cfg.algo == "iql":
            t = torch.tensor([total_loss, total_q_loss, total_v_loss, total_policy_loss, len(train_dataloader)], dtype=torch.float64, device=device)
        
        if is_dist_avail_and_initialized():
            distributed.all_reduce(t, op=distributed.ReduceOp.SUM)
        
        if cfg.algo == 'iql':
            sum_loss, sum_q_loss, sum_v_loss, sum_policy_loss, sum_batches = t.tolist()
        else:
            sum_loss, sum_q_loss, sum_policy_loss, sum_batches = t.tolist()
        
        if is_main_process():
            avg_loss = sum_loss / max(1, sum_batches)
            avg_q_loss = sum_q_loss / max(1, sum_batches)
            avg_policy_loss = sum_policy_loss / max(1, sum_batches)
            
            step_log = {
                'train/epoch': epoch,
                'train/global_step': global_step,
                'train/loss': avg_loss,
                'train/q_loss': avg_q_loss,
                'train/policy_loss': avg_policy_loss,
            }
            if cfg.algo == 'iql':
                avg_v_loss = sum_v_loss / max(1, sum_batches)
                step_log['train/v_loss'] = avg_v_loss
            logger.log(step_log)
            if wb:
                wandb.log(step_log, step=global_step)
        
        if epoch % cfg.eval_frequency == 0:
            eval_loss_dict = eval(
                q_function=q_function,
                v_function=v_function,
                policy=policy,
                dataloader=val_dataloader,
                epoch=epoch,
                cfg=cfg,
                device=device
            )
            
            if is_main_process():
                logger.log(eval_loss_dict)
                if wb:
                    wandb.log(eval_loss_dict, step=global_step)
                
                eval_loss = eval_loss_dict['eval/loss']
                if eval_loss <= best_loss:
                    print(f"Save best model of epoch {epoch} (loss={eval_loss:.4f})")
                    
                    checkpoint_path = os.path.join(checkpoint_dir, f"{cfg.algo}_epoch_{epoch:04d}_loss_{eval_loss:.3f}.pt")
                    
                    q_sd = q_function.module.state_dict() if isinstance(q_function, DDP) else q_function.state_dict()
                    p_sd = policy.module.state_dict() if isinstance(policy, DDP) else policy.state_dict()
                    if v_function is not None:
                        v_sd = v_function.module.state_dict() if isinstance(v_function, DDP) else v_function.state_dict()
                    
                    if cfg.algo == "iql":
                        torch.save({
                            'epoch': epoch,
                            'q_state_dict': q_sd,
                            'v_state_dict': v_sd,
                            'policy_state_dict': p_sd,
                            'q_optimizer_state_dict': q_optimizer.state_dict(),
                            'v_optimizer_state_dict': v_optimizer.state_dict(),
                            'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                            'policy_lr_scheduler_state_dict': policy_lr_scheduler.state_dict(),
                            'loss': avg_loss,
                            'eval_loss': eval_loss  
                        }, checkpoint_path)
                    elif cfg.algo == "cql":
                        torch.save({
                            'epoch': epoch,
                            'q_state_dict': q_sd,
                            'policy_state_dict': p_sd,
                            'q_optimizer_state_dict': q_optimizer.state_dict(),
                            'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                            'policy_lr_scheduler_state_dict': policy_lr_scheduler.state_dict(),
                            'loss': avg_loss,
                            'eval_loss': eval_loss  
                        }, checkpoint_path)
                    best_loss = eval_loss
    
    if is_main_process():
        print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")


@click.command()
@click.option("-c", "--config", type=str, required=True, default='train_opal_high_level_policy', help="config file name")
def main(config):
    ddp_enabled = setup_distributed()
    
    CONFIG_FILE = os.path.join(os.path.dirname(os.getcwd()), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_conf = OmegaConf.load(os.path.join(os.getcwd(), f'data/outputs/opal_high_level/{cfg.resume_ckpt_dir}/{config}.yaml'))
        cfg.data = resume_conf.data
        cfg.model = resume_conf.model
        cfg.train = resume_conf.train
        del resume_conf
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    if ddp_enabled:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        device = train_cfg.device
    train_cfg.device = device
    
    base_seed = int(train_cfg.seed)
    rank = get_rank()
    seed_all(base_seed + rank)
    
    # pretrained encoder
    ENCODER_DICT_PATH = os.path.join(os.path.dirname(os.getcwd()), f'data/outputs/opal/{train_cfg.encoder_dir}/{train_cfg.encoder_dict_name}.pt')
    encoder = Encoder(model_cfg)
    ckpt = torch.load(ENCODER_DICT_PATH)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    
    dataset = LatentDataset(seed=train_cfg.seed, gamma=train_cfg.discount, encoder=encoder, cfg=data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if ddp_enabled else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if ddp_enabled else None
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=data_cfg.train_batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler, 
        num_workers=data_cfg.num_workers, 
        pin_memory=data_cfg.pin_memory, 
        persistent_workers=(data_cfg.num_workers>0),
        worker_init_fn=_worker_init_fn
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=data_cfg.val_batch_size, 
        shuffle=False, 
        sampler=val_sampler, 
        num_workers=data_cfg.num_workers, 
        pin_memory=data_cfg.pin_memory,
        persistent_workers=(data_cfg.num_workers>0),
        worker_init_fn=_worker_init_fn
        )
    
    # Create model.
    q_func = TwinQ(model_cfg)
    policy = GaussianPolicy(model_cfg)
    q_func.initialize()
    policy.initialize()
    
    ckpt = None
    v_func = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f'data/outputs/opal_high_level/{cfg.resume_ckpt_dir}/*.pt')))
        ckpt = torch.load(ckpts[-1])
        q_func.load_state_dict(ckpt['q_state_dict'])
        policy.load_state_dict(ckpt['policy_state_dict'])
        if ddp_enabled:
            for m in [q_func, policy]:
                for p in m.parameters():
                    distributed.broadcast(p.data, src=0)
        if train_cfg.algo == "iql":
            v_func = ValueFunction(model_cfg)
            v_func.initialize()
            if cfg.resume:
                v_func.load_state_dict(ckpt['v_state_dict'])
            if ddp_enabled:
                for p in v_func.parameters():
                    distributed.broadcast(p.data, src=0)
    
    scaler = torch.cuda.amp.GradScaler()
    
    if is_main_process():
        checkpoint_dir = os.path.join(os.path.dirname(os.getcwd()), f"data/outputs/opal_high_level/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if cfg.verbose:
            print(f"Created output directory: {checkpoint_dir}.")
        OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
        
        if cfg.wb:
            wandb.init(project=cfg.wandb_project,
                   config=OmegaConf.to_container(cfg, resolve=True))
            wandb.run.tags = cfg.wandb_tag
            wandb.run.name = f"{cfg.wandb_name}-{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        checkpoint_dir = None
    
    if is_dist_avail_and_initialized():
        distributed.barrier()
        if is_main_process():
            path_bytes = checkpoint_dir.encode("utf-8")
            length = torch.tensor([len(path_bytes)], dtype=torch.int32, device=train_cfg.device)
        else:
            path_bytes = b""
            length = torch.tensor([0], dtype=torch.int32, device=train_cfg.device)
        
        distributed.broadcast(length, src=0)
        buf = torch.empty(length.item(), dtype=torch.uint8, device=train_cfg.device)
        if is_main_process():
            buf[:len(path_bytes)] = torch.tensor(list(path_bytes), dtype=torch.uint8, device=train_cfg.device)
        distributed.broadcast(buf, src=0)
        checkpoint_dir = bytes(buf.cpu().numpy().tolist()).decode('utf-8')
    
    if is_main_process():
        logger = JsonLogger(path=os.path.join(checkpoint_dir, "log.json"))
    else:
        class _DummyLogger:
            def start(self): pass
            def stop(self): pass
            def log(self, *_args, **_kwargs): pass
        logger = _DummyLogger()
    
    logger.start()
    
    try:
        train(
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
            logger=logger
            )
    finally:
        logger.stop()
        cleanup_distributed()
    

if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()