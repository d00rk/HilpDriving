import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import glob
import wandb
import datetime as dt
import numpy as np
import click
from tqdm import tqdm
import copy
from omegaconf import OmegaConf

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from model.hso_vp import Encoder
from model.policy import GaussianPolicy
from model.value_function import ValueFunction, TwinQ
from dataset.dataset import LatentDataset
from utils.seed_utils import seed_all
from utils.utils import asymmetric_l2_loss, update_exponential_moving_average, log_sum_exp
from utils.logger import JsonLogger, _stats_dict


EXP_ADV_MAX = 100.0


def eval(
    q_function,
    target_q_function,
    v_function,
    policy,
    dataloader,
    epoch,
    cfg
    ):
    q_function = q_function.eval()
    policy = policy.eval()
    if cfg.algo == 'iql':
        v_function = v_function.eval()
    
    eval_loss = {}
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        desc=f"[Validation] Epoch {epoch}/{cfg.num_epochs}", 
        leave=True, 
        ncols=100
        )
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=cfg.use_amp):
        total_loss = 0.0
        total_q_loss = 0.0
        total_v_loss = 0.0 if cfg.algo == "iql" else None
        total_policy_loss = 0.0
        for i, (state, z, next_state, reward, terminal, _) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            z = z.to(cfg.device, non_blocking=True)
            next_state = next_state.to(cfg.device, non_blocking=True)
            reward = reward.to(cfg.device, non_blocking=True)
            terminal = terminal.to(cfg.device, non_blocking=True)
            if cfg.algo == "iql":
                q_target = target_q_function(state, z)
                v = v_function(state)
                next_v = v_function(next_state)
                adv = q_target - v
                v_loss = asymmetric_l2_loss(adv, cfg.tau)
                
                targets = reward + (1.0 - terminal) * cfg.discount * next_v.detach()
                q1, q2 = q_function.both(state, z)
                q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
                
                exp_adv = torch.exp(cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
                z_mu, z_logstd = policy(state)
                dist = Normal(z_mu, z_logstd.exp())
                log_prob = dist.log_prob(z).sum(dim=-1)
                policy_loss = -(exp_adv * log_prob).mean()
                
                loss = v_loss.item() + q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += q_loss.item()
                total_v_loss += v_loss.item()
                total_policy_loss += policy_loss.item()
            
            elif cfg.algo == 'cql':
                z_next_mu, z_next_logstd = policy(next_state)
                dist_next = Normal(z_next_mu, z_next_logstd.exp())
                z_next_pred = dist_next.rsample()
                logp_next = dist_next.log_prob(z_next_pred).sum(dim=-1)
                
                q_next_target = target_q_function(next_state, z_next_pred)
                target_q = reward + (1.0 - terminal) * cfg.discount * (q_next_target - cfg.alpha_entropy * logp_next)
                
                q1, q2 = q_function.both(state, z)
                q_loss = sum(F.mse_loss(q, target_q) for q in [q1, q2]) / 2
                
                # conservative term: random + policy + next-policy
                batch_size, z_dim = z.shape
                random_z = torch.rand(batch_size, cfg.num_random, z_dim, device=cfg.device)
                random_z = random_z * (cfg.high - cfg.low) + cfg.low
                
                obs_expand = state.unsqueeze(1).repeat(1, cfg.num_random, 1, 1, 1)
                obs_expand = obs_expand.view(batch_size*cfg.num_random, *state.shape[1:])
                random_z = random_z.view(batch_size * cfg.num_random, z_dim)
                
                q1_random, q2_random = q_function.both(obs_expand, random_z)
                q1_random = q1_random.view(batch_size, cfg.num_random)
                q2_random = q2_random.view(batch_size, cfg.num_random)
                
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
                t_q_loss = q_loss + cql_loss_total
                
                policy_loss = (cfg.alpha_entropy * log_prob - q_new).mean()
                
                loss = t_q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += t_q_loss.item()
                total_policy_loss += policy_loss.item()

        avg_loss = total_loss /len(dataloader)
        avg_q_loss = total_q_loss / len(dataloader)
        avg_v_loss = total_v_loss / len(dataloader) if cfg.algo == "iql" else None
        avg_policy_loss = total_policy_loss / len(dataloader)
        
        eval_loss["eval/loss"] = avg_loss
        eval_loss["eval/q_loss"] = avg_q_loss
        eval_loss["eval/policy_loss"] = avg_policy_loss
        if cfg.algo == "iql":
            eval_loss["eval/v_loss"] = avg_v_loss
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
    logger
    ):
    
    q_function = q_function.to(cfg.device)
    policy = policy.to(cfg.device)
    target_q_function = copy.deepcopy(q_function).requires_grad_(False).to(cfg.device)
    if cfg.algo == 'iql':
        v_function = v_function.to(cfg.device)
    
    q_optimizer = optim.AdamW(q_function.parameters(), lr=cfg.q_lr)
    policy_optimizer = optim.AdamW(policy.parameters(), lr=cfg.policy_lr)
    v_optimizer = optim.AdamW(v_function.parameters(), lr=cfg.v_lr) if v_function is not None else None
    policy_lr_scheduler = CosineAnnealingLR(policy_optimizer, cfg.num_epochs*len(train_dataloader))
    
    if ckpt is not None:
        q_optimizer.load_state_dict(ckpt['q_optimizer_state_dict'])
        policy_optimizer.load_state_dict(ckpt['policy_optimizer_state_dict'])
        policy_lr_scheduler.load_state_dict(ckpt['policy_lr_scheduler_state_dict'])
        if v_optimizer is not None:
            v_optimizer.load_state_dict(ckpt['v_optimizer_state_dict'])
        del ckpt
    
    print(f"[Torch] {cfg.device} is used.")
    print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")

    best_loss = float(np.inf)
    global_step = 0
    for epoch in range(cfg.num_epochs):
        q_function.train()
        policy.train()
        if v_function is not None:
            v_function.train()
        
        total_q_loss = 0.0
        total_v_loss = 0.0 if cfg.algo == "iql" else None
        total_policy_loss = 0.0
        total_loss = 0.0
        pbar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f"[Epoch {epoch}/{cfg.num_epochs}]", 
            leave=True, 
            ncols=100
            )
        
        for i, (state, z, next_state, reward, terminal, _) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            z = z.to(cfg.device, non_blocking=True)
            next_state = next_state.to(cfg.device, non_blocking=True)
            reward = reward.to(cfg.device, non_blocking=True)
            terminal = terminal.to(cfg.device, non_blocking=True)
            
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
                        dist_next = torch.distributions.Normal(z_next_mu, z_next_logstd.exp())
                        z_next_pred = dist_next.rsample()
                        logp_next = dist_next.log_prob(z_next_pred).sum(dim=-1)
                        q_next_target = target_q_function(next_state, z_next_pred)
                        target_q = reward + (1.0 - terminal) * cfg.discount * (q_next_target - cfg.alpha_entropy * logp_next)
                
                    q1, q2 = q_function.both(state, z)
                    q_loss = sum(F.mse_loss(q, target_q) for q in [q1, q2]) / 2
                
                    # conservative term: random + policy + next-policy
                    batch_size, z_dim = z.shape
                    random_z = torch.rand(batch_size, cfg.num_random, z_dim, device=cfg.device)
                    random_z = random_z * (cfg.high - cfg.low) + cfg.low
                    
                    obs_expand = state.unsqueeze(1).repeat(1, cfg.num_random, 1, 1, 1)
                    obs_expand = obs_expand.view(batch_size*cfg.num_random, *state.shape[1:])
                    random_z = random_z.view(batch_size*cfg.num_random, z_dim)
                    
                    q1_random, q2_random = q_function.both(obs_expand, random_z)
                    q1_random = q1_random.view(batch_size, cfg.num_random)
                    q2_random = q2_random.view(batch_size, cfg.num_random)
                    
                    z_mu, z_logstd = policy(state)
                    dist = Normal(z_mu, z_logstd.exp())
                    z_pi = dist.rsample()
                    q1_policy, q2_policy = q_function.both(state, z_pi)
                    q_new = torch.min(q1_policy, q2_policy)
                    
                    q1_cat = torch.cat([q1_random, q1_policy.unsqueeze(1)], dim=1)
                    q2_cat = torch.cat([q2_random, q2_policy.unsqueeze(1)], dim=1)
                    
                    lse_q1 = log_sum_exp(q1_cat / cfg.temperature, dim=1)
                    cql_q1 = (lse_q1 * cfg.temperature - q1).mean()
                    lse_q2 = log_sum_exp(q2_cat/cfg.temperature, dim=1)
                    cql_q2 = (lse_q2 * cfg.temperature - q2).mean()
                    
                    cql_loss_total = cfg.cql_alpha * (cql_q1 + cql_q2)
                    t_q_loss = q_loss + cql_loss_total

                # q function update
                q_optimizer.zero_grad(set_to_none=True)
                scaler.scale(t_q_loss).backward()
                scaler.step(q_optimizer)
                
                update_exponential_moving_average(target_q_function, q_function, cfg.alpha)
                
                # policy update               
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    policy_loss = (cfg.alpha_entropy * log_prob - q_new).mean()
                
                policy_optimizer.zero_grad(set_to_none=True)
                scaler.scale(policy_loss).backward()
                scaler.step(policy_optimizer)
                policy_lr_scheduler.step()
                
                scaler.update()
                
                loss = t_q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += t_q_loss.item()
                total_policy_loss += policy_loss.item()
            else:
                raise ValueError(f"Invalid algorithm: {cfg.algo}")
            
            if (i % cfg.log_frequency) == 0:
                step_log = {
                    'train/epoch': epoch,
                    'train/global_step': global_step,
                    'train/loss': loss,
                    'train/q_loss': q_loss.item(),
                    'train/policy_loss': policy_loss.item(),
                    'train/lr': policy_lr_scheduler.get_last_lr()[0],
                    'debug/has_nan_input': int(
                        torch.isnan(state).any().item() or torch.isnan(next_state).any().item() or torch.isnan(z).any().item()
                    )
                }
                if cfg.algo == 'iql':
                    step_log['train/v_loss'] = v_loss.item()
                
                step_log.update(_stats_dict("debug/state", state))
                step_log.update(_stats_dict("debug/next_state", next_state))
                step_log.update(_stats_dict("debug/z", z))
                
                logger.log(step_log)
                if wb:
                    wandb.log(step_log, step=global_step)
            
            global_step += 1
            pbar.set_postfix({"Total Loss": f"{loss:.4f}"})

        avg_q_loss = total_q_loss / len(train_dataloader)
        avg_policy_loss = total_policy_loss / len(train_dataloader)
        avg_loss = total_loss / len(train_dataloader)
        
        log_dict = {"train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/loss": avg_loss,
                    "train/q_loss": avg_q_loss,
                    "train/policy_loss": avg_policy_loss,
                    "train/lr": policy_lr_scheduler.get_last_lr()[0]}
        
        if cfg.algo == "iql":
            avg_v_loss = total_v_loss / len(train_dataloader)
            log_dict["train/v_loss"] = avg_v_loss
        
        logger.log(log_dict)
        if wb:
            wandb.log(log_dict, step=global_step)
        
        if epoch % cfg.eval_frequency == 0:
            eval_loss_dict = eval(q_function=q_function,
                                  target_q_function=target_q_function,
                                  v_function=v_function,
                                  policy=policy,
                                  dataloader=val_dataloader, 
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
                
                checkpoint_path = os.path.join(checkpoint_dir, f"{cfg.algo}_epoch_{epoch:04d}_loss_{eval_loss:.3f}.pt")
                if cfg.algo == "iql":
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
                elif cfg.algo == "cql":
                    torch.save({
                        'epoch': epoch,
                        'q_state_dict': q_function.state_dict(),
                        'policy_state_dict': policy.state_dict(),
                        'q_optimizer_state_dict': q_optimizer.state_dict(),
                        'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                        'policy_lr_scheduler_state_dict': policy_lr_scheduler.state_dict(),
                        'loss': avg_loss,
                        'eval_loss': eval_loss 
                    }, checkpoint_path)
                best_loss = eval_loss
    
    print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")


@click.command()
@click.option('-c', '--config', required=True, default='train_hsovp_high_level_policy', help='config file name')
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_cfg = OmegaConf.load(os.path.join(os.getcwd(), f'data/outputs/hsovp_high_level_policy/{cfg.resume_ckpt_dir}/{config}.yaml'))
        cfg.data = resume_cfg.data
        cfg.model = resume_cfg.model
        cfg.train = resume_cfg.train
        del resume_cfg
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    seed_all(train_cfg.seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    ENCODER_DICT_PATH = os.path.join(os.getcwd(), f"data/outputs/hsovp/{train_cfg.hsovp_dir}/{train_cfg.hsovp_name}.pt")
    encoder = Encoder(model_cfg)
    ckpt = torch.load(ENCODER_DICT_PATH)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)
    
    dataset = LatentDataset(seed=train_cfg.seed, gamma=train_cfg.discount, encoder=encoder, cfg=data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    train_dataloader = DataLoader(train_dataset, batch_size=data_cfg.train_batch_size, shuffle=True, num_workers=data_cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=data_cfg.val_batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)
 
    # Create model
    q_func = TwinQ(model_cfg)
    policy = GaussianPolicy(model_cfg)
    q_func.initialize()
    policy.initialize()
    
    ckpt = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"data/outputs/hsovp_high_level_policy/{cfg.resume_ckpt_dir}/{train_cfg.algo}_*.pt")))
        ckpt = torch.load(ckpts[-1])
        q_func.load_state_dict(ckpt['q_state_dict'])
        policy.load_state_dict(ckpt['policy_state_dict'])
    
    if train_cfg.algo == "iql":
        v_func = ValueFunction(model_cfg)
        v_func.initialize()
        if cfg.resume:
            v_func.load_state_dict(ckpt['v_state_dict'])
    else:
        v_func = None

    checkpoint_dir = os.path.join(os.getcwd(), f"data/outputs/hsovp_high_level_policy/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Created output directory: {checkpoint_dir}.")
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
    
    logger = JsonLogger(path=os.path.join(checkpoint_dir, "log.json"))
    logger.start()
    if cfg.wb:
        wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = cfg.wandb_tag
        wandb.run.name = f"{cfg.wandb_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        train(q_function=q_func,
            v_function=v_func, 
            policy=policy, 
            scaler=scaler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            verbose=cfg.verbose,
            wb=cfg.wb,
            ckpt=ckpt,
            checkpoint_dir=checkpoint_dir,
            cfg=train_cfg,
            logger=logger)
    finally:
        logger.stop()
    
if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()