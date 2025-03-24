"""
Training Hilbert Foundation Policy (Low-Level Policy) pi(a|s, z) with pretrained hilbert representation model.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import wandb
import datetime as dt
import click
import tqdm
import copy
import glob
from omegaconf import OmegaConf

from model.hilp import HilbertRepresentation
from model.value_function import TwinQforHilbert, ValueFunctionforHilbert
from model.policy import GaussianPolicyforHilbert
from dataset.dataset import SubTrajDataset
from utils.utils import asymmetric_l2_loss, update_exponential_moving_average, log_sum_exp
from utils.seed_utils import seed_all
from utils.sampler import sample_latent_vectors

EXP_ADV_MAX = 100.

def eval(model, 
         q_function,
         target_q_function,
         v_function,
         policy, 
         dataloader,
         latent_dim,
         epoch,
         cfg):
    eval_loss = {}
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[Validation] Epoch {epoch}/{cfg.num_epochs}", leave=True, ncols=100)
    with torch.no_grad():
        total_loss = 0.0
        total_q_loss = 0.0
        total_v_loss = 0.0 if cfg.algo == "iql" else None
        total_policy_loss = 0.0
        for i, (states, actions, next_states, _, terminals, _) in pbar:
            states, actions, next_states, terminals = states.to(cfg.device), actions.to(cfg.device), next_states.to(cfg.device), terminals.to(cfg.device)
            _, _, a, b, c = states.shape
            batch_size, seq_len, _ = actions.shape
            z = sample_latent_vectors(batch_size=batch_size, latent_dim=latent_dim)
            z_expand = z.unsqueeze(1).expand(batch_size, seq_len, latent_dim).contiguous().view(batch_size*seq_len, latent_dim)
            z_expand = z_expand.to(cfg.device)
            states = states.view(batch_size * seq_len, a, b, c)
            actions = actions.view(batch_size * seq_len, -1)
            next_states = next_states.view(batch_size * seq_len, a, b, c)
            terminals = terminals.view(batch_size * seq_len)
            
            phi_s = model(states)
            phi_next_s = model(next_states)
            phi_s = phi_s.detach().cpu()
            phi_next_s = phi_next_s.detach().cpu()
            intrinsic_reward = torch.sum((phi_next_s - phi_s) * z_expand.cpu(), dim=-1)
            intrinsic_reward = intrinsic_reward.to(cfg.device)
            
            if cfg.algo == "iql":
                q_target = target_q_function(states, z_expand, actions)
                next_v = v_function(next_states, z_expand)
                v = v_function(states, z_expand)
                adv = q_target - v
                v_loss = asymmetric_l2_loss(adv, cfg.tau)
                
                targets = intrinsic_reward + (1.0 - terminals) * cfg.discount * next_v.detach()
                q1, q2 = q_function.both(states, z_expand, actions)
                q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
                
                exp_adv = torch.exp(cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
                a_mu, a_logstd = policy(states, z_expand)
                a_pred = torch.distributions.Normal(a_mu, a_logstd.exp()).rsample()
                bc_losses = torch.sum((a_pred - actions)**2, dim=1)
                policy_loss = torch.mean(exp_adv * bc_losses)
                
                loss = v_loss.item() + q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += q_loss.item()
                total_v_loss += v_loss.item()
                total_policy_loss += policy_loss.item()
            
            elif cfg.algo == 'cql':
                a_next_mu, a_next_logstd = policy(next_states, z_expand)
                a_next_pred = torch.distributions.Normal(a_next_mu, a_next_logstd.exp()).rsample()
                q_next_target = target_q_function(next_states, z_expand, a_next_pred)
                target_q = intrinsic_reward + (1.0 - terminals) * cfg.discount * q_next_target
                
                q1, q2 = q_function.both(states, z_expand, actions)
                q_loss = sum(F.mse_loss(q, target_q) for q in [q1, q2]) / 2
                
                batch_size, act_dim = actions.shape
                random_acts = torch.rand(batch_size, cfg.num_random, act_dim, device=cfg.device)
                random_acts = random_acts * (cfg.action_high - cfg.action_low) + cfg.action_low
                
                obs_expand = states.unsqueeze(1).expand(-1, cfg.num_random, -1)
                random_acts = random_acts.view(batch_size * cfg.num_random, act_dim)
                
                q1_random, q2_random = q_function.both(obs_expand.reshape(batch_size*cfg.num_random, -1), random_acts)
                q1_random = q1_random.view(batch_size, cfg.num_random)
                q2_random = q2_random.view(batch_size, cfg.num_random)
                
                lse_q1 = log_sum_exp(q1_random / cfg.temperature, dim=1)
                cql_q1 = (lse_q1 * cfg.temperature - q1).mean()
                lse_q2 = log_sum_exp(q2_random / cfg.temperature, dim=1)
                cql_q2 = (lse_q2 * cfg.temperature - q2).mean()
                cql_loss_total = cfg.cql_alpha * (cql_q1 + cql_q2)
                t_q_loss = q_loss + cql_loss_total
                
                a_mu, a_logstd = policy(states, z_expand)
                a_pred = torch.distributions.Normal(a_mu, a_logstd.exp()).rsample()
                log_prob = torch.distributions.Normal(a_mu, a_logstd.exp()).log_prob(a_pred).sum(dim=-1)
                q_new = q_function(states, z_expand, a_pred)
                policy_loss = (0.2 * log_prob - q_new).mean()
                
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


def train(model, 
          q_function,
          v_function,
          policy, 
          train_dataloader, 
          val_dataloader,  
          latent_dim,
          verbose,
          wb,
          checkpoint_dir,
          cfg):
    model = model.to(cfg.device)
    q_function = q_function.to(cfg.device)
    v_function = v_function.to(cfg.device)
    policy = policy.to(cfg.device)
    target_q_function = copy.deepcopy(q_function).requires_grad_(False).to(cfg.device)
    
    q_optimizer = torch.optim.AdamW(q_function.parameters(), lr=cfg.q_lr)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.policy_lr)
    v_optimizer = torch.optim.AdamW(v_function.parameters(), lr=cfg.v_lr) if v_function is not None else None
    
    policy_lr_scheduler = CosineAnnealingLR(policy_optimizer, cfg.max_steps)
    
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if verbose:
        print(f"[Torch] {cfg.device} is used.")
        print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")

    best_loss = np.inf
    global_step = 0
    for epoch in range(cfg.num_epochs):
        total_q_loss = 0.0
        total_v_loss = 0.0 if cfg.algo == "iql" else None
        total_policy_loss = 0.0
        total_loss = 0.0
        progress_bar = tqdm.tqdm(enumerate(train_dataloader), 
                                 total=len(train_dataloader), 
                                 desc=f"[Train] Epoch {epoch}/{cfg.num_epochs}", 
                                 leave=True, 
                                 ncols=100)
        
        for i, (state, action, next_state, _, terminal, _) in progress_bar:
            state, action, next_state, terminal = state.to(cfg.device), action.to(cfg.device), next_state.to(cfg.device), terminal.to(cfg.device)
            _, _, a, b, c = state.shape
            batch_size, seq_len, _ = action.shape
            
            z = sample_latent_vectors(batch_size=batch_size, latent_dim=latent_dim)
            z_expand = z.unsqueeze(1).expand(batch_size, seq_len, latent_dim).contiguous().view(batch_size*seq_len, latent_dim)
            z_expand = z_expand.to(cfg.device)
            
            state = state.view(batch_size * seq_len, a, b, c)
            action = action.view(batch_size * seq_len, -1)
            next_state = next_state.view(batch_size * seq_len, a, b, c)
            terminal = terminal.view(batch_size * seq_len)
            
            with torch.no_grad():
                phi_s = model(state)
                phi_next_s = model(next_state)
            phi_s = phi_s.detach().cpu()
            phi_next_s = phi_next_s.detach().cpu()
            intrinsic_reward = torch.sum((phi_next_s-phi_s) * z_expand.cpu(), dim=-1)
            intrinsic_reward = intrinsic_reward.to(cfg.device)
            
            if cfg.algo == "iql":
                with torch.no_grad():
                   q_target = target_q_function(state, z_expand, action)
                   next_v = v_function(next_state, z_expand)
                v = v_function(state, z_expand)
                adv = q_target - v
                v_loss = asymmetric_l2_loss(adv, cfg.tau)
                v_optimizer.zero_grad(set_to_none=True)
                v_loss.backward()
                v_optimizer.step()
                
                targets = intrinsic_reward + (1.0 - terminal) * cfg.discount * next_v.detach()
                q1, q2 = q_function.both(state, z_expand, action)
                q_loss = sum(F.mse_loss(q, targets) for q in [q1, q2]) / 2
                q_optimizer.zero_grad(set_to_none=True)
                q_loss.backward()
                q_optimizer.step()
                
                update_exponential_moving_average(target_q_function, q_function, cfg.alpha)
                
                exp_adv = torch.exp(cfg.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
                a_mu, a_logstd = policy(state, z_expand)
                a_pred = torch.distributions.Normal(a_mu, a_logstd.exp()).rsample()
                bc_losses = torch.sum((a_pred - action)**2, dim=1)
                policy_loss = torch.mean(exp_adv * bc_losses)
                policy_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                policy_optimizer.step()
                policy_lr_scheduler.step()
                
                loss = v_loss.item() + q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += q_loss.item()
                total_v_loss += v_loss.item()
                total_policy_loss += policy_loss.item()
            
            elif cfg.algo == 'cql':
                with torch.no_grad():
                    a_next_mu, a_next_logstd = policy(next_state, z_expand)
                    a_next_pred = torch.distributions.Normal(a_next_mu, a_next_logstd.exp()).rsample()
                    q_next_target = target_q_function(next_state, z_expand, a_next_pred)
                    target_q = intrinsic_reward + (1.0 - terminal) * cfg.discount * q_next_target
                
                q1, q2 = q_function.both(state, z, action)
                q_loss = sum(F.mse_loss(q, target_q) for q in [q1, q2]) / 2
                
                batch_size, act_dim = action.shape
                random_acts = torch.rand(batch_size, cfg.num_random, act_dim, device=cfg.device)
                random_acts = random_acts * (cfg.action_high - cfg.action_low) + cfg.action_low
                
                obs_expand = state.unsqueeze(1).expand(-1, cfg.num_random, -1)
                random_acts = random_acts.view(batch_size * cfg.num_random, act_dim)
                
                q1_random, q2_random = q_function.both(obs_expand.reshape(batch_size*cfg.num_random, -1), random_acts)
                q1_random = q1_random.view(batch_size, cfg.num_random)
                q2_random = q2_random.view(batch_size, cfg.num_random)
                
                lse_q1 = log_sum_exp(q1_random / cfg.temperature, dim=1)
                cql_q1 = (lse_q1 * cfg.temperature - q1).mean()
                lse_q2 = log_sum_exp(q2_random / cfg.temperature, dim=1)
                cql_q2 = (lse_q2 * cfg.temperature - q2).mean()
                
                cql_loss_total = cfg.cql_alpha * (cql_q1 + cql_q2)
                
                t_q_loss = q_loss + cql_loss_total
                
                q_optimizer.zero_grad(set_to_none=True)
                t_q_loss.backward()
                q_optimizer.step()
                
                update_exponential_moving_average(target_q_function, q_function, cfg.alpha)
                
                a_mu, a_logstd = policy(state, z)
                a_pred = torch.distributions.Normal(a_mu, a_logstd.exp()).rsample()
                log_prob = torch.distributions.Normal(a_mu, a_logstd.exp()).log_prob(a_pred).sum(dim=-1)
                
                q_new = q_function(state, z, a_pred)
                policy_loss = (0.2 * log_prob - q_new).mean()
                
                policy_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                policy_optimizer.step()
                policy_lr_scheduler.step()
                
                loss = t_q_loss.item() + policy_loss.item()
                total_loss += loss
                total_q_loss += t_q_loss.item()
                total_policy_loss += policy_loss.item()
            else:
                raise ValueError(f"Invalid algorithm: {cfg.algo}")
            
            global_step += 1
            progress_bar.set_postfix({"Total Loss": f"{loss:.4f}"})

        avg_q_loss = total_q_loss / len(train_dataloader)
        avg_policy_loss = total_policy_loss / len(train_dataloader)
        avg_loss = total_loss / len(train_dataloader)
        
        log_dict = {"train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/loss": avg_loss,
                    "train/q_loss": avg_q_loss,
                    "train/policy_loss": avg_policy_loss}
        
        if cfg.algo == "iql":
            avg_v_loss = total_v_loss / len(train_dataloader)
            log_dict["train/v_loss"] = avg_v_loss
        
        if verbose and cfg.algo == "iql":
            print(f"[Train] Epoch [{epoch}/{cfg.num_epochs}]   |   Total V Loss: {avg_v_loss:.4f}   |   Total Q Loss: {avg_q_loss:.4f}   |   Total Policy Loss: {avg_policy_loss:.4f}   |   Total Loss: {avg_loss:.4f}")
        elif verbose and cfg.algo == "cql":
            print(f"[Train] Epoch [{epoch}/{cfg.num_epochs}]   |   Total Q Loss: {avg_q_loss:.4f}   |   Total Policy Loss: {avg_policy_loss:.4f}   |   Total Loss: {avg_loss:.4f}")
        
        if wb:
            wandb.log(log_dict)
        
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
            
            if verbose:
                print(f"[Validation] Loss: {eval_loss_dict['eval/loss']:.4f}")
            if wb:
                wandb.log(eval_loss_dict)
                
            if eval_loss_dict['eval/loss'] <= best_loss:
                if verbose:
                    print(f"Save best model of epoch {epoch}")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"hilbert_policy_{cfg.algo}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch_{epoch}.pt")
                if cfg.algo == "iql":
                    torch.save({
                        'epoch': epoch,
                        'q_state_dict': q_function.state_dict(),
                        'v_state_dict': v_function.state_dict(),
                        'policy_state_dict': policy.state_dict(),
                        'loss': avg_loss,
                        'eval_loss': eval_loss_dict["eval/loss"]  
                    }, checkpoint_path)
                elif cfg.algo == "cql":
                    torch.save({
                        'epoch': epoch,
                        'q_state_dict': q_function.state_dict(),
                        'policy_state_dict': policy.state_dict(),
                        'loss': avg_loss,
                        'eval_loss': eval_loss_dict["eval/loss"]  
                    }, checkpoint_path)
                best_loss = eval_loss_dict["eval/loss"]
    
    if verbose:
        print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")



@click.command()
@click.option("-c", "--config", type=str, default='train_hilbert_policy', required=True, help="config file name")
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_conf = OmegaConf.load(os.path.join(os.getcwd(), f"outputs/hilbert_policy/{cfg.resume_ckpt_dir}/{config}.yaml"))
        cfg.data = resume_conf.data
        cfg.model = resume_conf.model
        cfg.train = resume_conf.train
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    seed_all(train_cfg.seed)

    # pretrained hilbert representation model
    HILP_DICT_PATH = os.path.join(os.getcwd(), f"outputs/hilp/{train_cfg.hilp_dir}/{train_cfg.hilp_dict_name}.pt")
    hilbert_representation = HilbertRepresentation(model_cfg)
    ckpt = torch.load(HILP_DICT_PATH)
    hilbert_representation.load_state_dict(ckpt['hilbert_representation_state_dict'])
    hilbert_representation.eval()
    
    if cfg.verbose:
        print("Create trajectory dataset")
        
    dataset = SubTrajDataset(train_cfg.seed, data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    train_dataloader = DataLoader(train_dataset, batch_size=data_cfg.train_batch_size, shuffle=True, num_workers=data_cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=data_cfg.val_batch_size, shuffle=True, num_workers=data_cfg.num_workers)
    
    if cfg.verbose:
        print("Created Dataset, DataLoader.")
        
    q_func = TwinQforHilbert(model_cfg)
    policy = GaussianPolicyforHilbert(model_cfg)
    
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"outputs/hilbert_policy/{cfg.resume_ckpt_dir}/hilbert_policy_{train_cfg.algo}_*.pt")))
        ckpt = torch.load(ckpts[-1])
        q_func.load_state_dict(ckpt['q_state_dict'])
        policy.load_state_dict(ckpt['policy_state_dict'])
    
    if train_cfg.algo == "iql":
        v_func = ValueFunctionforHilbert(model_cfg)
        if cfg.resume:
            v_func.load_state_dict(ckpt['v_state_dict'])
    else:
        v_func = None
    
    if cfg.wb:
        wandb.init(project=cfg.wandb_project,
                   config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = cfg.wandb_tag
        wandb.run.name = f"{cfg.wandb_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    checkpoint_dir = os.path.join(os.getcwd(), f"outputs/hilbert_policy/{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if cfg.verbose:
        print(f"Created output directory: {checkpoint_dir}.")
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
    
    train(model=hilbert_representation, 
          q_function=q_func, 
          v_function=v_func, 
          policy=policy, 
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          latent_dim=model_cfg.latent_dim,
          verbose=cfg.verbose,
          wb=cfg.wb,
          checkpoint_dir=checkpoint_dir,
          cfg=train_cfg)
    

if __name__=="__main__":   
    main()