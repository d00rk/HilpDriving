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
import tqdm
import click
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.hilp import HilbertRepresentation
from dataset.dataset import GoalDataset
from utils.seed import seed_all
from utils.utils import l2_expectile_loss

def eval_hilp(model, 
              target_model, 
              dataloader, 
              epoch,
              total_epoch,
              expectile_tau,
              device):
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[Validation] {epoch} / {total_epoch}", leave=True, ncols=100)
    with torch.no_grad():
        total_loss = 0.0
        for i, (state, next_state, goal) in pbar:
            state, next_state, goal = state.to(device), next_state.to(device), goal.to(device)
            
            phi_s = model(state)
            phi_g = model(goal) 
            phi_next_s = target_model(next_state).detach()
            phi_next_g = target_model(goal).detach()
                        
            temporal_dist = torch.norm(phi_s - phi_g, dim=-1)
            target_dist = -torch.norm(phi_next_s - phi_next_g, dim=-1)
            loss = l2_expectile_loss(target_dist - temporal_dist, expectile_tau)
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_hilp(model, 
               target_model, 
               train_dataloader, 
               val_dataloader, 
               verbose,
               wb,
               checkpoint_dir,
               cfg):
    
    model = model.to(cfg.device)
    target_model = target_model.to(cfg.device)
    
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr)
    
    if verbose:
        print(f"[Torch] {cfg.device} is used.")
        print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    
    best_loss = np.inf
    global_step = 0
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        progress_bar = tqdm.tqdm(enumerate(train_dataloader),
                                 total=len(train_dataloader), 
                                desc=f"[Train] Epoch {epoch}/{cfg.num_epochs}", 
                                leave=True, ncols=100)
        
        for i, (state, next_state, goal) in progress_bar:
            state, next_state, goal = state.to(cfg.device), next_state.to(cfg.device), goal.to(cfg.device)
            
            # phi(s), phi(g)
            phi_s = model(state)
            phi_g = model(goal)
            # target_phi(s'), target_phi(g) 
            target_phi_next_s = target_model(next_state).detach()
            target_phi_g = target_model(goal).detach()
                
            temporal_dist = torch.norm(phi_s - phi_g, dim=-1)
            target_dist = -torch.norm(target_phi_next_s - target_phi_g, dim=-1)
            loss = l2_expectile_loss(target_dist - temporal_dist, cfg.expectile_tau)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for param, target_param in zip(model.parameters(), target_model.parameters()):
                    target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
            
            total_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_dataloader)
        
        if verbose:
            print(f"[Train] Epoch {epoch}/{cfg.num_epochs}   |   Loss: {avg_loss:.4f}")
        
        if wb:
            wandb.log({"train/epoch": epoch,
                       "train/loss": avg_loss})
        
        
        if epoch % cfg.eval_frequency == 0:               
            eval_loss = eval_hilp(model=model, 
                                  target_model=target_model, 
                                  dataloader=val_dataloader, 
                                  epoch=epoch,
                                  total_epoch=cfg.num_epochs,
                                  device=cfg.device)
            
            if verbose:
                print(f"[Validation] Loss: {eval_loss:.4f}")
            if wb:
                wandb.log({"eval/loss": eval_loss})
            
            if eval_loss <= best_loss:
                if verbose:
                    print(f"Save best model of epoch {epoch}")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"hilbert_representation_{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_epoch_{epoch}.pt")
                
                torch.save({
                    "epoch": epoch,
                    "hilbert_representation_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "eval_loss": eval_loss
                }, checkpoint_path)
                best_loss = eval_loss
                
    if verbose:            
        print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")



@click.command()
@click.option("-c", "--config", default="train_hilp", type=str, help="Config to use")
def main(config):
    CONFIG_FILE = os.path.join(os.path.dirname(os.getcwd()), f'opal/config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)

    if cfg.resume:
        resum_conf = OmegaConf.load(os.path.join(os.path.dirname(os.getcwd()), f'opal/outputs/hilp/{cfg.resume_ckpt_dir}/{config}.yaml'))
        cfg.data = resum_conf.data
        cfg.model = resum_conf.model
        cfg.train = resum_conf.train
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
        
    seed_all(train_cfg.seed)
    
    model = HilbertRepresentation(model_cfg)
    target_model = HilbertRepresentation(model_cfg)
    target_model.load_state_dict(model.state_dict())
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.path.dirname(os.getcwd()), f"opal/outputs/hilp/{cfg.resume_ckpt_dir}/hilp_*.pt")))
        ckpt = torch.load(ckpts[-1])
        if cfg.verbose:
            print(f"Resume from {ckpts[-1]}")
        model.load_state_dict(ckpt["hilp_state_dict"])
        target_model.load_state_dict(ckpt["hilp_state_dict"])
    
    if cfg.verbose:
        print("Create dataset")
        
    dataset = GoalDataset(train_cfg.seed, data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.train_batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=train_cfg.val_batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    
    if cfg.verbose:
        print("Created Dataset, DataLoader.")
    
    if cfg.wb:
        wandb.init(project=cfg.wandb_project,
                   config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = cfg.wandb_tag
        wandb.run.name = f"{cfg.wandb_name}-{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    
    checkpoint_dir = os.path.join(os.path.dirname(os.getcwd()), f'opal/outputs/hilp/{dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
    
    
    train_hilp(model=model, 
               target_model=target_model, 
               train_dataloader=train_dataloader, 
               val_dataloader=val_dataloader, 
               verbose=cfg.verbose, 
               wb=cfg.wb,
               checkpoint_dir=checkpoint_dir,
               cfg=train_cfg)
    

if __name__=="__main__":
    main()