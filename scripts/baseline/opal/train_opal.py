import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import glob
import wandb
import datetime as dt
import numpy as np
import click
import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Normal

from model.opal import *
from dataset.dataset import SubTrajDataset
from utils.seed_utils import seed_all


def eval(encoder, 
         decoder, 
         prior, 
         dataloader, 
         kl_weight,
         epoch,
         num_epochs, 
         device):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    prior = prior.to(device)
    
    with torch.no_grad():
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[Validation] {epoch} / {num_epochs}", leave=True, ncols=100)
        for i, (states, actions, next_states, rewards, terminals, timeouts) in pbar:
            states = states.to(device)
            actions = actions.to(device)
            
            latent_mu, latent_logstd = encoder(states, actions)
            latent_std = torch.exp(0.5 * torch.clamp(latent_logstd, min=-10, max=10))
            latent_std = torch.clamp(latent_std, min=1e-6)
            z = Normal(latent_mu, latent_std).rsample()
            
            action_mu, action_logstd = decoder(states[:, 0, :], z)
            prior_mu, prior_logstd = prior(states[:, 0, :])
            
            kl_loss = 0.5 * torch.sum(prior_logstd - latent_logstd + (torch.exp(latent_logstd) + (latent_mu - prior_mu).pow(2)) / torch.exp(prior_logstd) - 1)
        
            reconstruction_loss = nn.MSELoss()(action_mu, actions[:, 0, :])
            
            loss = reconstruction_loss + kl_weight * kl_loss
            
            total_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
        
        total_loss = total_loss / len(dataloader)
        total_recon_loss = total_recon_loss / len(dataloader)
        total_kl_loss = total_kl_loss / len(dataloader)
        
    return total_loss, total_recon_loss, total_kl_loss


def train(encoder, 
          decoder, 
          prior, 
          train_dataloader, 
          val_datalaoader,
          verbose,
          wb,
          checkpoint_dir,
          cfg):
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()) + list(prior.parameters()), lr=cfg.lr)
    
    encoder = encoder.to(cfg.device)
    decoder = decoder.to(cfg.device)
    prior = prior.to(cfg.device)

    if verbose:
        print(f"[Torch] {cfg.device} is used.")
        print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    
    best_loss = np.inf
    global_step = 0
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        progress_bar = tqdm.tqdm(enumerate(train_dataloader), 
                                 total=len(train_dataloader), 
                                 desc=f"[Epoch {epoch}/{cfg.num_epochs}]", 
                                 leave=True, 
                                 ncols=100)
        
        for i, (state, action, next_state, _, terminal, _) in progress_bar:
            state, action, next_state, terminal = state.to(cfg.device), action.to(cfg.device), next_state.to(cfg.device), terminal.to(cfg.device)
            _, length, _, _, _ = state.shape
            latent_mu, latent_logstd = encoder(state, action)
            latent_std = torch.exp(0.5 * torch.clamp(latent_logstd, min=-10, max=10))
            latent_std = torch.clamp(latent_std, min=1e-6)
            z = Normal(latent_mu, latent_std).rsample()
            
            if z.dim() == 2:
                z = z.unsqueeze(1).expand(-1, length, -1)
            
            recon_losses = list()
            for t in range(length):
                action_mu_t, action_logstd_t = decoder(state[:, t, :], z[:, t, :])
                step_loss = nn.MSELoss()(action_mu_t, action[:, t, :])
                recon_losses.append(step_loss)

            reconstruction_loss = torch.stack(recon_losses).mean()
            
            prior_mu, prior_logstd = prior(state[:, 0, :])
            kl_loss = 0.5 * torch.mean(prior_logstd - latent_logstd + (torch.exp(latent_logstd) + (latent_mu - prior_mu).pow(2)) / torch.exp(prior_logstd) - 1)
            
            loss = reconstruction_loss + cfg.kl_weight * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
            
            global_step += 1
            progress_bar.set_postfix({"Total Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_dataloader)
        avg_recon_loss = total_recon_loss / len(train_dataloader)
        avg_kl_loss = total_kl_loss / len(train_dataloader)
        
        if verbose:
            print(f"Epoch [{epoch}/{cfg.num_epochs}]   |   Loss: {avg_loss:.4f}")
        
        if wb:
            wandb.log({"train/epoch": epoch,
                       "train/global_step": global_step,
                       "train/loss": avg_loss,
                       "train/recon_loss": avg_recon_loss,
                       "train/kl_loss": avg_kl_loss})
        
        if epoch % cfg.eval_frequency == 0:
            if verbose:
                print(f"[Evaluation] {epoch} / {cfg.num_epochs}")
            
            eval_loss, eval_recon_loss, eval_kl_loss = eval(encoder=encoder, 
                             decoder=decoder, 
                             prior=prior, 
                             dataloader=val_datalaoader, 
                             kl_weight=cfg.kl_weight, 
                             epoch=epoch,
                             num_epochs=cfg.num_epochs,
                             device=cfg.device)
            if verbose:
                print(f"[Validation] Loss: {eval_loss:.4f}")
            
            if wb:
                wandb.log({"eval/loss": eval_loss,
                           "eval/recon_loss": eval_recon_loss,
                           "eval/kl_loss": eval_kl_loss})
            
            if eval_loss <= best_loss:
                if verbose:
                    print(f"Save best model of epoch: {epoch}   |   Eval loss: {eval_loss}")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"opal_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'prior_state_dict': prior.state_dict(),
                    'oprimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'eval_loss': eval_loss
                }, checkpoint_path)
                best_loss = eval_loss
                
    if verbose:
        print(f"[Train] Finished at {dt.datetime.now().replace(microsecond=0)}")



@click.command()
@click.option("-c", "--config", type=str, default='train_opal', required=True, help="config file name")
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_conf = OmegaConf.load(os.path.join(os.getcwd(), f'outputs/opal/{cfg.resume_ckpt_dir}/{config}.yaml'))
        cfg.data = resume_conf.data
        cfg.model = resume_conf.model
        cfg.train = resume_conf.train
        del resume_conf
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
 
    seed_all(train_cfg.seed)
    
    # Create dataset, dataloader
    if cfg.verbose:
        print("Create Trajectory dataset")
        
    dataset = SubTrajDataset(seed=train_cfg.seed, cfg=data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    train_dataloader = DataLoader(train_dataset, batch_size=data_cfg.train_batch_size, shuffle=True, num_workers=data_cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=data_cfg.val_batch_size, shuffle=True, num_workers=data_cfg.num_workers)
    
    if cfg.verbose:
        print("Create dataset, dataloader.")
    
    # Create model
    encoder = Encoder(model_cfg)
    decoder = Decoder(model_cfg)
    prior = Prior(model_cfg)
    encoder.initialize()
    decoder.initialize()
    prior.initialize()
    
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"outputs/opal/{cfg.resume_ckpt_dir}/opal_*.pt")))
        ckpt = torch.load(ckpts[-1])
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        prior.load_state_dict(ckpt['prior_state_dict'])

    if cfg.wb:
        wandb.init(project=cfg.wandb_project,
                   config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = cfg.wandb_tag
        wandb.run.name = f"{cfg.wandb_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
     
    checkpoint_dir = os.path.join(os.getcwd(), f"outputs/opal/{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if cfg.verbose:
        print(f"Created output directory: {checkpoint_dir}.")
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
    
    train(encoder=encoder, 
          decoder=decoder, 
          prior=prior, 
          train_dataloader=train_dataloader,
          val_datalaoader=val_dataloader,
          verbose=cfg.verbose,
          wb=cfg.wb,
          checkpoint_dir=checkpoint_dir,
          cfg=train_cfg)


if __name__=="__main__":
    main()