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
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model.hso_vp import * 
from dataset.dataset import FilteredDataset
from utils.seed_utils import seed_all


def eval(encoder, 
         decoder, 
         prior, 
         dataloader,
         epoch,
         num_epochs,
         beta_y,
         beta_z,
         device):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    prior = prior.to(device)
    
    with torch.no_grad():
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_y_loss = 0.0
        total_kl_z_loss = 0.0
        
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[Validation] {epoch} / {num_epochs}", leave=True, ncols=100)
        for i, (states, actions, next_states, rewards, terminals, timeouts, labels) in pbar:
            states = states.float().to(device)
            actions = actions.float().to(device)
           
            z_mu, z_std, logits, discrete_y = encoder(states, actions)
            z = Normal(z_mu, z_std).rsample()
            action_pred = decoder(states, z)
            prior_mu, prior_std, prior_logits, prior_discrete_y = prior(states[:, 0, :, :, :])
                
            recon_loss = F.mse_loss(action_pred, actions)
            
            q = F.softmax(logits, dim=-1)
            p = F.softmax(prior_logits, dim=-1)
                             
            kl_y = (q * (torch.log(q + 1e-8) - torch.log(p + 1e-8))).sum(dim=-1).mean()
            kl_z = ((torch.log(prior_std + 1e-8) - torch.log(z_std + 1e-8)) + (z_std.pow(2) + (z_mu - prior_mu).pow(2)) / (2 * prior_std.pow(2)) - 0.5).sum(dim=-1).mean()
                
            loss = recon_loss + beta_y * kl_y + beta_z * kl_z
                
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_y_loss += kl_y.item()
            total_kl_z_loss += kl_z.item()
            
        total_loss = total_loss / len(dataloader)
        total_recon_loss = total_recon_loss / len(dataloader)
        total_kl_y_loss = total_kl_y_loss / len(dataloader)
        total_kl_z_loss = total_kl_z_loss / len(dataloader)
        
    return total_loss, total_recon_loss, total_kl_y_loss, total_kl_z_loss
                
    
def train(encoder, 
          decoder, 
          prior, 
          train_dataloader, 
          val_dataloader,
          verbose,
          wb,
          checkpoint_dir,
          cfg):
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()) + list(prior.parameters()), lr=cfg.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    
    encoder = encoder.to(cfg.device)
    decoder = decoder.to(cfg.device)
    prior = prior.to(cfg.device)
    if verbose:
        print(f"[Torch] {cfg.device} is used.")
        print(f"[Train] Start at {dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    best_loss = np.inf
    global_step = 0
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss_discrete = 0.0
        total_kl_loss_continuous = 0.0
        
        pbar = tqdm.tqdm(enumerate(train_dataloader), 
                         total=len(train_dataloader), 
                         desc=f"[Epoch {epoch}/{cfg.num_epochs}]", 
                         leave=True, 
                         ncols=100)
        
        for i, (states, actions, next_states, rewards, terminals, timeouts, labels) in pbar:
            states = states.float().to(cfg.device)
            actions = actions.float().to(cfg.device)
            z_mu, z_std, z_logits, discrete_y = encoder(states, actions)
            z = Normal(z_mu, z_std).rsample()
            
            action_pred = decoder(states, z)
            prior_mu, prior_std, prior_logits, prior_discrete_y = prior(states[:, 0, :, :, :])

            recon_loss = F.mse_loss(action_pred, actions)
            
            q = F.softmax(z_logits, dim=-1)
            p = F.softmax(prior_logits, dim=-1)
            kl_y = (q * (torch.log(q + 1e-8) - torch.log(p + 1e-8))).sum(dim=-1).mean()
            kl_z = (torch.log(prior_std + 1e-8) - torch.log(z_std + 1e-8) + (z_std.pow(2) + (z_mu - prior_mu).pow(2)) / (2 * prior_std.pow(2)) - 0.5).sum(dim=-1).mean()
            
            loss = recon_loss + cfg.beta_y * kl_y + cfg.beta_z * kl_z
            
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()) + list(prior.parameters()), max_norm=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss_discrete += kl_y.item()
            total_kl_loss_continuous += kl_z.item()
            
            global_step += 1
            pbar.set_postfix({"Total Loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_dataloader)
        avg_recon_loss = total_recon_loss / len(train_dataloader)
        avg_kl_loss_y = total_kl_loss_discrete / len(train_dataloader)
        avg_kl_loss_z = total_kl_loss_continuous / len(train_dataloader)
        
        if verbose:
            print(f"[Train] Epoch [{epoch}/{cfg.num_epochs}]   |   Loss: {avg_loss:.4f}   |   Recon Loss: {avg_recon_loss:.4f}   |   KL y loss: {avg_kl_loss_y:.4f}   |   KL z loss: {avg_kl_loss_z:.4f}")

        if wb:
            wandb.log({"train/epoch": epoch,
                       "train/global_step": global_step,
                       "train/loss": avg_loss,
                       "train/reconstruction_loss": avg_recon_loss,
                       "train/kl_y_loss": avg_kl_loss_y,
                       "train/kl_z_loss": avg_kl_loss_z,
                       "train/lr": lr_scheduler.get_last_lr()[0]})
        
        if epoch % cfg.eval_frequency == 0:
            if verbose:
                print(f"[Evaluation] {epoch} / {cfg.num_epochs}")
                
            eval_loss, eval_recon_loss, eval_kl_y_loss, eval_kl_z_loss = eval(encoder=encoder, decoder=decoder, prior=prior,  dataloader=val_dataloader, epoch=epoch, num_epochs=cfg.num_epochs, beta_y=cfg.beta_y, beta_z=cfg.beta_z, device=cfg.device)
            if verbose:
                print(f"[Evaluation] Loss:  {eval_loss:.4f}")
            
            if wb:
                wandb.log({"eval/loss": eval_loss,
                           "eval/reconstruction_loss": eval_recon_loss,
                           "eval/kl_y_loss": eval_kl_y_loss,
                           "eval/kl_z_loss": eval_kl_z_loss})
            
            if eval_loss <= best_loss:
                if verbose:
                    print(f"Save best model of epoch {epoch}  |   Eval loss: {eval_loss:.4f}")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"hsovp_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch_{epoch}.pt")
                torch.save({
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "prior_state_dict": prior.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "eval_loss": eval_loss
                }, checkpoint_path)
                best_loss = eval_loss
                
    if verbose:
        print(f"[Train] Finished at {dt.datetime.now().replace(microsecond=0)}")


@click.command()
@click.option('-c', '--config', required=True, default='train_hsovp', help='config file name')
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_cfg = OmegaConf.load(os.path.join(os.getcwd(), f'outputs/hsovp/{cfg.resume_ckpt_dir}/{config}.yaml'))
        cfg.data = resume_cfg.data
        cfg.model = resume_cfg.model
        cfg.train = resume_cfg.train
        del resume_cfg
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    seed_all(train_cfg.seed)
    
    # Create dataset, dataloader
    if cfg.verbose:
        print("Create Filtered Dataset.")
        
    dataset = FilteredDataset(seed=train_cfg.seed, cfg=data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    train_dataloader = DataLoader(train_dataset, batch_size=data_cfg.train_batch_size, shuffle=True, num_workers=data_cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=data_cfg.val_batch_size, shuffle=True, num_workers=data_cfg.num_workers)
 
    if cfg.verbose:
        print("Created Filtered Dataset.")

    # Create model
    encoder = Encoder(model_cfg)
    decoder = Decoder(model_cfg)
    prior = Prior(model_cfg)
    encoder.initialize()
    decoder.initialize()
    prior.initialize()
    
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"outputs/hsovp/{cfg.resume_ckpt_dir}", f"hsovp_*.pt")))
        ckpt = torch.load(ckpts[-1])
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        prior.load_state_dict(ckpt['prior_state_dict'])
        if cfg.verbose:
            print(f"Resume from {ckpts[-1]}")
        
    if cfg.wb:
        wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = cfg.wandb_tag
        wandb.run.name = f"{cfg.wandb_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    checkpoint_dir = os.path.join(os.getcwd(), f"outputs/hsovp/{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
        
    train(encoder=encoder, 
          decoder=decoder, 
          prior=prior,
          train_dataloader=train_dataloader, 
          val_dataloader=val_dataloader,
          verbose=cfg.verbose,
          wb=cfg.wb,
          checkpoint_dir=checkpoint_dir,
          cfg=train_cfg)
    
if __name__=="__main__":
    main()