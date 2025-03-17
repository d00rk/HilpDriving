import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import glob
import wandb
import datetime as dt
import numpy as np
import json
from PIL import Image
import json
import click
import tqdm
import shutil
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.hso_vp import * 
from dataset.dataset import *
from utils.seed import seed_all

        
def skill_data_filtering(dataset,
                         num_clusters, 
                         distance_threshold,
                         seed=42):
    action_sequences = []
    print(1)
    for i in range(len(dataset)):
        state, action, next_state, reward, terminal, timeout = dataset[i]
        action_sequences.append(action.cpu().numpy().flatten())
    action_sequences = np.array(action_sequences)
    print(2)
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    cluster_labels = kmeans.fit_predict(action_sequences)
    print(3)
    filtered_indices = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_actions = action_sequences[cluster_indices]
        distances = pairwise_distances(cluster_actions)
        
        retained_indices = []
        for i, row in enumerate(distances):
            if all(d > distance_threshold for d in row[:i]):
                retained_indices.append(cluster_indices[i])
        filtered_indices.extend(retained_indices)
    
    filtered_dataset = [dataset[i] for i in filtered_indices]
    return filtered_dataset


def eval(encoder, 
         decoder, 
         prior, 
         dataloader, 
         num_evals, 
         device):
    eval_loss = []
    with torch.no_grad():
        for _ in range(num_evals):
            total_loss = 0.0
            for states, actions, next_states, rewards, terminals, timeouts in dataloader:
                states = states.float().to(device)
                actions = actions.float().to(device)
               
                z, logits, z_mean, z_std = encoder(states, actions)
                recon_actions = decoder(states[:, 0, :, :, :], z)
                prior_z, prior_logits, prior_mean, prior_std = prior(states[:, 0, :, :, :])
                
                recon_loss = F.mse_loss(recon_actions, actions)
                             
                kl_y = (logits * (torch.log(logits + 1e-8) - torch.log(prior_logits + 1e-8))).sum(dim=-1).mean()
                kl_z = ((torch.log(z_std + 1e-8) - torch.log(prior_std + 1e-8)) + (prior_std.pow(2) + (z_mean - prior_mean).pow(2)) / (2 * z_std.pow(2)) - 0.5).sum(dim=-1).mean()
                
                loss = recon_loss + 0.01 * kl_y + 0.01 * kl_z
                
                total_loss += loss
            
            avg_loss = total_loss / len(dataloader)
            eval_loss.append(avg_loss)
        
        eval_loss = torch.tensor(eval_loss, dtype=torch.float32).cpu().numpy()
        return np.mean(eval_loss)
                
    
def train(encoder, 
          decoder, 
          prior, 
          train_dataloader, 
          val_dataloader,
          num_epochs, 
          num_evals, 
          eval_frequency, 
          lr, 
          device,
          verbose,
          wb,
          checkpoint_dir):   
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()) + list(prior.parameters()), lr=lr)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    prior = prior.to(device)
    
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if verbose:
        print(f"[torch] {device} is used.")
    
    best_loss = np.inf
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss_discrete = 0.0
        total_kl_loss_continuous = 0.0
        
        pbar = tqdm.tqdm(enumerate(train_dataloader), 
                         total=len(train_dataloader), 
                         desc=f"[Epoch {epoch}/{num_epochs}]", 
                         leave=True, 
                         ncols=100)
        
        for i, (states, actions, next_states, rewards, terminals, timeouts) in pbar:
            states = states.float().to(device)
            actions = actions.float().to(device)
            
            z, logits, z_mean, z_std = encoder(states, actions)
            recon_actions = decoder(states[:, 0, :, :, :], z)
            prior_z, prior_logits, prior_means, prior_stds = prior(states[:, 0, :, :, :])

            recon_loss = F.mse_loss(recon_actions, actions)
            
            kl_y = (logits * (torch.log(logits + 1e-8) - torch.log(prior_logits + 1e-8))).sum(dim=-1).mean()
            kl_z = ((torch.log(z_std + 1e-8) - torch.log(prior_stds + 1e-8)) + (prior_stds.pow(2) + (z_mean - prior_means).pow(2)) / (2 * z_std.pow(2)) - 0.5).sum(dim=-1).mean()
            
            loss = recon_loss + 0.01 * kl_y + 0.01 * kl_z
            
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()) + list(prior.parameters()), max_norm=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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
            print(f"[Train] Epoch [{epoch}/{num_epochs}]   |   Loss: {avg_loss:.4f}   |   Recon Loss: {avg_recon_loss:.4f}   |   KL y loss: {avg_kl_loss_y:.4f}   |   KL z loss: {avg_kl_loss_z:.4f}")

        if wb:
            wandb.log({"train/epoch": epoch,
                       "train/global_step": global_step,
                       "train/loss": avg_loss,
                       "train/reconstruction_loss": avg_recon_loss,
                       "train/kl_y_loss": avg_kl_loss_y,
                       "train/kl_z_loss": avg_kl_loss_z})
        
        if epoch % eval_frequency == 0:
            if verbose:
                print(f"[Evaluation] {epoch} / {num_epochs}")
                
            eval_loss = eval(encoder=encoder, 
                             decoder=decoder, 
                             prior=prior, 
                             num_evals=num_evals, 
                             dataloader=val_dataloader, 
                             device=device)
            if verbose:
                print(f"[Evaluation] Loss:  {eval_loss:.4f}")
            if wb:
                wandb.log({"eval/loss": eval_loss})
            
            if eval_loss <= best_loss:
                if verbose:
                    print(f"Save best model of epoch {epoch}  |   Eval loss: {eval_loss:.4f}")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"hsovp_{dt.datetime.now().replace(microsecond=0)}_epoch_{epoch}.pt")
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
    CONFIG_FILE = os.path.join(os.path.dirname(os.getcwd()), f'opal/config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.train.resume:
        resume_cfg = OmegaConf.load(os.path.join(os.path.dirname(os.getcwd()), f'opal/outputs/hsovp/{cfg.train.resume_ckpt_dir}/{config}.yaml'))
    
    seed = resume_cfg.seed
    seed_all(seed)
    
    data_algo = resume_cfg.data_algo
    data_benchmark = resume_cfg.data_benchmark
    data_town = resume_cfg.data_town       
    
    state_dim = resume_cfg.model.state_dim
    action_dim = resume_cfg.model.action_dim
    latent_dim = resume_cfg.model.latent_dim  
    
    num_workers = resume_cfg.dataset.num_workers
    train_batch_size = resume_cfg.dataset.train_batch_size
    val_batch_size = resume_cfg.dataset.val_batch_size
    val_ratio = resume_cfg.dataset.val_ratio
    trajectory_length = resume_cfg.dataset.trajectory_length
    discrete_option = resume_cfg.dataset.discrete_option
    distance_threshold = resume_cfg.dataset.distance_threshold
    
    device = resume_cfg.train.device
    num_epochs = resume_cfg.train.num_epochs
    num_evals = resume_cfg.train.num_evals
    eval_frequency = resume_cfg.train.eval_frequency
    lr = resume_cfg.train.lr
    resume = cfg.train.resume
    
    verbose = cfg.verbose
    wb = cfg.wb
    wandb_project = cfg.wandb_project
    wandb_name = cfg.wandb_name
    wandb_tag = cfg.wandb_tag
    
    if wb:
        wandb.init(project=wandb_project,
                   config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = wandb_tag
        wandb.run.name = f"{wandb_name}-{dt.datetime.now().replace(microsecond=0)}"
    
    checkpoint_dir = os.path.join(os.path.dirname(os.getcwd()), f'opal/outputs/hsovp/{dt.datetime.now().replace(microsecond=0)}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    shutil.copy(CONFIG_FILE, os.path.join(checkpoint_dir, f"{config}.yaml"))
    
    if verbose:
        print("Creating trajectory dataset")
    dataset = SubTrajDataset(data_algo, data_benchmark, data_town, length=trajectory_length, seed=seed)   
    if verbose:
        print('Creating filtered dataset...')
    filtered_data = skill_data_filtering(dataset=dataset, num_clusters=discrete_option, distance_threshold=distance_threshold, seed=seed)
    filtered_dataset = FilteredDataset(filtered_data)
    train_dataset, val_dataset = filtered_dataset.split_train_val(val_ratio=val_ratio, seed=seed)
    if verbose:
        print(f'Filtered dataset size: {len(filtered_data)}')

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=num_workers)
    
    encoder = Encoder(state_dim=state_dim, 
                      action_dim=action_dim, 
                      latent_dim=latent_dim, 
                      discrete_option=discrete_option)
    decoder = Decoder(state_dim=state_dim, 
                      action_dim=action_dim, 
                      latent_dim=latent_dim,
                      seq_len=trajectory_length)
    prior = Prior(state_dim=state_dim, 
                  latent_dim=latent_dim, 
                  discrete_option=discrete_option)
    encoder.initialize()
    decoder.initialize()
    prior.initialize()
    
    if resume:
        resume_ckpt_dir = cfg.train.resume_ckpt_dir
        ckpts = sorted(glob.glob(os.path.join(os.path.dirname(os.getcwd()), f"opal/outputs/hsovp/{resume_ckpt_dir}", f"hsovp_*.pt")))
        ckpt = torch.load(ckpts[-1])
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        prior.load_state_dict(ckpt['prior_state_dict'])
    
    if verbose:
        print(f"[Train] Train start at {dt.datetime.now().replace(microsecond=0)}")
        
    train(encoder=encoder, 
          decoder=decoder, 
          prior=prior,
          train_dataloader=train_dataloader, 
          val_dataloader=val_dataloader,
          num_epochs=num_epochs, 
          num_evals=num_evals, 
          eval_frequency=eval_frequency, 
          lr=lr, 
          device=device,
          verbose=verbose,
          wb=wb,
          checkpoint_dir=checkpoint_dir)
    
if __name__=="__main__":
    main()