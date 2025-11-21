import sys
import os
from pathlib import Path

# Ensure project root is on the path so imports like `model.*` work even when
# running from nested directories.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import glob
import wandb
import datetime as dt
import numpy as np
import click
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from model.hso_vp import * 
from dataset.dataset import FilteredDataset
from utils.utils import ensure_chw
from utils.seed_utils import seed_all
from utils.logger import JsonLogger, _stats_dict


def eval(
    encoder, 
    decoder, 
    prior, 
    dataloader,
    epoch,
    beta_y,
    beta_z,
    cfg
    ):
    encoder.eval()
    decoder.eval()
    prior.eval()
    
    eval_loss = {}
    pbar = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            desc=f"[Validation] {epoch}/{cfg.num_epochs}", 
            leave=True, 
            ncols=100
            )
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=cfg.use_amp)
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_y_loss = 0.0
        total_kl_z_loss = 0.0
        for i, (state, action, next_state, reward, terminal, timeout, label) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            action = action.to(cfg.device, non_blocking=True)
           
            z_mu, z_std, logits, discrete_y = encoder(state, action)
            z = Normal(z_mu, z_std).rsample()
            action_pred = decoder(state, z)
            prior_mu, prior_std, prior_logits, prior_discrete_y = prior(state[:, 0, :, :, :])
            
            recon_loss = F.mse_loss(action_pred, action)
            
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
        
        eval_loss['eval/loss'] = total_loss
        eval_loss['eval/recon_loss'] = total_recon_loss
        eval_loss['eval/kl_y_loss'] = total_kl_y_loss
        eval_loss['eval/kl_z_loss'] = total_kl_z_loss
        
    return eval_loss


def train(
    encoder, 
    decoder, 
    prior, 
    scaler,
    train_dataloader, 
    val_dataloader,
    wb,
    ckpt,
    checkpoint_dir,
    cfg,
    logger
    ):
    encoder = encoder.to(cfg.device)
    decoder = decoder.to(cfg.device)
    prior = prior.to(cfg.device)
    
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(prior.parameters())
    optimizer = optim.AdamW(params, lr=cfg.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs*len(train_dataloader))
    
    if ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        del ckpt
    
    print(f"[Torch] {cfg.device} is used.")
    print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")
    
    best_loss = float(np.inf)
    global_step = 0
    for epoch in range(cfg.num_epochs):
        encoder.train()
        decoder.train()
        prior.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss_discrete = 0.0
        total_kl_loss_continuous = 0.0
        
        pbar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f"[Epoch {epoch}/{cfg.num_epochs}]", 
            leave=True, 
            ncols=100
            )
        
        for i, (state, action, next_state, reward, terminal, timeout, label) in pbar:
            state = state.to(cfg.device, non_blocking=True)
            action = action.to(cfg.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                z_mu, z_std, z_logits, discrete_y = encoder(state, action)
                z = Normal(z_mu, z_std).rsample()
                
                action_pred = decoder(state, z)
                prior_mu, prior_std, prior_logits, prior_discrete_y = prior(state[:, 0, :, :, :])

                recon_loss = F.mse_loss(action_pred, action)
                
                q = F.softmax(z_logits, dim=-1)
                p = F.softmax(prior_logits, dim=-1)
                kl_y = (q * (torch.log(q + 1e-8) - torch.log(p + 1e-8))).sum(dim=-1).mean()
                kl_z = (torch.log(prior_std + 1e-8) - torch.log(z_std + 1e-8) + (z_std.pow(2) + (z_mu - prior_mu).pow(2)) / (2 * prior_std.pow(2)) - 0.5).sum(dim=-1).mean()
                
                loss = recon_loss + cfg.beta_y * kl_y + cfg.beta_z * kl_z
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(optimizer)
            lr_scheduler.step()
            scaler.update()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss_discrete += kl_y.item()
            total_kl_loss_continuous += kl_z.item()
            
            if (i % cfg.log_frequency) == 0:
                step_log = {'train/epoch': epoch,
                            'train/global_step': global_step,
                            'train/loss': loss.item(),
                            'train/reconstruction_loss': recon_loss.item(),
                            'train/kl_y_loss': kl_y.item(),
                            'train/kl_z_loss': kl_z.item(),
                            'train/lr': lr_scheduler.get_last_lr()[0],
                            'debug/has_nan_input': int(
                                torch.isnan(state).any().item() or torch.isnan(action).any().item() or torch.isnan(z).any().item()
                            )
                            }
                step_log.update(_stats_dict("debug/state", state))
                logger.log(step_log)
                if wb:
                    wandb.log(step_log, step=global_step)
            
            global_step += 1
            pbar.set_postfix({"Total Loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_dataloader)
        avg_recon_loss = total_recon_loss / len(train_dataloader)
        avg_kl_loss_y = total_kl_loss_discrete / len(train_dataloader)
        avg_kl_loss_z = total_kl_loss_continuous / len(train_dataloader)

        step_log = {"train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/loss": avg_loss,
                    "train/reconstruction_loss": avg_recon_loss,
                    "train/kl_y_loss": avg_kl_loss_y,
                    "train/kl_z_loss": avg_kl_loss_z,
                    "train/lr": lr_scheduler.get_last_lr()[0]}
        logger.log(step_log)
        if wb:
            wandb.log(step_log, step=global_step)
        
        if epoch % cfg.eval_frequency == 0:
            eval_loss_dict = eval(encoder=encoder, decoder=decoder, prior=prior,  dataloader=val_dataloader, epoch=epoch, num_epochs=cfg.num_epochs, beta_y=cfg.beta_y, beta_z=cfg.beta_z, device=cfg.device)
            
            eval_loss = eval_loss_dict['eval/loss']
            print(f"[Validation] Loss:  {eval_loss:.4f}")
            
            logger.log(eval_loss_dict)
            if wb:
                wandb.log(eval_loss_dict, step=global_step)
            
            if eval_loss <= best_loss:
                import gc
                gc.collect()
                
                print(f"Save best model of epoch {epoch} (loss={eval_loss:.4f})")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_loss_{eval_loss:.3f}.pt")
                torch.save({
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "prior_state_dict": prior.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": avg_loss,
                    "eval_loss": eval_loss
                }, checkpoint_path)
                best_loss = eval_loss
                
    print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")


@click.command()
@click.option('-c', '--config', required=True, default='train_hsovp', help='config file name')
def main(config):
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_cfg = OmegaConf.load(os.path.join(os.getcwd(), f'data/outputs/hsovp/{cfg.resume_ckpt_dir}/{config}.yaml'))
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
    
    dataset = FilteredDataset(seed=train_cfg.seed, cfg=data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    train_dataloader = DataLoader(train_dataset, batch_size=data_cfg.train_batch_size, shuffle=True, num_workers=data_cfg.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=data_cfg.val_batch_size, shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True)
    
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)
    
    # Create model
    encoder = Encoder(model_cfg)
    decoder = Decoder(model_cfg)
    prior = Prior(model_cfg)
    encoder.initialize()
    decoder.initialize()
    prior.initialize()
    
    ckpt = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"data/outputs/hsovp/{cfg.resume_ckpt_dir}/*.pt")))
        ckpt = torch.load(ckpts[-1])
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        prior.load_state_dict(ckpt['prior_state_dict'])
        if cfg.verbose:
            print(f"Resume from {ckpts[-1]}")
         
    checkpoint_dir = os.path.join(os.getcwd(), f"data/outputs/hsovp/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Created output directory: {checkpoint_dir}.")
    OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
    
    logger = JsonLogger(path=os.path.join(checkpoint_dir, 'log.json'))
    logger.start()
    if cfg.wb:
        wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = cfg.wandb_tag
        wandb.run.name = f"{cfg.wandb_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        train(encoder=encoder, 
            decoder=decoder, 
            prior=prior,
            scaler=scaler,
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader,
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