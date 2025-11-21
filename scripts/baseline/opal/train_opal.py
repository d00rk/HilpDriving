import sys
import os
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP


from model.opal import *
from dataset.dataset import SubTrajDataset
from utils.seed_utils import seed_all
from utils.logger import JsonLogger
from utils.utils import gaussian_nll, kl_diag_gaussians_logvar
from utils.ddp import is_dist_avail_and_initialized, get_rank, is_main_process, setup_distributed, cleanup_distributed, unwrap
from utils.multiprocessing import _worker_init_fn


def eval(
    encoder,
    decoder,
    prior,
    dataloader, 
    kl_weight,
    epoch,
    cfg
    ):
    
    encoder = unwrap(encoder)
    decoder = unwrap(decoder)
    prior = unwrap(prior)
    
    encoder.eval()
    decoder.eval()
    prior.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    pbar = enumerate(dataloader)
    if is_main_process():
        pbar = tqdm(
            pbar, 
            total=len(dataloader), 
            desc=f"[Validation] {epoch}/{cfg.num_epochs}", 
            leave=False, 
            ncols=100
            )
    
    for i, (state, action, _, _, _, _) in pbar:
        state = state.to(cfg.device, non_blocking=True)
        action = action.to(cfg.device, non_blocking=True)
        
        B, L = action.shape
        latent_mu, latent_logstd = encoder(state, action)
        latent_std = torch.exp(0.5 * torch.clamp(latent_logstd, min=-10, max=10))
        latent_std = torch.clamp(latent_std, min=1e-6)
        z = Normal(latent_mu, latent_std).rsample()
        
        prior_mu, prior_logstd = prior(state[:, 0, :])
        
        recon_nll_list = []
        for t in range(L):
            a_mu_t, a_logstd_t = decoder(state[:, t, :], z)
            nll_t = gaussian_nll(action[:, t, :], a_mu_t, a_logstd_t)
            recon_nll_list.append(nll_t)
        recon_nll = torch.stack(recon_nll_list, dim=1).mean()
        
        kl = kl_diag_gaussians_logvar(latent_mu, latent_logstd, prior_mu, prior_logstd).mean()
        
        loss = recon_nll + kl_weight * kl
        
        total_loss += float(loss.item())
        total_recon_loss += float(recon_nll.item())
        total_kl_loss += float(kl.item())
        
        num_batches += 1
    
    device0 = cfg.device
    t = torch.tensor([total_loss, total_recon_loss, total_kl_loss, num_batches], dtype=torch.float64, device=device0)
    if is_dist_avail_and_initialized():
        distributed.all_reduce(t, op=distributed.ReduceOp.SUM)
    
    sum_loss, sum_recon, sum_kl, sum_batches = t.tolist()
    if sum_batches == 0:
        return 0.0, 0.0, 0.0
        
    return sum_loss / sum_batches, sum_recon / sum_batches, sum_kl / sum_batches


def train(
    encoder, 
    decoder, 
    prior, 
    scaler,
    train_dataloader, 
    val_dataloader,
    wb,
    checkpoint_dir,
    cfg,
    logger,
    ckpt
    ):
    device = cfg.device
    
    if is_dist_avail_and_initialized():
        encoder = DDP(encoder.to(device), device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), find_unused_parameters=False)
        decoder = DDP(decoder.to(device), device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), find_unused_parameters=False)
        prior = DDP(prior.to(device), device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), find_unused_parameters=False)
    else:
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        prior = prior.to(device)

    model_params = list(encoder.parameters()) + list(decoder.parameters()) + list(prior.parameters())
    optimizer = optim.AdamW(model_params, lr=cfg.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs*len(train_dataloader))
    
    if ckpt is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        del ckpt
    
    if is_main_process():
        print(f"[Torch] {device} is used.")
        print(f"[Train] Start at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")
    
    best_loss = np.inf
    global_step = 0
    
    train_sampler = train_dataloader.sampler if isinstance(train_dataloader.sampler, DistributedSampler) else None
    val_sampler   = val_dataloader.sampler if isinstance(val_dataloader.sampler, DistributedSampler) else None

    for epoch in range(cfg.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        encoder.train()
        decoder.train()
        prior.train()
        
        pbar = enumerate(train_dataloader)
        if is_main_process():
            pbar = tqdm(
                pbar, 
                total=len(train_dataloader), 
                desc=f"[Epoch {epoch}/{cfg.num_epochs}]", 
                leave=False, 
                ncols=100
                )
        
        for i, (state, action, next_state, _, terminal, _) in pbar:
            state = state.to(device, non_blocking=True)                 # (B, L, C, H, W)
            action = action.to(device, non_blocking=True)               # (B, L, A)
            next_state = next_state.to(device, non_blocking=True)       # (B, L, C, H, W)
            terminal = terminal.to(device, non_blocking=True)           # (B, L)
            
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                B, L, A = action.shape
                latent_mu, latent_logstd = encoder(state, action)
                latent_std = torch.exp(0.5 * torch.clamp(latent_logstd, min=-10, max=10))
                latent_std = torch.clamp(latent_std, min=1e-6)
                z = Normal(latent_mu, latent_std).rsample()
                
                prior_mu, prior_logstd = prior(state[:, 0, :])
                
                recon_nll_list = []
                for t in range(L):
                    a_mu_t, a_logstd_t = decoder(state[:, t, :], z)
                    nll_t = gaussian_nll(action[:, t, :], a_mu_t, a_logstd_t)
                    recon_nll_list.append(nll_t)
                recon_nll = torch.stack(recon_nll_list, dim=1).mean()
                
                kl = kl_diag_gaussians_logvar(latent_mu, latent_logstd, prior_mu, prior_logstd).mean()
                
                loss = recon_nll + cfg.kl_weight * kl
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            lr_scheduler.step()
            
            scaler.update()
            
            total_loss += loss.item()
            total_recon_loss += recon_nll.item()
            total_kl_loss += kl.item()
            
            if is_main_process():
                step_log = {
                    'train/epoch': epoch,
                    'train/global_step': global_step,
                    'train/loss': loss.item(),
                    'train/recon_loss': recon_nll.item(),
                    'train/kl': kl.item(),
                    'train/lr': lr_scheduler.get_last_lr()[0]
                }
                logger.log(step_log)
                if wb:
                    wandb.log(step_log, step=global_step)
                if isinstance(pbar, tqdm.tqdm):
                    pbar.set_postfix({"Total Loss": f"{loss.item():.4f}"})
            
            global_step += 1
        
        # Reduce epoch stats across ranks (average)
        t = torch.tensor([total_loss, total_recon_loss, total_kl_loss, len(train_dataloader)], dtype=torch.float64, device=device)
        if is_dist_avail_and_initialized():
            distributed.all_reduce(t, op=distributed.ReduceOp.SUM)
        sum_loss, sum_recon, sum_kl, sum_batches = t.tolist()
        
        avg_loss = sum_loss / max(1, sum_batches)
        avg_recon_loss = sum_recon / max(1, sum_batches)
        avg_kl_loss = sum_kl / max(1, sum_batches)
        if is_main_process():
            step_log = {
                'train/epoch': epoch,
                'train/global_step': global_step,
                'train/loss': avg_loss,
                'train/recon_loss': avg_recon_loss,
                'train/kl': avg_kl_loss
            }
            
            logger.log(step_log)
            if wb:
                wandb.log(step_log, step=global_step)
        
        if epoch % cfg.eval_frequency == 0:
            eval_loss_dict = eval(encoder=encoder, 
                                  decoder=decoder, 
                                  prior=prior, 
                                  dataloader=val_dataloader, 
                                  kl_weight=cfg.kl_weight, 
                                  epoch=epoch,
                                  cfg=cfg)
            
            if is_main_process():
                logger.log(eval_loss_dict)
                if wb:
                    wandb.log(eval_loss_dict, step=global_step)

                eval_loss = eval_loss_dict['eval/loss']
                if eval_loss <= best_loss:
                    print(f"Save best model of epoch: {epoch} (loss={eval_loss:.4f})")
                    
                    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_loss_{eval_loss:.3f}.pt")
                    
                    enc_sd = encoder.module.state_dict() if isinstance(encoder, DDP) else encoder.state_dict()
                    dec_sd = decoder.module.state_dict() if isinstance(decoder, DDP) else decoder.state_dict()
                    pri_sd = prior.module.state_dict() if isinstance(prior, DDP) else prior.state_dict()
                    
                    torch.save({
                        'epoch': epoch,
                        'encoder_state_dict': enc_sd,
                        'decoder_state_dict': dec_sd,
                        'prior_state_dict': pri_sd,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'loss': avg_loss,
                        'eval_loss': eval_loss
                    }, checkpoint_path)
                    best_loss = eval_loss
    
    if is_main_process():
        print(f"[Train] Finished at {dt.datetime.now().strftime('%Y_%m_%d %H:%M:%S')}")


@click.command()
@click.option("-c", "--config", type=str, default='train_opal', required=True, help="config file name")
def main(config):
    ddp_enabled = setup_distributed()
    
    CONFIG_FILE = os.path.join(os.getcwd(), f'config/{config}.yaml')
    cfg = OmegaConf.load(CONFIG_FILE)
    
    if cfg.resume:
        resume_conf = OmegaConf.load(os.path.join(os.getcwd(), f'data/outputs/opal/{cfg.resume_ckpt_dir}/{config}.yaml'))
        cfg.data = resume_conf.data
        cfg.model = resume_conf.model
        cfg.train = resume_conf.train
        del resume_conf
    
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    if ddp_enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        device = train_cfg.device
    train_cfg.device = device
    
    base_seed = int(train_cfg.seed)
    rank = get_rank()
    seed_all(base_seed + rank)
        
    dataset = SubTrajDataset(seed=train_cfg.seed, cfg=data_cfg)
    train_dataset, val_dataset = dataset.split_train_val()
    
    # Samplers for DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if ddp_enabled else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if ddp_enabled else None
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg.train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=data_cfg.persistent_workers,
        prefetch_factor=data_cfg.prefetch_factor
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=data_cfg.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        worker_init_fn=_worker_init_fn,
        persistent_workers=data_cfg.persistent_workers,
        prefetch_factor=data_cfg.prefetch_factor       
    )
    
    # Create model
    encoder = Encoder(model_cfg)
    decoder = Decoder(model_cfg)
    prior = Prior(model_cfg)
    encoder.initialize()
    decoder.initialize()
    prior.initialize()
    
    ckpt = None
    if cfg.resume:
        ckpts = sorted(glob.glob(os.path.join(os.getcwd(), f"data/outputs/opal/{cfg.resume_ckpt_dir}/*.pt")))
        ckpt = torch.load(ckpts[-1])
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        prior.load_state_dict(ckpt['prior_state_dict'])
        if ddp_enabled:
            for m in [encoder, decoder, prior]:
                for p in m.parameters():
                    distributed.broadcast(p.data, src=0)
    
    scaler = torch.cuda.amp.GradScaler()
    
    if cfg.wb and is_main_process():
        wandb.init(project=cfg.wandb_project,
                   config=OmegaConf.to_container(cfg, resolve=True))
        wandb.run.tags = cfg.wandb_tag
        wandb.run.name = f"{cfg.wandb_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if is_main_process():
        checkpoint_dir = os.path.join(os.getcwd(), f"data/outputs/opal/{dt.datetime.now().strftime('%Y_%m_%d')}/{dt.datetime.now().strftime('%H_%M_%S')}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if cfg.verbose:
            print(f"Created output directory: {checkpoint_dir}.")
        OmegaConf.save(cfg, os.path.join(checkpoint_dir, f"{config}.yaml"))
    else:
        checkpoint_dir = None
    
    if is_dist_avail_and_initialized():
        distributed.barrier()
        # Broadcast checkpoint_dir path from rank0
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
        checkpoint_dir = bytes(buf.cpu().numpy().tolist()).decode("utf-8")

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
        cleanup_distributed()


if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()