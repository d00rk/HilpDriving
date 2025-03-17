import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal

import numpy as np
import h5py
import wandb
import datetime as dt

from model.opal import *
from dataset.dataset import *

def finetune_decoder(decoder, dataloader, num_epochs, lr, device, checkpoint_dir):
    decoder = decoder.to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    
    best_loss = np.inf
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for transition in dataloader: 
            print(transition)
            print(transition.shape)
            states, actions, z = zip(*transition)
            batch_size, seq_len, _ = states.size()
            
            optimizer.zero_grad()
            loss = 0
            for t in range(len(seq_len)):
                state_t = states[:, t, :]
                action_t = actions[:, t]
                mu, logstd = decoder(state_t, z)
                pred_action = Normal(mu, torch.exp(0.5 * logstd)).rsample()
                loss += nn.CrossEntropyLoss(pred_action, action_t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch}/{num_epochs}   |   Loss: {avg_loss:.4f}")
        
        wandb.log({"train/epoch": epoch,
                   "train/loss": avg_loss})
        
        if avg_loss <= best_loss:
            print(f"Save best model of epoch {epoch}")
            checkpoint_path = os.path.join(checkpoint_dir, f"decoder_finetune_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            best_loss = avg_loss
    
    
def main(data_root, opal_path, state_dim, action_dim, latent_dim, checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[torch] {device} is used.")
    
    data = h5py.File(data_root)
    opal_cpt = torch.load(opal_path)
    
    opal_encoder = Encoder(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim)
    opal_encoder.load_state_dict(opal_cpt['encoder_state_dict'])
    opal_decoder = Decoder(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim)
    opal_decoder.load_state_dict(opal_cpt['decoder_state_dict'])
    
    print('Creating Dataset...')
    subtraj_dataset = SubTrajDataset(data, 10)
    lowlevel_dataset = LowLevelDataset(subtraj_dataset, opal_encoder)   
    lowlevel_dataloader = DataLoader(lowlevel_dataset, batch_size=50, shuffle=True)
    
    print('[Train] Start')
    finetune_decoder(decoder=opal_decoder, dataloader=lowlevel_dataloader, num_epochs=200, device=device, checkpoint_dir=checkpoint_dir)
    
if __name__=="__main__":
    DATA_ROOT = os.path.join(os.path.dirname(os.getcwd()), 'd4rl/carla_lane_follow_flat-v0.hdf5')
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.getcwd()), 'checkpoint')
    OPAL_ROOT = os.path.join(CHECKPOINT_DIR, 'epoch_80.pt')
    
    wandb.init(project="carla_opal")
    wandb.run.name = f"opal-finetune-{dt.datetime.now().replace(microsecond=0)}"
    
    state_dim = 6912
    action_dim = 2
    latent_dim = 8
    
    main(data_root=DATA_ROOT, opal_path=OPAL_ROOT, state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim, checkpoint_dir=CHECKPOINT_DIR)