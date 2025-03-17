import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import initialize_weights

# Encoder
class Encoder(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 latent_dim, 
                 discrete_option):
        super(Encoder, self).__init__()
        
        self.bev_layer = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, 4, 4, 0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 4, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32*4*4, 16),
            nn.ReLU()
        )
        
        self.rnn = nn.GRU(input_size=16 + action_dim, 
                         hidden_size=16, 
                         batch_first=True, 
                         bidirectional=True)
        
        self.discrete_fc = nn.Linear(16*2, discrete_option)
        self.continuous_mean = nn.ModuleList([nn.Linear(16*2, latent_dim) for _ in range(discrete_option)])
        self.continuous_logstd = nn.ModuleList([nn.Linear(16*2, latent_dim) for _ in range(discrete_option)])
        
    def forward(self, state, action):
        batch_size, seq_len, c, h, w = state.shape
        state = state.view(-1, c, h, w)
        state_features = self.bev_layer(state)
        state_features = state_features.view(batch_size, seq_len, -1)
        state_features = torch.nan_to_num(state_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        rnn_input = torch.cat([state_features, action], dim=-1)
        rnn_output, _ = self.rnn(rnn_input)
        
        logits = self.discrete_fc(rnn_output[:, -1, :])
        discrete_y = F.gumbel_softmax(logits, tau=1.0, hard=True)
        
        z_means = torch.stack([layer(rnn_output[:, -1, :]) for layer in self.continuous_mean])
        z_log_stds = torch.stack([layer(rnn_output[:, -1, :]) for layer in self.continuous_logstd])
        
        z_means = z_means.permute(1, 0, 2)
        z_log_stds = z_log_stds.permute(1, 0, 2)
        
        z_stds = torch.exp(z_log_stds)
        
        z_mean = (discrete_y.unsqueeze(-1) * z_means).sum(dim=1)
        z_std = (discrete_y.unsqueeze(-1) * z_stds).sum(dim=1)
        z = z_mean + z_std * torch.randn_like(z_std)
        
        return z, logits, z_mean, z_std
    
    def initialize(self):
        self.apply(initialize_weights)
    
    
class Decoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, seq_len=10):
        super(Decoder, self).__init__()
        
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        self.bev_layer = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, 4, 4, 0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 4, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32*4*4, latent_dim),
            nn.ReLU()
        )
        
        self.gru = nn.GRUCell(input_size=2*latent_dim, hidden_size=latent_dim)
        self.decoder = nn.Linear(latent_dim, action_dim)
        
    def forward(self, bev, z):
        batch_size = bev.size(0)
        bev_features = self.bev_layer(bev)
        bev_features = torch.nan_to_num(bev_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        actions = []
        hidden_state = torch.zeros(batch_size, self.latent_dim, device=bev.device)
        
        for t in range(self.seq_len):
            gru_input = torch.cat([bev_features, z], dim=-1)
            hidden_state = self.gru(gru_input, hidden_state)
            action = self.decoder(hidden_state)
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)
        return actions
    
    def initialize(self):
        self.apply(initialize_weights)
    
    
class Prior(nn.Module):
    def __init__(self, state_dim, latent_dim, discrete_option):
        super(Prior, self).__init__()
        
        self.bev_layer = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, 4, 4, 0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, 4, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32*4*4, 256),
            nn.ReLU()
        )
        
        self.discrete_fc = nn.Linear(256, discrete_option)
        self.continuous_mean = nn.ModuleList([nn.Linear(256, latent_dim) for _ in range(discrete_option)])
        self.continuous_logstd = nn.ModuleList([nn.Linear(256, latent_dim) for _ in range(discrete_option)])
        
    def forward(self, bev):
        bev_features = self.bev_layer(bev)
        bev_features = torch.nan_to_num(bev_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logits = self.discrete_fc(bev_features)
        discrete_y = F.gumbel_softmax(logits, tau=1.0, hard=True)
        
        z_means = torch.stack([layer(bev_features) for layer in self.continuous_mean])
        z_log_stds = torch.stack([layer(bev_features) for layer in self.continuous_logstd])
        
        z_means = z_means.permute(1, 0, 2)
        z_log_stds = z_log_stds.permute(1, 0, 2)
        
        z_stds = torch.exp(z_log_stds)
        
        z_mean = (discrete_y.unsqueeze(-1) * z_means).sum(dim=1)
        z_std = (discrete_y.unsqueeze(-1) * z_stds).sum(dim=1)
        z = z_mean + z_std * torch.randn_like(z_std)
        
        return z, logits, z_mean, z_std
    
    def initialize(self):
        self.apply(initialize_weights)