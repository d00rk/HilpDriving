import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import initialize_weights

# Encoder
class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.feature_extractor = nn.Sequential(
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
            nn.Linear(32*4*4, cfg.obs_feature_dim),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(input_size=cfg.obs_feature_dim + cfg.action_dim, 
                         hidden_size=cfg.gru_hidden_dim, 
                         num_layers=cfg.gru_layers, 
                         batch_first=True, 
                         bidirectional=True)
        
        self.discrete_fc = nn.Linear(cfg.gru_hidden_dim * 2, cfg.discrete_option)
        self.continuous_mean = nn.ModuleList([nn.Linear(cfg.gru_hidden_dim * 2, cfg.latent_dim) for _ in range(cfg.discrete_option)])
        self.continuous_logstd = nn.ModuleList([nn.Linear(cfg.gru_hidden_dim * 2, cfg.latent_dim) for _ in range(cfg.discrete_option)])
        
        
    def forward(self, state, action):
        if state.dim() == 5 and state.shape[2] != 3 and state.shape[-1] == 3:
            state = state.permute(0, 1, 4, 2, 3)
        batch_size, seq_len, c, h, w = state.shape
        state = state.view(-1, c, h, w)
        state_features = self.feature_extractor(state)
        state_features = state_features.view(batch_size, seq_len, -1)
        state_features = torch.nan_to_num(state_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        torch.cuda.empty_cache()
        
        gru_input = torch.cat([state_features, action], dim=-1)
        gru_output, _ = self.gru(gru_input)
        gru_output = torch.nan_to_num(gru_output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logits = self.discrete_fc(gru_output[:, -1, :])
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        discrete_y = F.gumbel_softmax(logits, tau=1.0, hard=True)
        discrete_y = torch.nan_to_num(discrete_y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        z_means = torch.stack([layer(gru_output[:, -1, :]) for layer in self.continuous_mean])
        z_log_stds = torch.stack([layer(gru_output[:, -1, :]) for layer in self.continuous_logstd])
        
        z_means = z_means.permute(1, 0, 2)
        z_log_stds = z_log_stds.permute(1, 0, 2)
        z_log_stds = torch.clamp(z_log_stds, min=-2, max=2)
        z_stds = F.softplus(z_log_stds) + 1e-6
        
        z_mean = (discrete_y.unsqueeze(-1) * z_means).sum(dim=1)
        z_std = (discrete_y.unsqueeze(-1) * z_stds).sum(dim=1)
        
        z_mean = torch.nan_to_num(z_mean, nan=0.0, posinf=1.0, neginf=-1.0)
        z_std = torch.nan_to_num(z_std, nan=1e-3, posinf=1.0, neginf=-1.0)
        
        return z_mean, z_std, logits, discrete_y
    
    def initialize(self):
        self.apply(initialize_weights)
    
    
class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.feature_extractor = nn.Sequential(
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
            nn.Linear(32*4*4, cfg.obs_feature_dim),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(input_size=cfg.obs_feature_dim + cfg.latent_dim, 
                          hidden_size=cfg.gru_hidden_dim,
                          num_layers=cfg.gru_layers,
                          bidirectional=True,
                          batch_first=True)
        self.decoder = nn.Linear(cfg.gru_hidden_dim * 2, cfg.action_dim)
        
        
    def forward(self, state, z):
        if state.dim() == 5 and state.shape[2] != 3 and state.shape[-1] == 3:
            state = state.permute(0, 1, 4, 2, 3)
        batch_size, seq_len, c, h, w = state.shape
        state = state.view(batch_size * seq_len, c, h, w)
        state_features = self.feature_extractor(state)
        state_features = state_features.view(batch_size, seq_len, -1)
        state_features = torch.nan_to_num(state_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        torch.cuda.empty_cache()
        
        z_expand = z.unsqueeze(1).expand(-1, seq_len, -1)
        
        gru_input = torch.cat([state_features, z_expand], dim=-1)
        gru_output, _ = self.gru(gru_input)
        gru_output = torch.nan_to_num(gru_output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        action = self.decoder(gru_output)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return action
    
    def initialize(self):
        self.apply(initialize_weights)
    
    
    
class Prior(nn.Module):
    def __init__(self, cfg):
        super(Prior, self).__init__()
        
        self.feature_extractor = nn.Sequential(
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
            nn.Linear(32*4*4, cfg.obs_feature_dim),
            nn.ReLU()
        )
        
        self.discrete_fc = nn.Linear(cfg.obs_feature_dim, cfg.discrete_option)
        self.continuous_mean = nn.ModuleList([nn.Linear(cfg.obs_feature_dim, cfg.latent_dim) for _ in range(cfg.discrete_option)])
        self.continuous_logstd = nn.ModuleList([nn.Linear(cfg.obs_feature_dim, cfg.latent_dim) for _ in range(cfg.discrete_option)])
        
    def forward(self, state):
        if state.dim() == 4 and state.shape[1] != 3 and state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)
        batch_size, c, h, w = state.shape
        state_features = self.feature_extractor(state)
        state_features = torch.nan_to_num(state_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logits = self.discrete_fc(state_features)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        discrete_y = F.gumbel_softmax(logits, tau=1.0, hard=True)
        discrete_y = torch.nan_to_num(discrete_y, nan=0.0, posinf=1.0, neginf=-1.0)
        z_means = torch.stack([layer(state_features) for layer in self.continuous_mean])
        z_log_stds = torch.stack([layer(state_features) for layer in self.continuous_logstd])
        
        z_means = z_means.permute(1, 0, 2)
        z_log_stds = z_log_stds.permute(1, 0, 2)
        z_log_stds = torch.clamp(z_log_stds, min=-10, max=2)
        z_stds = torch.clamp(torch.exp(z_log_stds), min=1e-3)
        
        z_mean = (discrete_y.unsqueeze(-1) * z_means).sum(dim=1)
        z_std = (discrete_y.unsqueeze(-1) * z_stds).sum(dim=1)
        
        z_mean = torch.nan_to_num(z_mean, nan=0.0, posinf=1.0, neginf=-1.0)
        z_std = torch.nan_to_num(z_std, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return z_mean, z_std, logits, discrete_y
    
    def initialize(self):
        self.apply(initialize_weights)