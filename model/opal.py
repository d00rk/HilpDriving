import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from utils.utils import initialize_weights, kl_divergence

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
                          bidirectional=True, 
                          batch_first=True)
        self.fc_mu = nn.Linear(2 * cfg.gru_hidden_dim, cfg.latent_dim)
        self.fc_logstd = nn.Linear(2 * cfg.gru_hidden_dim, cfg.latent_dim)
        # print(f"Expected GRU input size: {cfg.obs_feature_dim + cfg.action_dim}")  # 66이어야 함

    def forward(self, states, actions):
        if states.dim() == 5 and states.shape[2] != 3 and states.shape[-1] == 3:
            states = states.permute(0, 1, 4, 2, 3)
        batch_size, seq_len, c, h, w = states.size()
        states = states.view(-1, c, h, w)
        features = self.feature_extractor(states)
        features = features.view(batch_size, seq_len, -1)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        torch.cuda.empty_cache()
        
        gru_input = torch.cat([features, actions], dim=-1)
        gru_output, _ = self.gru(gru_input)
        final_output = gru_output[:, -1, :]
        
        mu = self.fc_mu(final_output)
        logstd = self.fc_logstd(final_output)
        logstd = torch.clamp(logstd, min=-10, max=10)
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)

# Decoder (Primitive Policy)
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
        self.fc = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim + cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.fc_logstd = nn.Linear(cfg.hidden_dim, cfg.action_dim)

    def forward(self, state, latent):
        if state.dim() == 4 and state.shape[1] != 3 and state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)
        batch_size, c, h, w = state.shape
        state_features = self.feature_extractor(state)
        
        x = torch.cat([state_features, latent], dim=-1)
        x = self.fc(x)
        
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)

# Prior
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
        
        self.fc_mu = nn.Linear(cfg.obs_feature_dim, cfg.latent_dim)
        self.fc_logstd = nn.Linear(cfg.obs_feature_dim, cfg.latent_dim)

    def forward(self, state):
        x = self.feature_extractor(state)
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)

