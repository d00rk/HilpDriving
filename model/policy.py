import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from utils.utils import initialize_weights

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

"""
Gaussian Policy: pi(a|s)
"""
class GaussianPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        action_dim = cfg.latent_dim if cfg.latent_as_action else cfg.action_dim
        
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
        
        self.fc_mu = nn.Linear(cfg.obs_feature_dim, action_dim)
        self.fc_logstd = nn.Linear(cfg.obs_feature_dim, action_dim)
        
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.feature_extractor(x)
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=1.0, neginf=-1.0)
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)
       
       
"""
Gaussian Policy: pi(a|s, z)
"""
class GaussianPolicyforHilbert(GaussianPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fc = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim + cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x, z):
        if x.dim() == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.feature_extractor(x)
        xz = torch.cat([x, z], dim=-1)
        xz = self.fc(xz)
        mu = self.fc_mu(xz)
        logstd = self.fc_logstd(xz)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=1.0, neginf=-1.0)
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)