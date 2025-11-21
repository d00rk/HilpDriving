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
        x = self.feature_extractor(x)
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=1.0, neginf=-1.0)
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)
       
       
"""
Conditioned Gaussian Policy: pi(a|s, z)
"""
class ConditionedGaussianPolicy(GaussianPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fc = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim + cfg.condition_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.fc_logstd = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        
    def forward(self, x, z):
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


"""
Goal-Conditioned Gaussian Policy: pi(a|s, g)
"""
class GoalConditionedGaussianPolicy(GaussianPolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fc = nn.Sequential(
            nn.Linear(2*cfg.obs_feature_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.fc_logstd = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        
    def forward(self, x, g):
        x = self.feature_extractor(x)
        g = self.feature_extractor(g)
        xg = torch.cat([x, g], dim=-1)
        xg = self.fc(xg)
        mu = self.fc_mu(xg)
        logstd = self.fc_logstd(xg)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=1.0, neginf=-1.0)
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)


class MLPGaussianPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pass


class MLPConditionedGaussianPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.latent_dim + cfg.latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.action_dim)
        self.fc_logstd = nn.Linear(cfg.hidden_dim, cfg.action_dim)
    
    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        x = self.net(x)
        
        mu = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)


class MLPGoalConditionedGaussianPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        input_dim = cfg.latent_dim + cfg.latent_dim
        
        output_dim = cfg.latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(cfg.hidden_dim, output_dim)
        self.fc_logstd = nn.Linear(cfg.hidden_dim, output_dim)
    
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = self.net(x)
        
        mu = self.gc_mu(x)
        logstd = self.fc_logstd(x)
        
        return mu, logstd
    
    def initialize(self):
        self.apply(initialize_weights)