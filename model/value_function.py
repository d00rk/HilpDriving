import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from utils.utils import initialize_weights

class TwinQ(nn.Module):
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
            
        self.q1 = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim + action_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, 1),
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim + action_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, 1)
        )
        
    def both(self, state, action):
        s = self.feature_extractor(state)
        sa = torch.cat([s, action], dim=-1)
        q1 = self.q1(sa).squeeze(-1)
        q2 = self.q2(sa).squeeze(-1)
        return q1, q2
    
    def forward(self, state, action):
        return torch.min(*self.both(state, action))
    
    def initialize(self):
        self.apply(initialize_weights)
    
    
class ValueFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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
        self.v = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim, cfg.v_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.v_hidden_dim, cfg.v_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.v_hidden_dim, 1)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.v(x)
        x = x.squeeze(-1)
        return x
    
    def initialize(self):
        self.apply(initialize_weights)
    
class TwinQforHilbert(TwinQ):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.q1 = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim + cfg.latent_dim + cfg.action_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, 1),
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim + cfg.latent_dim + cfg.action_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, cfg.q_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.q_hidden_dim, 1)
        )
        
    def both(self, state, z, action):
        s = self.feature_extractor(state)
        sza = torch.cat([s, z, action], dim=-1)
        q1 = self.q1(sza).squeeze(-1)
        q2 = self.q2(sza).squeeze(-1)
        return q1, q2
    
    def forward(self, state, z, action):
        return torch.min(*self.both(state, z, action))
    
    def initialize(self):
        self.apply(initialize_weights)
    
class ValueFunctionforHilbert(ValueFunction):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.v = nn.Sequential(
            nn.Linear(cfg.obs_feature_dim + cfg.latent_dim, cfg.v_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.v_hidden_dim, cfg.v_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.v_hidden_dim, 1)
        )
        
    def forward(self, x, z):
        x = self.feature_extractor(x)
        xz = torch.cat([x, z], dim=-1)
        xz = self.v(xz)
        xz = xz.squeeze(-1)
        return xz
    
    def initialize(self):
        self.apply(initialize_weights)