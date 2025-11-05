import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import initialize_weights

# Define the neural network for the Hilbert representation
class HilbertRepresentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        
        self.feature_extractor = nn.Sequential(
            # nn.BatchNorm2d(3),
            nn.Conv2d(3, 8, 4, 4, 0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 4, 4, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32*4*4, self.latent_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.feature_extractor(x)
    
    def initialize(self):
        self.apply(initialize_weights)


# Define the latent-conditioned policy network
class LatentConditionedPolicy(nn.Module):
    def __init__(self, config):
        super(LatentConditionedPolicy, self).__init__()
        self.latent_dim = config.latent_dim
        self.action_dim = config.action_dim
        
        self.policy_net = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh()
        )

    def forward(self, latent_state, latent_goal):
        x = torch.cat([latent_state, latent_goal], dim=-1)
        return self.policy_net(x)
