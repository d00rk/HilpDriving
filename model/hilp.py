import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x
    
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


class ViewAwareHilbertRepresentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        
        self.num_views = 3 
        
        # 1. Shared Image Encoder (Lightweight CNN)
        self.feature_dim = 256 # Transformer dimension
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # [32, 32, 32]
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [64, 16, 16]
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # [128, 8, 8]
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),# [128, 4, 4]
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # enforce fixed spatial size before flatten
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh() # Feature bounded
        )

        # 2. Learnable View Embeddings & Positional Encodings
        self.view_embs = nn.Parameter(torch.randn(1, 1, self.num_views, self.feature_dim))
        
        # 3. Spatiotemporal Transformer
        # Input Sequence: [Batch, Time * Views, Feature_Dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ---------------------------------------------------------
        # Head 1: Hilbert Head (For Policy & Planning)
        # produce z vector (in hilbert space)
        # ---------------------------------------------------------
        self.hilbert_head = nn.Sequential(
            nn.Linear(self.feature_dim * self.num_views, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim) # z dimension (e.g., 32)
        )

        # ---------------------------------------------------------
        # Head 2: BEV Decoder (Auxiliary Task)
        # reconstruction BEV image from transformer output
        # ---------------------------------------------------------
        self.bev_decoder = nn.Sequential(
            nn.Linear(self.feature_dim * self.num_views, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)), # [B, 128, 4, 4]
            
            # Upsampling layers (Deconvolution)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # [64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # [16, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),   # [3, 64, 64] (RGB BEV or Semantic)
            nn.Sigmoid()
        )

    def forward(self, images):
        """
        Args:
            images: Dictionary {'front': [B, T, C, H, W], 'left': ..., 'right': ...}
        Returns:
            z: Hilbert representation [B, Latent_Dim]
            bev_pred: Reconstructed BEV [B, 3, 64, 64]
        """
        # 1. Data Preparation
        # Stack views: [B, T, 3, C, H, W]
        img_stack = torch.stack([images['front_rgb'], images['left_rgb'], images['right_rgb']], dim=2)
        b, t, v, c, h, w = img_stack.shape
        
        # Collapse dims for CNN: [B*T*V, C, H, W]
        flat_imgs = img_stack.view(-1, c, h, w)
        
        # 2. CNN Encoding
        feats = self.encoder(flat_imgs) # [B*T*V, Feature_Dim]
        
        # Reshape back to Sequence: [B, T, V, Feature_Dim]
        feats = feats.view(b, t, v, -1)
        
        # 3. Add View Embeddings (Broadcasting along Batch and Time)
        # feats: [B, T, 3, D] + view_embs: [1, 1, 3, D]
        feats = feats + self.view_embs
        
        # 4. Transformer Processing
        # Flatten Time and View to make a single sequence
        # Seq Len = T * V. (Example: History 3 * View 3 = 9 tokens)
        transformer_input = feats.view(b, t * v, -1) 
        
        # Apply Transformer (Self-Attention across time and views)
        # Output: [B, T*V, Feature_Dim]
        out = self.transformer(transformer_input)
        
        # 5. Aggregation for Heads
        # Reshape back: [B, T, V, D]
        out = out.view(b, t, v, -1)
        
        last_step_feats = out[:, -1, :, :] 
        
        # Flatten views: [B, 3 * D]
        context_vector = last_step_feats.view(b, -1)
        
        # ---------------------------------------------------------
        # Head 1 Output: Hilbert Latent z
        # ---------------------------------------------------------
        z = self.hilbert_head(context_vector)
        z = F.normalize(z, dim=-1) # HILP requires unit norm
        
        # ---------------------------------------------------------
        # Head 2 Output: BEV Prediction
        # ---------------------------------------------------------
        bev_pred = self.bev_decoder(context_vector)
        
        return z, bev_pred

    def initialize(self):
        self.apply(initialize_weights)
        nn.init.trunc_normal_(self.view_embs, std=0.02)
