import torch
import numpy as np

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio < 0:
        return val_mask
    
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

def sample_latent_vectors(batch_size, latent_dim):
    z = torch.randn(batch_size, latent_dim)
    z = z / torch.norm(z, dim=1, keepdim=True)
    return z