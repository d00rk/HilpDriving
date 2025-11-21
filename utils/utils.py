import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import xml.etree.ElementTree as ET
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np


@torch.no_grad()
def compute_z(hilbert, state, goal, eps=1e-8):
    phi_s = hilbert(state)
    phi_g = hilbert(goal)
    vec = phi_g - phi_s
    z = vec / (vec.norm(dim=-1, keepdim=True) + eps)
    return z

def cosine_align_loss(mu, z, eps=1e-8):
    mu = mu / (mu.norm(dim=-1, keepdim=True) + eps)
    z = z / (z.norm(dim=-1, keepdim=True) + eps)
    return (1.0 - (mu * z).sum(dim=-1)).mean()

def get_lambda(cfg, epoch):
    base = cfg.get('lambda_align', 0.0)
    schedule = cfg.get('aling_schedule', 'cosine')
    if base <= 0.0:
        return 0.0
    if schedule == 'cosine':
        return float(base * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, cfg.num_epochs))))
    elif schedule == 'constant':
        return float(base)
    else:
        return float(base)
    
def kl_divergence(mu1, logstd1, mu2, logstd2):
    kl = 0.5 * (logstd2 - logstd1 + (torch.exp(logstd1) + (mu1 - mu2).pow(2)) / torch.exp(logstd2) - 1)
    return kl.sum()

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def l2_expectile_loss(x, tau):
    tau_t = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
    weight = torch.where(x < 0, 1.0-tau_t, tau_t)
    return torch.mean(weight * (x ** 2))

def update_exponential_moving_average(target, online, alpha):
    online_ref = online.module if isinstance(online, DDP) else online
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), online_ref.parameters()):
            target_param.data.mul_(1.0 - alpha).add_(source_param.data, alpha=alpha)
        
def log_sum_exp(tensor, dim=1, keepdim=False):
    m, _ = torch.max(tensor, dim=dim, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(tensor - m), dim=dim, keepdim=True))

def normalize_control(control):
    return float(np.clip(control, -1.0, 1.0))

def initialize_weights(m):
    """
    Initialize weights for various layers including CNN, RNN, and Transformer.
    """
    # 1. Linear & Convolutional Layers (Include ConvTranspose2d for BEV Decoder)
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    # 2. Recurrent Layers (GRU, LSTM)
    elif isinstance(m, (nn.GRU, nn.GRUCell, nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    # 3. Normalization Layers (LayerNorm used in Transformer, BatchNorm)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # 4. Transformer MultiheadAttention
    # (TransformerEncoderLayer internally uses MultiheadAttention)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            nn.init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)


def get_reward_statics(dataloader):
    rewards = list()
    for _, _, _, r, _, _ in dataloader:
        rewards.append(r)
    rewards = torch.cat(rewards, dim=0)
    return rewards.mean(), rewards.std()

def parse_route_xml(route_xml):
    tree = ET.parse(route_xml)
    root = tree.getroot()
    
    all_routes = []
    for route_elem in root.findall('route'):
        route_id = route_elem.get('id', '')
        town = route_elem.get('town', '')
        
        weathers_data = []
        weathers_elem = route_elem.find('weathers')
        if weathers_elem is not None:
            for w_elem in weathers_elem.findall('weather'):
                w_data = {
                    'route_percentage': float(w_elem.get('route_percentage', 0.0)),
                    'cloudiness': float(w_elem.get('cloudiness', 0.0)),
                    'precipitation': float(w_elem.get('precipitation', 0.0)),
                    'precipitation_deposits': float(w_elem.get('precipitation_deposits', 0.0)),
                    'wetness': float(w_elem.get('wetness', 0.0)),
                    'wind_intensity': float(w_elem.get('wind_intensity', 0.0)),
                    'sun_azimuth_angle': float(w_elem.get('sun_azimuth_angle', 0.0)),
                    'sun_altitude_angle': float(w_elem.get('sun_altitude_angle', 0.0)),
                    'fog_density': float(w_elem.get('fog_density', 0.0)),
                }
                weathers_data.append(w_data)
        
        waypoints_data = []
        wpts_elem = route_elem.find('waypoints')
        if wpts_elem is not None:
            for pos_elem in wpts_elem.findall('position'):
                x = float(pos_elem.get('x', 0.0))
                y = float(pos_elem.get('y', 0.0))
                z = float(pos_elem.get('z', 0.0))
                waypoints_data.append({'x': x,
                                       'y': y,
                                       'z': z})
        keypoints = waypoints_data
        
        scenarios_data = []
        scs_elem = route_elem.find('scenarios')
        if scs_elem is not None:
            for sc_elem in scs_elem.findall('scenario'):
                sc_name = sc_elem.get('name', '')
                sc_type = sc_elem.get('type', '')

                tp_elem = sc_elem.find('trigger_point')  # XML 요소 찾기
                trigger_point = {}
                if tp_elem is not None:
                    trigger_point = {
                        'x': float(tp_elem.get('x', 0.0)),
                        'y': float(tp_elem.get('y', 0.0)),
                        'z': float(tp_elem.get('z', 0.0)),
                        'yaw': float(tp_elem.get('yaw', 0.0)),
                    }

                extra_args = {}
                for child in sc_elem:
                    if child.tag != 'trigger_point':  # trigger_point는 이미 처리했음
                        key = child.tag
                        val_str = child.get('value')
                        try:
                            val = float(val_str)
                        except (TypeError, ValueError):
                            val = val_str
                        extra_args[key] = val

                sc_data = {
                    'name': sc_name,
                    'type': sc_type,
                    'trigger_point': trigger_point,
                    'extra_args': extra_args
                }
                scenarios_data.append(sc_data)
        
        route_info = {
            'id': route_id,
            'town': town,
            'weathers': weathers_data,
            'waypoints': waypoints_data,
            'scenarios': scenarios_data,
            'keypoints': keypoints
        }
        all_routes.append(route_info)
        
    return all_routes

def gaussian_nll(action, mu, log_std):
    """Compute negative log-likelihood of a Gaussian with diagonal covariance.
    action, mu, log_std: shape [B, A]
    Returns per-sample scalar NLL: shape [B]
    """
    # clamp for numerical stability
    log_std = torch.clamp(log_std, min=-20.0, max=2.0)
    var = torch.exp(2.0 * log_std)  # since log_std = log(sigma)
    nll = 0.5 * ((action - mu)**2 / var + 2.0 * log_std + np.log(2.0 * np.pi))
    return nll.sum(dim=-1)  # sum over action dims

def kl_diag_gaussians_logvar(mu_q, logvar_q, mu_p, logvar_p):
    """KL( N(mu_q, diag(exp(logvar_q))) || N(mu_p, diag(exp(logvar_p))) )
    Returns per-sample scalar KL: shape [B]
    """
    # clamp logvars for stability
    logvar_q = torch.clamp(logvar_q, min=-10.0, max=10.0)
    logvar_p = torch.clamp(logvar_p, min=-10.0, max=10.0)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    # KL for diagonal Gaussians
    kl = 0.5 * ( (logvar_p - logvar_q) + (var_q + (mu_q - mu_p)**2) / var_p - 1.0 )
    return kl.sum(dim=-1)  # sum over latent dims

def ensure_chw(img):
    if img.dtype == torch.uint8:
        if img.dim() == 4 and img.shape[-1] in (1, 3, 4):  # HWC -> CHW
            img = img.permute(0, 3, 1, 2).contiguous()
            img = img.to(torch.float32).div_(255.0)
        elif img.dim() == 5 and img.shape[-1] in (1, 3, 4):  # BHWC -> BCHW
            img = img.permute(0, 1, 4, 2, 3).contiguous()
            img = img.to(torch.float32).div_(255.0)
        return img
    
    if img.dim() == 4 and img.shape[1] not in (1, 3, 4) and img.shape[-1] in (1, 3, 4):
        img = img.permute(0, 3, 1, 2).contiguous()
    elif img.dim() == 5 and img.shape[2] not in (1, 3, 4) and img.shape[-1] in (1, 3, 4):
        img = img.permute(0, 1, 4, 2, 3).contiguous()
    return img
