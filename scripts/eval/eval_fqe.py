from omegaconf import OmegaConf
import click
import datetime as dt
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataset.dataset import *
from model.value_function import *
from model.policy import *
from model.hilp import HilbertRepresentation
from model.opal import Encoder as OpalEncoder
from model.hso_vp import Encoder as HsoVpEncoder


def evaluate_policy(q_network, policy, dataloader, cfg):
    q_network.eval()
    total_q = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            state, _, _, _, _ = batch
            state = state.to(cfg.device)
            action = policy(state)
            q_values = q_network(state, action)
            total_q += q_values.sum().item()
            count += state.size(0)
    avg_q = total_q / count
    return avg_q


@click.command()
@click.option("--config", type=str, default="eval_fqe", help="config file name")
def main(config):
    cfg = OmegaConf.load(os.path.join(os.getcwd(), f"opal/config/{config}.yaml"))
    
    q_function = TwinQ(cfg.model)
    policy = GaussianPolicy(cfg.model)
    
    if cfg.algo == "hilp":
        hilbert_representation = HilbertRepresentation(cfg.model)
        ckpt = torch.load(os.path.join(os.getcwd(), f"outputs/hilbert_representation/{cfg.resume_ckpt_dir}/{cfg.hilp_dict_name}.pt"))
        hilbert_representation.load_state_dict(ckpt['hilbert_representation_state_dict'])
        hilbert_representation.eval()
        dataset = LatentDataset(cfg.seed, cfg.gamma, hilbert_representation, cfg.data)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    elif cfg.algo == "opal":
        encoder = OpalEncoder(cfg.model)
        ckpt = torch.load(os.path.join(os.getcwd(), f"outputs/opal/{cfg.resume_ckpt_dir}/{cfg.encoder_dict_name}.pt"))
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        encoder.eval()
        dataset = LatentDataset(cfg.seed, cfg.gamma, encoder, cfg.data)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    elif cfg.algo == "hsovp":
        encoder = HsoVpEncoder(cfg.model)
        ckpt = torch.load(os.path.join(os.getcwd(), f"outputs/opal/{cfg.resume_ckpt_dir}/{cfg.encoder_dict_name}.pt"))
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        encoder.eval()
        dataset = LatentDataset(cfg.seed, cfg.gamma, encoder, cfg.data)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    else:
        dataset = TrajectoryDataset(cfg.seed, cfg.data)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    avg_q_value = evaluate_policy(q_function, policy, dataloader, cfg)
    print(f"[Evaluation] Average Q Value: {avg_q_value:.4f}")
    
    result = {"average_q_value": avg_q_value}
    
    with open(os.path.join(os.getcwd(), f"outputs/eval_fqe/{cfg.algo}/{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(result, f, indent=4)
    
if __name__ == "__main__":
    main()
