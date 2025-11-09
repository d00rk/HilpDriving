import os, sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def setup_distributed():
    """Initialize torch.distributed if torchrun env vars are present."""
    ddp = False
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        ddp = True
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return ddp

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def unwrap(m):
    return m.module if isinstance(m, DDP) else m