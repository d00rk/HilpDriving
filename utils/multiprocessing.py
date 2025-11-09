import torch
from torch.utils.data import get_worker_info

def _worker_init_fn(_):
    wi = get_worker_info()
    ds = wi.dataset
    if hasattr(ds, "_file_cache"): ds._file_cache = None
    if hasattr(ds, "_cache_lock"): ds._cache_lock = None