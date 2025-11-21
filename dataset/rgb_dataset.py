import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import h5py
import threading
import glob
import random
import copy
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.sampler import get_val_mask
from utils.seed_utils import seed_all


# =============================================================================
# GoalRGBDataset: (current, next, goal, bev, is_goal_now)
#   - current [Dict[torch.tensor]]: THWC uint8 rgb images
#   - next [Dict[torch.tensor]]: THWC uint8 rgb images
#   - goal [Dict[torch.tensor]]: THWC uint8 rgb images
#   - bev [torch.tensor]: HWC uint8 bev image at current timestep t
# =============================================================================
class GoalRGBDataset(Dataset):
    def __init__(self, seed, cfg):
        seed_all(seed)
        self.seed = seed
        self.p = cfg.p
        self.val_ratio = cfg.val_ratio
        
        self.history_length = cfg.history_length
        self.img_size = cfg.img_size
        
        self.index_list = list()
        
        self._file_cache = None
        self._cache_lock = threading.Lock()
        self._episode_cache = None
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/data/{town}/{t.lower()}/*.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            file = h5py.File(hdf5_path, 'r')
            epi = list(file.keys())[0]
            data = file[epi]
            epi_length = data.attrs['episode_length']
            for i in range(epi_length-1):
                current_step_key = i
                
                if i == (epi_length - 1):
                    next_step_key = i
                    goal_step_key = i
                else:
                    step = 1
                    max_future_step = (epi_length - 1) - i
                    next_step_key = min(i+1, epi_length-1)
                    while True:
                        if random.random() < self.p:
                            break
                        step += 1
                        if step >= max_future_step:
                            step = max_future_step
                            break
                    goal_step_key = min(i+step, epi_length-1)
                is_goal_now = (current_step_key == goal_step_key)
                
                self.index_list.append(
                    (hdf5_path, epi, current_step_key, next_step_key, goal_step_key, is_goal_now)
                )
            file.close()
    
    def __len__(self):
        return len(self.index_list)
    
    def __getstate__(self):
        """Remove unpicklable members before pickling (for worker processes)."""
        state = self.__dict__.copy()
        state["_file_cache"] = None
        state["_cache_lock"] = None
        return state

    def __setstate__(self, state):
        """Recreate members after unpickling (inside each worker)."""
        self.__dict__.update(state)
        self._file_cache = None
        self._cache_lock = None
    
    def _ensure_handles(self):
        """Lazily create per-process cache and lock."""
        if self._file_cache is None:
            self._file_cache = {}
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
    
    def _get_file(self, path):
        self._ensure_handles()
        with self._cache_lock:
            f = self._file_cache.get(path)
            if f is None:
                f = h5py.File(
                    path, 'r', libver='latest', swmr=True, rdcc_nbytes=int(256*1024*1024), rdcc_nslots=1_000_003
                )
                self._file_cache[path] = f
        return f
    
    def _read_multiview_sequence(self, data_group, end_idx):
        start_idx = end_idx - self.history_length + 1
        idxs = np.arange(start_idx, end_idx+1)
        effective_idxs = np.maximum(idxs, 0)
        
        views = ['front_rgb', 'left_rgb', 'right_rgb']
        seq_images = {v: [] for v in views}
        
        if start_idx >= 0:
            for v in views:
                imgs = data_group[v][start_idx:end_idx+1]
                seq_images[v] = imgs
        else:
            for v in views:
                valid_len = end_idx+1
                pad_len = self.history_length - valid_len
                
                valid_imgs = data_group[v][0:valid_len]
                first_img = valid_imgs[0]
                
                padded_imgs = np.concatenate([
                    np.tile(first_img[None, ...], (pad_len, 1, 1, 1)),
                    valid_imgs
                ], axis=0)
                seq_images[v] = padded_imgs
        
        out_tensors = {}
        for v in views:
            img_np = seq_images[v]
            tensor = torch.from_numpy(img_np)
            out_tensors[v] = tensor
        
        return out_tensors
    
    def _read_bev(self, data_group, idx):
        bev = torch.from_numpy(data_group['birdview'][idx])
        return bev
        
    def __getitem__(self, index):
        file_path, epi_key, current_key, next_key, goal_key, is_goal_now = self.index_list[index]
        f = self._get_file(file_path)
        d = f[epi_key]
        
        current_images = self._read_multiview_sequence(d, current_key)
        next_images = self._read_multiview_sequence(d, next_key)
        goal_images = self._read_multiview_sequence(d, goal_key)
        
        bev = self._read_bev(d, current_key)
        
        is_goal_now = torch.tensor(is_goal_now, dtype=torch.float32)
        
        return (current_images, next_images, goal_images, bev, is_goal_now)
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.index_list[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.index_list[i] for i in val_idxs]
        
        return train_dataset, val_dataset


class SubTrajRGBDataset(Dataset):
    def __init__(self, seed, cfg):
        self.seed = seed
        seed_all(self.seed)
        
        self.val_ratio = cfg.val_ratio
        self.length = cfg.length    # sub-trajectory length
        self.history_length = cfg.history_length    # history length
        
        self._file_cache = None
        self._cache_lock = threading.Lock()
        
        self.index_list = list()
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/data/{town}/{t.lower()}/*.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                epi = list(f.keys())[0]
                data = f[epi]
                epi_length = data.attrs['episode_length']
                f.close()
            
            last_start = epi_length - (self.length + 1)
            if last_start < 0:
                continue
            
            for i in range(last_start+1):
                self.index_list.append((hdf5_path, epi, i))
    
    def _ensure_handles(self):
        if self._file_cache is None:
            self._file_cache = {}
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
    
    def _get_file(self, path):
        self._ensure_handles()
        with self._cache_lock:
            f = self._file_cache.get(path)
            if f is None:
                f = h5py.File(
                    path, 'r', libver='latest', swmr=True, rdcc_nbytes=int(512*1024*1024)
                )
                self._file_cache[path] = f
        return f
    
    def _read_multiview_sequence_traj(self, data_group, start_idx):
        L = self.length
        H = self.history_length
        
        read_start = start_idx - H + 1
        read_end = start_idx + L
        
        pad_len = 0
        if read_start < 0:
            pad_len = abs(read_start)
            read_start = 0
        
        views = ['front_rgb', 'left_rgb', 'right_rgb']
        out_tensors = {}
        for v in views:
            raw_frames = torch.from_numpy(data_group[v][read_start:read_end+1])
            
            if pad_len > 0:
                first_frame = raw_frames[0:1]
                padding = first_frame.repeat(pad_len, 1, 1, 1)
                frames = torch.cat([padding, frames], dim=0)
            
            sequences = frames.unfold(0, H, 1)
            out_tensors[v] = sequences
        
        return out_tensors
    
    def __getitem__(self, index):
        file_path, epi, start_idx = self.index_list[index]
        L = self.length
        
        f = self._get_file(file_path)
        data = f[epi]
        
        image_seqs = self._read_multiview_sequence_traj(data, start_idx)
        
        obs = {}
        next_obs = {}
        for k, v in image_seqs.items():
            obs[k] = v[0:L]
            next_obs[k] = v[1:L+1]
        
        steer = torch.from_numpy(data['measurements']['steer'][start_idx:start_idx+L]).float()
        throttle = torch.from_numpy(data['measurements']['throttle'][start_idx:start_idx+L]).float()
        brake = torch.from_numpy(data['measurements']['brake'][start_idx:start_idx+L]).float()
        actions = torch.stack([steer, throttle, brake], dim=-1)
        
        rewards = torch.from_numpy(data['reward']['r_sum'][start_idx:start_idx+L]).float()
        
        epi_last = int(data.attrs['episode_length']) - 1
        t_idx = np.arange(start_idx, start_idx+L, dtype=np.int64)
        terminals = torch.from_numpy((t_idx == epi_last).astype(np.float32))
        timeouts = torch.zeros(L, dtype=torch.float32)
        
        return (obs, actions, next_obs, rewards, terminals, timeouts)

    def __len__(self):
        return len(self.index_list)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_cache"] = None
        state["_cache_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_cache = None
        self._cache_lock = None
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.index_list[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.index_list[i] for i in val_idxs]
        
        return train_dataset, val_dataset