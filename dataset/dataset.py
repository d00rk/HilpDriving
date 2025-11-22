import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import h5py
import threading
import glob
import random
import copy
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from torch.distributions import Normal
from sklearn.cluster import KMeans
import numpy as np

from utils.sampler import get_val_mask
from utils.seed_utils import seed_all
from utils.utils import ensure_chw


# =============================================================================
# TrajectoryDataset: {(s, a, s', r, terminal, timeout)_0^T}
# s: HWC uint8 image at timestep t
# a: [steer, throttle, brake] at timestep t
# s': HWC uint8 image at timestep t+1
# r: float reward at timestep t
# =============================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, seed, cfg):
        self.val_ratio = cfg.val_ratio
        self.seed = seed
        seed_all(self.seed)
        
        self.index_list = list()
        self.cache_episode_birdview = bool(getattr(cfg, "cache_episode_birdview", False))
        self.max_cached_episodes = int(getattr(cfg, "max_cached_episodes", 0))
        
        self._file_cache = None
        self._cache_lock = threading.Lock()
        self._episode_cache = None
        self._episode_buckets = {}
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/{town}/{t.lower()}/*.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
  
        for hdf5_path in hdf5_paths:
            file = h5py.File(hdf5_path, 'r')
            epi = list(file.keys())[0]
            data = file[epi]
            epi_length = data.attrs['episode_length']
            for i in range(epi_length):
                if i == (epi_length - 1):
                    next_key = i
                    terminal = True
                else:
                    next_key = i+1
                    terminal = False
                self.index_list.append((hdf5_path, epi, i, next_key, terminal))
            file.close()
        self._rebuild_episode_buckets()
    
    def _load_frame(self, data, idx):
        frame = torch.from_numpy(data['birdview'][idx])
        return frame
    
    def _get_episode_tensor(self, file_path, epi_key):
        if not (self.cache_episode_birdview and self.max_cached_episodes > 0):
            return None
        self._ensure_handles()
        if self._episode_cache is None:
            self._episode_cache = OrderedDict()
        cache_key = (file_path, epi_key)
        with self._cache_lock:
            cached = self._episode_cache.get(cache_key)
            if cached is not None:
                self._episode_cache.move_to_end(cache_key)
                return cached

        f = self._get_file(file_path)
        data = f[epi_key]
        frames = torch.from_numpy(data['birdview'][()])
        steer = torch.from_numpy(data['measurements']['steer'][()])
        throttle = torch.from_numpy(data['measurements']['throttle'][()])
        brake = torch.from_numpy(data['measurements']['brake'][()])
        rewards = torch.from_numpy(data['reward']['r_sum'][()])

        episode_tuple = (frames, steer, throttle, brake, rewards)
        with self._cache_lock:
            self._episode_cache[cache_key] = episode_tuple
            self._episode_cache.move_to_end(cache_key)
            while len(self._episode_cache) > self.max_cached_episodes:
                self._episode_cache.popitem(last=False)
        return episode_tuple
    
    def __getstate__(self):
        """Remove unpicklable members before pickling (for worker processes)."""
        state = self.__dict__.copy()
        state["_file_cache"] = None
        state["_cache_lock"] = None
        state["_episode_cache"] = None
        return state

    def __setstate__(self, state):
        """Recreate members after unpickling (inside each worker)."""
        self.__dict__.update(state)
        self._file_cache = None
        self._cache_lock = None
        self._episode_cache = None

    def _ensure_handles(self):
        """Lazily create per-process cache and lock."""
        if self._file_cache is None:
            self._file_cache = {}
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
        if self.cache_episode_birdview and self._episode_cache is None:
            self._episode_cache = OrderedDict()
    
    
    def _rebuild_episode_buckets(self):
        """Group sample indices by (file_path, episode) for sequential sampling."""
        buckets = {}
        for idx, (file_path, epi_key, *_rest) in enumerate(self.index_list):
            buckets.setdefault((file_path, epi_key), []).append(idx)
        self._episode_buckets = buckets
    
    def __len__(self):
        return len(self.index_list)

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
    
    def __getitem__(self, index):
        file_path, epi, current_idx, next_idx, terminal_flag = self.index_list[index]
        cached = self._get_episode_tensor(file_path, epi)
        f = self._get_file(file_path)
        data = f[epi]

        if cached is not None:
            episode_frames, episode_steer, episode_throttle, episode_brake, episode_rewards = cached
            current_obs = episode_frames[current_idx]
            next_obs = episode_frames[next_idx]
            steer = float(episode_steer[current_idx])
            throttle = float(episode_throttle[current_idx])
            brake = float(episode_brake[current_idx])
            reward_value = float(episode_rewards[current_idx])
        else:
            current_obs = self._load_frame(data, current_idx)
            next_obs = self._load_frame(data, next_idx)
            steer = float(data['measurements']['steer'][current_idx])
            throttle = float(data['measurements']['throttle'][current_idx])
            brake = float(data['measurements']['brake'][current_idx])
            reward_value = float(data['reward']['r_sum'][current_idx])

        action = torch.tensor([steer, throttle, brake], dtype=torch.float32)
        reward = torch.tensor([reward_value], dtype=torch.float32)
        terminal = torch.as_tensor([terminal_flag], dtype=torch.float32)

        return (current_obs, action, next_obs, reward, terminal, False)
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.index_list[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.index_list[i] for i in val_idxs]
        
        return train_dataset, val_dataset


# =============================================================================
# GoalDataset: {(s, s', g)_0^T}
# s: HWC uint8 image at timestep t
# s': HWC uint8 image at timestep t+1
# g: HWC uint8 image at timestpe t+H (H is random)
# =============================================================================
class GoalDataset(Dataset):
    def __init__(self, seed, cfg):
        seed_all(seed)
        self.seed = seed
        self.p = cfg.p
        self.p_goal = cfg.p_goal
        self.val_ratio = cfg.val_ratio
        self.index_list = list()
        self.cache_episode_birdview = bool(getattr(cfg, "cache_episode_birdview", False))
        self.max_cached_episodes = int(getattr(cfg, "max_cached_episodes", 0))
        
        self._file_cache = None
        self._cache_lock = threading.Lock()
        self._episode_cache = None
        self._episode_buckets = {}
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/{town}/{t.lower()}/*.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            file = h5py.File(hdf5_path, 'r')
            epi = list(file.keys())[0]
            data = file[epi]
            epi_length = data.attrs['episode_length']
            for i in range(epi_length):
                current_step_key = i
                
                if i == (epi_length - 1):
                    next_step_key = i
                    goal_step_key = i
                else:
                    step = 1
                    max_future_step = (epi_length - 1) - i
                    next_step_key = min(i + 1, epi_length - 1)
                    while True:
                        if random.random() < self.p:
                            break
                        step += 1
                        if step >= max_future_step:
                            step = max_future_step
                            break
                    goal_step_key = min(i + step, epi_length- 1)
                is_goal_now = (current_step_key == goal_step_key)
                    
                self.index_list.append(
                    (hdf5_path, epi, current_step_key, next_step_key, goal_step_key, is_goal_now)
                )
            file.close()
        self._rebuild_episode_buckets()

    def _load_frame(self, data, idx):
        """Read a single birdview frame as HWC uint8 tensor."""
        frame = torch.from_numpy(data['birdview'][idx])
        return frame

    def _maybe_get_episode_tensor(self, file_path, epi_key):
        """Optionally cache the entire birdview tensor for the episode."""
        if not (self.cache_episode_birdview and self.max_cached_episodes > 0):
            return None
        self._ensure_handles()
        if self._episode_cache is None:
            self._episode_cache = OrderedDict()
        cache_key = (file_path, epi_key)
        with self._cache_lock:
            cached = self._episode_cache.get(cache_key)
            if cached is not None:
                self._episode_cache.move_to_end(cache_key)
                return cached
        f = self._get_file(file_path)
        data = f[epi_key]
        frames = torch.from_numpy(data['birdview'][()])
        steer = torch.from_numpy(data['measurements']['steer'][()])
        throttle = torch.from_numpy(data['measurements']['throttle'][()])
        brake = torch.from_numpy(data['measurements']['brake'][()])
        rewards = torch.from_numpy(data['reward']['r_sum'][()])
        episode_tuple = (frames, steer, throttle, brake, rewards)
        with self._cache_lock:
            self._episode_cache[cache_key] = episode_tuple
            self._episode_cache.move_to_end(cache_key)
            while len(self._episode_cache) > self.max_cached_episodes:
                self._episode_cache.popitem(last=False)
        return episode_tuple

    def _rebuild_episode_buckets(self):
        """Group sample indices by (file_path, episode) for sequential sampling."""
        buckets = {}
        for idx, (file_path, epi_key, *_rest) in enumerate(self.index_list):
            buckets.setdefault((file_path, epi_key), []).append(idx)
        self._episode_buckets = buckets
                    
    def __len__(self):
        return len(self.index_list)
    
    def __getstate__(self):
        """Remove unpicklable members before pickling (for worker processes)."""
        state = self.__dict__.copy()
        state["_file_cache"] = None
        state["_cache_lock"] = None
        state["_episode_cache"] = None
        return state

    def __setstate__(self, state):
        """Recreate members after unpickling (inside each worker)."""
        self.__dict__.update(state)
        self._file_cache = None
        self._cache_lock = None
        self._episode_cache = None

    def _ensure_handles(self):
        """Lazily create per-process cache and lock."""
        if self._file_cache is None:
            self._file_cache = {}
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
        if self.cache_episode_birdview and self._episode_cache is None:
            self._episode_cache = OrderedDict()
    
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
    
    def __getitem__(self, index):
        file_path, epi_key, current_key, next_key, goal_key, is_goal_now = self.index_list[index]
        episode_tensor = self._maybe_get_episode_tensor(file_path, epi_key)

        if episode_tensor is not None:
            current_obs = episode_tensor[current_key]
            next_obs = episode_tensor[next_key]
            goal_obs = episode_tensor[goal_key]
        else:
            f = self._get_file(file_path)
            d = f[epi_key]
            
            current_obs = self._load_frame(d, current_key)
            next_obs = self._load_frame(d, next_key)
            goal_obs = self._load_frame(d, goal_key)
        
        is_goal_now = torch.tensor(is_goal_now, dtype=torch.float32)
        
        return (current_obs, next_obs, goal_obs, is_goal_now)
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.index_list[i] for i in train_idxs]
        train_dataset._rebuild_episode_buckets()
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.index_list[i] for i in val_idxs]
        val_dataset._rebuild_episode_buckets()
        
        return train_dataset, val_dataset


# =============================================================================
# LatentGoalDataset: {(s, z, s', g, R, terminal)} (for OPAL, HsO-VP high-level)
# s: HWC uint8 image at time t
# z: skill latent summarizing the sub-trajectory
# s': HWC uint8 image at time t+L (L is fixed)
# g: HWC uint8 image goal image
# R: discounted reward over the sub-trajectory
# =============================================================================
class LatentGoalDataset(Dataset):
    def __init__(self, seed, gamma, encoder, cfg):
        self.seed = seed
        seed_all(self.seed)
        
        self.val_ratio = cfg.val_ratio
        self.gamma = gamma
        self.encoder = encoder
        self.algo = cfg.algo
        
        self.cache_episode_birdview = bool(getattr(cfg, "cache_episode_birdview", False))
        self.max_cached_episodes = int(getattr(cfg, "max_cached_episodes", 0))
        self._file_cache = None
        self._cache_lock = threading.Lock()
        self._episode_cache = None
        self._episode_buckets = {}
        
        self.index_list = list()
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/{town}/{t.lower()}/*.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            file = h5py.File(hdf5_path, 'r')
            epi = list(file.keys())[0]
            data = file[epi]
            epi_length = data.attrs['episode_length']
            goal_idx = epi_length - 1
            for i in range(epi_length - cfg.trajectory_length + 1):
                next_idx = i + cfg.trajectory_length - 1
                terminal = (next_idx == (epi_length - 1))
                self.index_list.append((hdf5_path, epi, i, next_idx, goal_idx, terminal))
            file.close()
        self._rebuild_episode_buckets()
    
    def _load_frame(self, data, idx):
        """Read a single birdview frame as HWC uint8 tensor."""
        frame = torch.from_numpy(data['birdview'][idx])
        return frame

    def _maybe_get_episode_tensor(self, file_path, epi_key):
        """Optionally cache the entire birdview tensor for the episode."""
        if not (self.cache_episode_birdview and self.max_cached_episodes > 0):
            return None
        self._ensure_handles()
        if self._episode_cache is None:
            self._episode_cache = OrderedDict()
        cache_key = (file_path, epi_key)
        with self._cache_lock:
            cached = self._episode_cache.get(cache_key)
            if cached is not None:
                self._episode_cache.move_to_end(cache_key)
                return cached
        f = self._get_file(file_path)
        data = f[epi_key]
        frames = torch.from_numpy(data['birdview'][()])
        steer = torch.from_numpy(data['measurements']['steer'][()])
        throttle = torch.from_numpy(data['measurements']['throttle'][()])
        brake = torch.from_numpy(data['measurements']['brake'][()])
        rewards = torch.from_numpy(data['reward']['r_sum'][()])
        with self._cache_lock:
            self._episode_cache[cache_key] = (frames, steer, throttle, brake, rewards)
            self._episode_cache.move_to_end(cache_key)
            while len(self._episode_cache) > self.max_cached_episodes:
                self._episode_cache.popitem(last=False)
        return self._episode_cache[cache_key]

    def _rebuild_episode_buckets(self):
        """Group sample indices by (file_path, episode) for sequential sampling."""
        buckets = {}
        for idx, (file_path, epi_key, *_rest) in enumerate(self.index_list):
            buckets.setdefault((file_path, epi_key), []).append(idx)
        self._episode_buckets = buckets
    
    def __len__(self):
        return len(self.index_list)
    
    def __getstate__(self):
        """Remove unpicklable members before pickling (for worker processes)."""
        state = self.__dict__.copy()
        state["_file_cache"] = None
        state["_cache_lock"] = None
        state["_episode_cache"] = None
        return state

    def __setstate__(self, state):
        """Recreate members after unpickling (inside each worker)."""
        self.__dict__.update(state)
        self._file_cache = None
        self._cache_lock = None
        self._episode_cache = None

    def _ensure_handles(self):
        """Lazily create per-process cache and lock."""
        if self._file_cache is None:
            self._file_cache = {}
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
        if self.cache_episode_birdview and self._episode_cache is None:
            self._episode_cache = OrderedDict()
    
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
        
    def __getitem__(self, index):
        file_path, epi_key, start_idx, end_idx, goal_idx, terminal = self.index_list[index]
        cached = self._maybe_get_episode_tensor(file_path, epi_key)
        f = self._get_file(file_path)
        d = f[epi_key]

        if cached is not None:
            episode_tensor, episode_steer, episode_throttle, episode_brake, episode_rewards = cached
            current_obs = episode_tensor[start_idx]
            end_obs = episode_tensor[end_idx]
            goal_obs = episode_tensor[goal_idx]
        else:
            episode_rewards = None
            current_obs = self._load_frame(d, start_idx)
            end_obs = self._load_frame(d, end_idx)
            goal_obs = self._load_frame(d, goal_idx)

        if self.algo != "hilp":
            if cached is not None:
                obs_seq = episode_tensor[start_idx:end_idx+1].unsqueeze(0).contiguous()
                steer_seq = episode_steer[start_idx:end_idx+1]
                throttle_seq = episode_throttle[start_idx:end_idx+1]
                brake_seq = episode_brake[start_idx:end_idx+1]
                action_seq = torch.stack([steer_seq, throttle_seq, brake_seq], dim=-1).to(torch.float32)
                action_seq = action_seq.unsqueeze(0).contiguous()
            else:
                obss = []
                actions = []
                for j in range(start_idx, end_idx+1):
                    obs = self._load_frame(d, j)
                    obss.append(obs)

                    steer = float(d['measurements']['steer'][j])
                    throttle = float(d['measurements']['throttle'][j])
                    brake = float(d['measurements']['brake'][j])
                    action = torch.tensor([steer, throttle, brake], dtype=torch.float32)
                    actions.append(action)

                obs_seq = torch.stack(obss, dim=0).unsqueeze(0).contiguous()
                action_seq = torch.stack(actions, dim=0).unsqueeze(0).contiguous()

            with torch.no_grad():
                z = self.encoder(obs_seq, action_seq)
                if z.ndim == 2:
                    z = z[0]
        else:
            with torch.no_grad():
                # Normalize inputs for the pretrained encoder (expects float NCHW)
                device = next(self.encoder.parameters()).device
                current_input = ensure_chw(current_obs.unsqueeze(0)).to(device)
                end_input = ensure_chw(end_obs.unsqueeze(0)).to(device)

                z = self.encoder(current_input)
                z_goal = self.encoder(end_input)
                vec = z_goal - z
                norm = torch.norm(vec, dim=-1, keepdim=True)
                z = vec / (norm + 1e-6)
                z = z.squeeze(0)
        z = z.cpu()
        if cached is not None:
            reward_slice = episode_rewards[start_idx:end_idx+1].to(torch.float32)
        else:
            reward_slice = torch.as_tensor(d['reward']['r_sum'][start_idx:end_idx+1], dtype=torch.float32)
        R = 0.0
        for t, r_val in enumerate(reward_slice):
            R += (self.gamma ** t) * float(r_val)

        return (
            current_obs,
            z,
            end_obs,
            goal_obs,
            torch.tensor([R], dtype=torch.float32),
            torch.tensor([terminal], dtype=torch.float32),
            torch.tensor([False], dtype=torch.float32),
        )
         
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.index_list[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.index_list[i] for i in val_idxs]
        
        return train_dataset, val_dataset


# =============================================================================
# LatentDataset: {(s, z, s', g, R, terminal)} (for OPAL, HsO-VP high-level)
# s: CHW float [0, 1] at time t
# z: skill latent summarizing the sub-trajectory
# s': CHW float [0, 1] at time t+L (L is fixed)
# R: discounted reward over the sub-trajectory
# =============================================================================
class LatentDataset(Dataset):
    def __init__(self, seed, gamma, encoder, cfg):
        self.seed = seed
        seed_all(self.seed)
        
        self.val_ratio = cfg.val_ratio
        self.gamma = gamma
        self.encoder = encoder
        self.algo = cfg.algo
        
        self._file_cache = None
        self._cache_lock = threading.Lock()
        
        self.index_list = list()
        hdf5_paths = []
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/{town}/{t.lower()}/*.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            file = h5py.File(hdf5_path, 'r')
            epi = list(file.keys())[0]
            data = file[epi]
            epi_length = data.attrs['episode_length']
            for i in range(epi_length - cfg.trajectory_length + 1):
                next_idx = i + cfg.trajectory_length - 1
                terminal = (next_idx == epi_length - 1)
                self.index_list.append((hdf5_path, epi, i, next_idx, terminal))
            file.close()
        
    def __len__(self):
        return len(self.index_list)
    
    def _get_file(self, path):
        if self._file_cache is None:
            self._file_cache = {}
        
        with self._cache_lock:
            f = self._file_cache.get(path)
            if f is None:
                f = h5py.File(
                    path, 'r', libver='latest', swmr=True, rdcc_nbytes=int(256*1024*1024), rdcc_nslots=1_000_003
                )
                self._file_cache[path] = f
        return f
        
    def __getitem__(self, index):
        file_path, epi_key, start_idx, end_idx, terminal = self.index_list[index]
        f = self._get_file(file_path)
        
        data = f[epi_key]
        
        current_obs = torch.from_numpy(data['birdview'][start_idx].astype('float32')) / 255.0
        goal_obs = torch.from_numpy(data['birdview'][end_idx].astype('float32')) / 255.0
        current_obs = current_obs.permute(2, 0, 1).contiguous()
        goal_obs = goal_obs.permute(2, 0, 1).contiguous()
        
        R = 0.0
        if self.algo == "hilp":
            with torch.no_grad():
                z = self.encoder(current_obs.unsqueeze(0))
                z_goal = self.encoder(goal_obs.unsqueeze(0))
                vec = z_goal - z
                norm = torch.norm(vec)
                z = vec / (norm + 1e-6)
                z = z.squeeze(0)
        else:
            s_seq, a_seq, r_seq = [], [], []
            for j in range(start_idx, end_idx):
                s = torch.from_numpy(data['birdview'][j].astype('float32')) / 255.0
                s = s.permute(2, 0, 1).contiguous()
                s_seq.append(s)
                
                steer = float(data['measurements']['steer'][j])
                throttle = float(data['measurements']['throttle'][j])
                brake = float(data['measurements']['brake'][j])
                a_seq.append(torch.tensor([steer, throttle, brake], dtype=torch.float32))
                
                r_j = float(data['reward']['r_sum'][j])
                r_seq.append(r_j)

            s_seq = torch.stack(s_seq, dim=0).unsqueeze(0).contiguous()
            a_seq = torch.stack(a_seq, dim=0).unsqueeze(0).contiguous()
            
            with torch.no_grad():
                z = self.encoder(s_seq, a_seq)
                if z.ndim == 2:
                    z = z[0]
            
            for t, r in enumerate(r_seq):
                R += (self.gamma ** t) * r

        return (current_obs, z, goal_obs, torch.tensor([R], dtype=torch.float32), torch.tensor([terminal], dtype=torch.float32), torch.tensor([False], dtype=torch.float32))
         
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.index_list[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.index_list[i] for i in val_idxs]
        
        return train_dataset, val_dataset


# =============================================================================
# SubTrajDataset: {(s, a, s', r, terminal)}_0^L
# s: HWC uint8 image at time t
# a: [steer, throttle, brake] float at time t
# s': HWC uint8 at time t+1
# r: float reward at time t
# terminal: terminal flag at time t
# =============================================================================
class SubTrajDataset(Dataset):
    def __init__(self, seed, cfg):
        self.seed = seed
        seed_all(self.seed)
        
        self.val_ratio = cfg.val_ratio
        self.length = cfg.length
        self.cache_episode_birdview = bool(getattr(cfg, "cache_episode_birdview", False))
        self.max_cached_episodes = int(getattr(cfg, "max_cached_episodes", 0))
        
        self.index_list = list()
        self._file_cache = None
        self._cache_lock = threading.Lock()
        self._episode_cache = None
        self._episode_buckets = {}
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/{town}/{t.lower()}/*.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)

        for hdf5_path in hdf5_paths:
            file = h5py.File(hdf5_path, 'r')
            epi = list(file.keys())[0]
            data = file[epi]
            epi_length = data.attrs['episode_length']
            last_start = epi_length - (self.length + 1)
            if last_start < 0:
                continue
            for i in range(last_start+1):
                self.index_list.append((hdf5_path, epi, i))
            file.close()
        self._rebuild_episode_buckets()
    def _load_frame(self, data, idx):
        """Read a single birdview frame as HWC uint8 tensor."""
        frame = torch.from_numpy(data['birdview'][idx])
        return frame

    def _maybe_get_episode_tensor(self, file_path, epi_key):
        """Optionally cache the entire birdview tensor for the episode."""
        if not (self.cache_episode_birdview and self.max_cached_episodes > 0):
            return None
        self._ensure_handles()
        if self._episode_cache is None:
            self._episode_cache = OrderedDict()
        cache_key = (file_path, epi_key)
        with self._cache_lock:
            cached = self._episode_cache.get(cache_key)
            if cached is not None:
                self._episode_cache.move_to_end(cache_key)
                return cached
        f = self._get_file(file_path)
        data = f[epi_key]
        frames = torch.from_numpy(data['birdview'][()])
        steer = torch.from_numpy(data['measurements']['steer'][()])
        throttle = torch.from_numpy(data['measurements']['throttle'][()])
        brake = torch.from_numpy(data['measurements']['brake'][()])
        rewards = torch.from_numpy(data['reward']['r_sum'][()])
        episode_tuple = (frames, steer, throttle, brake, rewards)
        with self._cache_lock:
            self._episode_cache[cache_key] = episode_tuple
            self._episode_cache.move_to_end(cache_key)
            while len(self._episode_cache) > self.max_cached_episodes:
                self._episode_cache.popitem(last=False)
        return episode_tuple

    def _rebuild_episode_buckets(self):
        """Group sample indices by (file_path, episode) for sequential sampling."""
        buckets = {}
        for idx, (file_path, epi_key, *_rest) in enumerate(self.index_list):
            buckets.setdefault((file_path, epi_key), []).append(idx)
        self._episode_buckets = buckets
    
    def __len__(self):
        return len(self.index_list)
    
    def _ensure_handles(self):
        """Lazily create per-process cache and lock."""
        if self._file_cache is None:
            self._file_cache = {}
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
        if self.cache_episode_birdview and self._episode_cache is None:
            self._episode_cache = OrderedDict()
    
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
    
    def __getitem__(self, index):
        file_path, epi_key, start_idx = self.index_list[index]
        L = self.length
        cached = self._maybe_get_episode_tensor(file_path, epi_key)
        f = self._get_file(file_path)
        data = f[epi_key]

        if cached is not None:
            episode_tensor, episode_steer, episode_throttle, episode_brake, episode_rewards = cached
            obs = episode_tensor[start_idx:start_idx+L+1]
            steer = episode_steer[start_idx:start_idx+L]
            throttle = episode_throttle[start_idx:start_idx+L]
            brake = episode_brake[start_idx:start_idx+L]
            rewards = episode_rewards[start_idx:start_idx+L].to(torch.float32)
        else:
            obs = torch.from_numpy(data['birdview'][start_idx:start_idx+L+1])
            steer = torch.from_numpy(data['measurements']['steer'][start_idx:start_idx+L]).to(torch.float32)
            throttle = torch.from_numpy(data['measurements']['throttle'][start_idx:start_idx+L]).to(torch.float32)
            brake = torch.from_numpy(data['measurements']['brake'][start_idx:start_idx+L]).to(torch.float32)
            rewards = torch.from_numpy(data['reward']['r_sum'][start_idx:start_idx+L].astype('float32'))

        actions = torch.stack([steer, throttle, brake], dim=-1)
        states = obs[:-1]
        next_states = obs[1:]

        epi_last = int(data.attrs['episode_length']) - 1
        t_idx = np.arange(start_idx, start_idx+L, dtype=np.int64)
        terminals = torch.from_numpy((t_idx == epi_last).astype(np.float32))
        timeouts = torch.zeros(L, dtype=torch.float32)
        
        return (states, actions, next_states, rewards, terminals, timeouts)
    
    def __getstate__(self):
        """Remove unpicklable members before pickling (for worker processes)."""
        state = self.__dict__.copy()
        state["_file_cache"] = None
        state["_cache_lock"] = None
        state["_episode_cache"] = None
        return state

    def __setstate__(self, state):
        """Recreate members after unpickling (inside each worker)."""
        self.__dict__.update(state)
        self._file_cache = None
        self._cache_lock = None
        self._episode_cache = None
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.index_list[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.index_list[i] for i in val_idxs]
        
        return train_dataset, val_dataset


# =============================================================================
# LowLevelDataset: {(s, a, z)} (for OPAL, HsO-VP Decoder)
# s: HWC uint8 image at time (t, t+L)
# a: [steer, throttle, brake] float at time (t, t+L)
# z: latent encoded from freezed encoder
# =============================================================================
class LowLevelDataset(SubTrajDataset):
    def __init__(self, seed, cfg, encoder, sample_z):
        super().__init__(seed=seed, cfg=cfg)
        self.encoder = encoder
        self.sample_z = sample_z
    
    def __getitem__(self, index):
        s, a, s_n, r, terminal, timeouts = super().__getitem__(index)

        device = next(self.encoder.parameters()).device
        states = s.unsqueeze(0)
        acts = a.unsqueeze(0)
        
        with torch.no_grad():
            mu, logstd = self.encoder(states, acts)
            if self.sample_z:
                std = torch.exp(logstd)
                z = Normal(mu, std).rsample()
            else:
                z = mu
        z = z.squeeze(0).cpu()
        
        return (s, a, z)



# =============================================================================
# FilteredDataset (for HsO-VP): labeling sub-trajectories using K-means clustering
# =============================================================================
class FilteredDataset(Dataset):
    def __init__(self, seed, cfg):
        self.seed = seed
        seed_all(self.seed)
        self.cfg = cfg
        
        self.length = cfg.length
        self.num_cluster = cfg.discrete_option if hasattr(cfg, "discrete_option") else 6
        self.keep_ratio = getattr(cfg, "keep_ratio", 0.5)
        self.tau_k = getattr(cfg, "tau_k", 0.0)
        
        self._file_cache = None
        self._cache_lock = threading.Lock()
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/{town}/{t.lower()}/*.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        self.index_list = list()
        action_feature_list = list()
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as h5_file:
                epi = list(h5_file.keys())[0]
                data = h5_file[epi]
                epi_length = data.attrs['episode_length']
                last_sequence_start = epi_length - (self.length + 1)
                if last_sequence_start < 0:
                    continue
                for i in range(last_sequence_start + 1):
                    end_idx = i + self.length
                    terminal = (end_idx == (epi_length - 1))
                    self.index_list.append((hdf5_path, epi, i, end_idx, terminal))

                    actions = []
                    for j in range(i, end_idx):
                        steer = data['measurements']['steer'][j]
                        throttle = data['measurements']['throttle'][j]
                        brake = data['measurements']['brake'][j]
                        action = np.array([steer, throttle, brake], dtype=np.float32)
                        actions.append(action)
                    concat_action = np.concatenate(actions, axis=0)
                    action_feature_list.append(concat_action)
        self.action_features = np.asarray(action_feature_list)
        
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=self.seed)
        cluster_labels = kmeans.fit_predict(self.action_features)
        cluster_centers = kmeans.cluster_centers_
        
        clusters = {k: np.where(cluster_labels == k)[0] for k in range(self.num_cluster)}
        
        target_total = int(len(self.action_features) * self.keep_ratio)
        target_per_cluster = max(1, target_total // self.num_cluster)
        print(f"[FilteredDataset] k={self.num_cluster}, keep_ratio={self.keep_ratio}, \ntarget_total={target_total}, per_cluster={target_per_cluster}")

        selected_indices = []
        for k, idxs in clusters.items():
            feats = self.action_features[idxs]
            center = cluster_centers[k]
            
            dists = np.linalg.norm(feats - center, axis=1)
            
            if self.tau_k > 0.0:
                cluster_tau = self.tau_k * float(np.mean(dists))
            else:
                cluster_tau = 0.0  # no diversity constraint if tau_k <= 0
            
            order = np.argsort(dists)
            
            chosen_local  = []
            for ii in order:
                f = feats[ii]
                if len(chosen_local ) == 0:
                    chosen_local .append(ii)
                else:
                    if cluster_tau > 0.0:
                        # Enforce minimum pairwise distance within the cluster
                        f = feats[ii]
                        ok = True
                        for jj in chosen_local:
                            if np.linalg.norm(f - feats[jj]) <= cluster_tau:
                                ok = False
                                break
                        if not ok:
                            continue
                    chosen_local.append(ii)
                
                if len(chosen_local) >= target_per_cluster:
                    break
            
            if len(chosen_local) < target_per_cluster:
                for ii in order:
                    if ii not in chosen_local:
                        chosen_local.append(ii)
                    if len(chosen_local) >= target_per_cluster:
                        break
            chosen_global = idxs[np.asarray(chosen_local, dtype=int)]
            selected_indices.extend(chosen_global.tolist())
        
        self.filtered_index_list = [self.index_list[i] for i in selected_indices]
        self.filtered_cluster_labels = cluster_labels[selected_indices].tolist()

        print(f"[FilteredDataset] filtered_index_list length: {len(self.filtered_index_list)}")
    
    def _get_file(self, path):
        if self._file_cache is None:
            self._file_cache = {}
        
        with self._cache_lock:
            f = self._file_cache.get(path)
            if f is None:
                f = h5py.File(
                    path, 'r', libver='latest', swmr=True, rdcc_nbytes=int(256*1024*1024), rdcc_nslots=1_000_003
                )
                self._file_cache[path] = f
        return f
    
    def __len__(self):
        return len(self.filtered_index_list)
    
    def __getitem__(self, index):
        file_path, epi, start_idx, end_idx, term = self.filtered_index_list[index]
        cluster_label = int(self.filtered_cluster_labels[index])
        
        f = self._get_file(file_path)
        data = f[epi]
        
        s_seq, a_seq, sn_seq, r_seq, term_seq, to_seq = [], [], [], [], [], []
        for j in range(start_idx, end_idx):
            obs = torch.from_numpy(data['birdview'][j].astype('float32')) / 255.0
            next_obs = torch.from_numpy(data['birdview'][j+1].astype('float32')) / 255.0
            
            obs = obs.permute(2, 0, 1).contiguous()
            next_obs = next_obs.permute(2, 0, 1).contiguous()
            
            steer = float(data['measurements']['steer'][j])
            throttle = float(data['measurements']['throttle'][j])
            brake = float(data['measurements']['brake'][j])
            action = torch.tensor([steer, throttle, brake], dtype=torch.float32)
            
            reward = float(data['reward']['r_sum'][j])
            reward = torch.tensor([reward], dtype=torch.float32)

            terminal = float((j+1)==end_idx and term)
            timeout = False
            
            s_seq.append(obs)
            a_seq.append(action)
            sn_seq.append(next_obs)
            r_seq.append(reward)
            term_seq.append(torch.tensor([terminal], dtype=torch.float32))
            to_seq.append(torch.tensor([timeout], dtype=torch.float32))
        
        s = torch.stack(s_seq, dim=0)
        a = torch.stack(a_seq, dim=0)
        s_n = torch.stack(sn_seq, dim=0)
        r = torch.stack(r_seq, dim=0)
        term = torch.stack(term_seq, dim=0)
        to = torch.stack(to_seq, dim=0)

        return (s, a, s_n, r, term, to, cluster_label)
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.cfg.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.filtered_index_list = [self.filtered_index_list[i] for i in train_idxs]
        train_dataset.filtered_cluster_labels = [self.filtered_cluster_labels[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.filtered_index_list = [self.filtered_index_list[i] for i in val_idxs]
        val_dataset.filtered_cluster_labels = [self.filtered_cluster_labels[i] for i in val_idxs]
        
        return train_dataset, val_dataset
