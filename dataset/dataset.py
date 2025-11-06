import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import h5py
import threading
import glob
import random
import copy
import torch
from torch.utils.data import Dataset
from torch.distributions import Normal
from sklearn.cluster import KMeans
import numpy as np

from utils.sampler import get_val_mask
from utils.seed_utils import seed_all


# =============================================================================
# TrajectoryDataset: {(s, a, s', r, terminal, timeout)_0^T}
# s: CHW float [0, 1] at timestep t
# a: [steer, throttle, brake] at timestep t
# s': CHW float [0, 1] at timestep t+1
# r: float reward at timestep t
# =============================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, seed, cfg):
        self.val_ratio = cfg.val_ratio
        self.seed = seed
        seed_all(self.seed)
        
        self.index_list = list()
        self._file_cache = None
        self._cache_lock = threading.Lock()
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/data/{town.lower()}_{t.lower()}.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
  
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                epi_keys = sorted([k for k in f.keys() if k.startswith("episode_")], 
                                   key=lambda x: int(x.split("_")[1]))
                for epi in epi_keys:
                    data = f[epi]
                    epi_length = data.attrs['episode_length']
                    for i in range(epi_length):
                        if i == (epi_length - 1):
                            next_key = i
                            terminal = True
                        else:
                            next_key = i+1
                            terminal = False
                        self.index_list.append((hdf5_path, epi, i, next_key, terminal)) 
    
    def __len__(self):
        return len(self.index_list)

    def _get_file(self, path):
        if self._file_cache is None:
            self._file_cache = {}
        
        with self._cache_lock:
            f = self._file_cache.get(path)
            if f is None:
                f = h5py.File(
                    path, 'r', libver='latest', swmr=False, rdcc_nbytes=int(256*1024*1024), rdcc_nslots=1_000_003
                )
                self._file_cache[path] = f
        return f
    
    def __getitem__(self, index):
        file_path, epi, current_idx, next_idx, terminal = self.index_list[index]
        
        f = self._get_file(file_path)
        data = f[epi]
        
        current_obs = torch.from_numpy(data['birdview'][current_idx].astype('float32')) / 255.0
        next_obs = torch.from_numpy(data['birdview'][next_idx].astype('float32')) / 255.0
        current_obs = current_obs.permute(2, 0, 1).contiguous()
        next_obs = next_obs.permute(2, 0, 1).contiguous()
        
        steer = float(data['measurements']['steer'][current_idx])
        throttle = float(data['measurements']['throttle'][current_idx])
        brake = float(data['measurements']['brake'][current_idx])
        action = torch.tensor([steer, throttle, brake], dtype=torch.float32)
        
        reward = float(data['reward']['r_sum'][current_idx])
        reward = torch.tensor([reward], dtype=torch.float32)
        
        terminal = torch.tensor([terminal], dtype=torch.float32)
        
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
# s: CHW float [0, 1] at timestep t
# s': CHW float [0, 1] at timestep t+1
# g: CHW float [0, 1] at timestpe t+H (H is random)
# =============================================================================
class GoalDataset(Dataset):
    def __init__(self, seed, cfg):
        seed_all(seed)
        self.seed = seed
        self.p = cfg.p
        self.val_ratio = cfg.val_ratio
        self.index_list = list()
        
        self._file_cache = None
        self._cache_lock = threading.Lock()
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/data/{town.lower()}_{t.lower()}.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                epi_keys = sorted([k for k in f.keys() if k.startswith("episode_")], 
                                   key=lambda x: int(x.split("_")[1]))
                for epi in epi_keys:
                    data = f[epi]
                    epi_length = data.attrs['episode_length']
                    for i in range(epi_length-1):
                        current_step_key = i
                        next_step_key = i+1
                        
                        step = 1
                        max_future_step = (epi_length - 1) - i
                        
                        while True:
                            if random.random() < self.p:
                                break
                            step += 1
                            if step >= max_future_step:
                                step = max_future_step
                                break
                        
                        goal_step_key = i+step
                        assert goal_step_key != current_step_key, "g == s should be avoided"
                        
                        is_goal_now = (current_step_key == goal_step_key)
                        
                        self.index_list.append(
                            (hdf5_path, epi, current_step_key, next_step_key, goal_step_key, is_goal_now)
                        )
                    
    def __len__(self):
        return len(self.index_list)
    
    def _get_file(self, path):
        if self._file_cache is None:
            self._file_cache = {}
        
        with self._cache_lock:
            f = self._file_cache.get(path)
            if f is None:
                f = h5py.File(
                    path, 'r', libver='latest', swmr=False, rdcc_nbytes=int(256*1024*1024), rdcc_nslots=1_000_003
                )
                self._file_cache[path] = f
        return f
    
    def __getitem__(self, index):
        file_path, epi_key, current_key, next_key, goal_key, is_goal_now = self.index_list[index]
        f = self._get_file(file_path)

        d = f[epi_key]
        current_obs =  torch.from_numpy(d['birdview'][current_key].astype('float32')) / 255.0
        next_obs = torch.from_numpy(d['birdview'][next_key].astype('float32')) / 255.0
        goal_obs = torch.from_numpy(d['birdview'][goal_key].astype('float32')) / 255.0
            
        current_obs = current_obs.permute(2, 0, 1).contiguous() 
        next_obs = next_obs.permute(2, 0, 1).contiguous()
        goal_obs = goal_obs.permute(2, 0, 1).contiguous()
        
        return (current_obs, next_obs, goal_obs, is_goal_now)
    
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
# LatentDataset: {(s, z, s', R, terminal)} (for OPAL, HsO-VP high-level)
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
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/data/{town.lower()}_{t.lower()}.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                epi_keys = sorted([k for k in f.keys() if k.startswith("episode_")], key=lambda x: int(x.split("_")[1]))
                
                for epi in epi_keys:
                    data = f[epi]
                    epi_length = data.attrs['episode_length']
                    for i in range(epi_length - cfg.trajectory_length + 1):
                        next_idx = i + cfg.trajectory_length - 1
                        terminal = (next_idx == epi_length - 1)
                        self.index_list.append((hdf5_path, epi, i, next_idx, terminal))
        
    def __len__(self):
        return len(self.index_list)
    
    def _get_file(self, path):
        if self._file_cache is None:
            self._file_cache = {}
        
        with self._cache_lock:
            f = self._file_cache.get(path)
            if f in None:
                f = h5py.File(
                    path, 'r', libver='latest', swmr=False, rdcc_nbytes=int(256*1024*1024), rdcc_nslots=1_000_003
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
            
            R = 0.0
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
# SubTrajDataset: {(s, a, s', r, terminal)}_0^L (for OPAL, HsO-VP VAE)
# s: CHW float [0, 1] at time t
# a: [steer, throttle, brake] float at time t
# s': CHW float [0, 1] at time t+1
# r: float reward at time t
# terminal: terminal flag at time t
# =============================================================================
class SubTrajDataset(Dataset):
    def __init__(self, seed, cfg):
        self.seed = seed
        seed_all(self.seed)
        
        self.val_ratio = cfg.val_ratio
        self.length = cfg.length
        
        self.index_list = list()
        self._file_cache = None
        self._cache_lock = threading.Lock()
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/data/{town.lower()}_{t.lower()}.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)

        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                epi_keys = sorted([k for k in f.keys() if k.startswith("episode_")],
                                   key=lambda x: int(x.split("_")[1]))
                for epi in epi_keys:
                    data = f[epi]
                    epi_length = data.attrs['episode_length']
                    last_start = epi_length - (self.length + 1)
                    if last_start < 0:
                        continue
                    for i in range(last_start+1):
                        self.index_list.append((hdf5_path, epi, i))
        
    def __len__(self):
        return len(self.index_list)
    
    def _get_file(self, path):
        if self._file_cache is None:
            self._file_cache = {}
        
        with self._cache_lock:
            f = self._file_cache.get(path)
            if f in None:
                f = h5py.File(
                    path, 'r', libver='latest', swmr=False, rdcc_nbytes=int(256*1024*1024), rdcc_nslots=1_000_003
                )
                self._file_cache[path] = f
        return f
    
    def __getitem__(self, index):
        file_path, epi_idx, start_idx = self.index_list[index]
        L = self.length

        f = self._get_file(file_path)
        data = f[epi_idx]

        states, actions, next_states = [], [], []
        rewards, terminals, timeouts = [], [], []
        for t in range(start_idx, start_idx+L):
            t_next = t + 1
            
            s = torch.from_numpy(data['birdview'][t].astype('float32')) / 255.0
            s_next = torch.from_numpy(data['birdview'][t_next].astype('float32')) / 255.0
            s = s.permute(2, 0, 1).contiguous()
            s_next = s_next.permute(2, 0, 1).contiguous()
            
            steer = float(data['measurements']['steer'][t])
            throttle = float(data['measurements']['throttle'][t])
            brake = float(data['measurements']['brake'][t])
            a = torch.tensor([steer, throttle, brake], dtype=torch.float32)
            
            reward = float(data['reward']['r_sum'][t])
            
            is_terminal = float(t == (int(data.attrs['episode_length'])-1))
            
            states.append(s)
            actions.append(a)
            next_states.append(s_next)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            terminals.append(torch.tensor(is_terminal, dtype=torch.float32))
            timeouts.append(torch.tensor(0.0, dtype=torch.float32))
                
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        next_states = torch.stack(next_states, dim=0)
        rewards = torch.stack(rewards, dim=0)
        terminals = torch.stack(terminals, dim=0)
        timeouts = torch.stack(timeouts, dim=0)
        
        return (states, actions, next_states, rewards, terminals, timeouts)
    
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
# s: CHW float [0, 1] at time (t, t+L)
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
        
        self._file_cache = None
        self._cache_lock = threading.Lock()
        
        hdf5_paths = list()
        for town in cfg.data_town:
            for t in cfg.type:
                hp = glob.glob(os.path.join(os.getcwd(), f'data/lmdrive/data/{town.lower()}_{t.lower()}.hdf5'))
                hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        self.index_list = list()
        action_feature_list = list()
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                episode_keys = sorted([k for k in f.keys() if k.startswith("episode_")],
                                   key=lambda x: int(x.split("_")[1]))
                for epi in episode_keys:
                    data = f[epi]
                    epi_length = data.attrs['episode_length']
                    last_sequence_start = epi_length - (self.length + 1)
                    for i in range(last_sequence_start + 1):
                        end_idx = i + self.length
                        terminal = (end_idx == (epi_length - 1))
                        self.index_list.append((hdf5_path, epi, i, end_idx, terminal))
                        
                        actions = list()
                        for j in range(i, end_idx):
                            steer = data['measurements']['steer'][j]
                            throttle = data['measurements']['throttle'][j]
                            brake = data['measurements']['brake'][j]
                            action = np.array([steer, throttle, brake], dtype=np.float32)
                            actions.append(action)
                        concat_action = np.concatenate(actions, axis=0)
                        action_feature_list.append(concat_action)
        self.action_features = np.asarray(action_feature_list)
        
        self.num_cluster = cfg.discrete_option if hasattr(cfg, "discrete_option") else 6
        self.keep_ratio = cfg.keep_ratio if hasattr(cfg, "keep_ratio") else 0.5
        
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=self.seed)
        cluster_labels = kmeans.fit_predict(self.action_features)
        cluster_centers = kmeans.cluster_centers_
        
        clusters = {k: np.where(cluster_labels == k)[0] for k in range(self.num_cluster)}
        
        target_total = int(len(self.action_features) * cfg.keep_ratio)
        target_per_cluster = max(1, target_total // self.num_cluster)
        print(f"[FilteredDataset] k={self.num_cluster}, keep_ratio={self.keep_ratio}, \ntarget_total={target_total}, per_cluster={target_per_cluster}")

        selected_indices = []
        for k, idxs in clusters.items():
            feats = self.action_features[idxs]
            center = cluster_centers[k]
            
            dists = np.linalg.norm(feats - center, axis=1)
            order = np.argsort(dists)[::-1]
            
            chosen = []
            for ii in order:
                f = feats[ii]
                if len(chosen) == 0:
                    chosen.append(ii)
                else:
                    ok = True
                    for jj in chosen:
                        if np.linalg.norm(f - feats[jj]) <= self.tau_k:
                            ok = False
                            break
                    if ok:
                        chosen.append(ii)
                if len(chosen) >= target_per_cluster:
                    break
            
            if len(chosen) < target_per_cluster:
                for ii in order:
                    if ii not in chosen:
                        chosen.append(ii)
                    if len(chosen) >= target_per_cluster:
                        break
            chosen = idxs[np.asarray(chosen, dtype=int)]
            selected_indices.extend(chosen.tolist())
        
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
                    path, 'r', libver='latest', swmr=False, rdcc_nbytes=int(256*1024*1024), rdcc_nslots=1_000_003
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