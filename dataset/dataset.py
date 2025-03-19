import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import h5py
import glob
import random
import copy
import torch
from torch.utils.data import Dataset
from torch.distributions import Normal
from sklearn.cluster import KMeans
import numpy as np

from utils.sampler import get_val_mask
from utils.seed import seed_all

# =============================================================================
# TrajectoryDataset: {(s, a, s', r, terminal, timeout)_0^T}
# =============================================================================
class TrajectoryDataset(Dataset):
    def __init__(self, seed, cfg):
        self.val_ratio = cfg.val_ratio
        self.seed = seed
        seed_all(self.seed)
        
        self.index_list = list()
        hdf5_paths = list()
        for t in cfg.data_town:
            hp = glob.glob(os.path.join(os.path.dirname(os.getcwd()), f'datasets/{cfg.data_algo}/{cfg.data_benchmark}/{t.lower()}_*.hdf5'))
            hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
  
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                step_keys = sorted([k for k in f.keys() if k.startswith("step_")], 
                                   key=lambda x: int(x.split("_")[1]))
                epi_length = len(step_keys)
                for i in range(epi_length):
                    current_key = step_keys[i]
                    if i == (epi_length - 1):
                        next_key = current_key
                        terminal = True
                    else:
                        next_key = step_keys[i+1]
                        terminal = False
                    self.index_list.append((hdf5_path, current_key, next_key, terminal)) 
    
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        file_path, current_key, next_key, terminal = self.index_list[index]
        with h5py.File(file_path, 'r') as f:
            current_group = f[current_key]
            current_obs = current_group['obs']['birdview']['birdview'][:]
            action = current_group['supervision']['action'][:]
            next_obs = f[next_key]['obs']['birdview']['birdview'][:]
            reward = current_group['reward'][()]
        current_obs_tensor = torch.tensor(current_obs, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        terminal_tensor = torch.tensor(terminal, dtype=torch.float32)
        return (current_obs_tensor, action_tensor, next_obs_tensor, reward_tensor, terminal_tensor, False)
    
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
# =============================================================================
class GoalDataset(Dataset):
    def __init__(self, seed, cfg):
        seed_all(seed)
        self.seed = seed
        self.p = cfg.p
        self.val_ratio = cfg.val_ratio
        self.index_list = list()
        
        hdf5_paths = list()
        for t in cfg.data_town:
            hp = glob.glob(os.path.join(os.path.dirname(os.getcwd()),
                                        f'datasets/{cfg.data_algo}/{cfg.data_benchmark}/{t.lower()}_*.hdf5'))
            hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                step_keys = sorted([k for k in f.keys() if k.startswith("step_")], 
                                   key=lambda x: int(x.split("_")[1]))
                epi_length = len(step_keys)
                for i in range(epi_length):
                    current_step_key = step_keys[i]
                    if i == (epi_length - 1):
                        next_step_key = step_keys[i]
                        goal_step_key = step_keys[i]
                    else:
                        step = 1
                        max_future_step = (epi_length - 1) - i
                        next_step_key = step_keys[min(i+1, epi_length-1)]
                        while True:
                            if random.random() < self.p:
                                break
                            step += 1
                            if step >= max_future_step:
                                step = max_future_step
                                break
                        goal_step_key = step_keys[min(i+step, epi_length-1)]
                    self.index_list.append((hdf5_path, current_step_key, next_step_key, goal_step_key))
                            
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        file_path, current_key, next_key, goal_key = self.index_list[index]
        
        with h5py.File(file_path, 'r') as f:
            current_obs = f[current_key]['obs']['birdview']['birdview'][:]
            next_obs = f[next_key]['obs']['birdview']['birdview'][:]
            goal_obs = f[goal_key]['obs']['birdview']['birdview'][:]
            
        current_obs_tensor = torch.tensor(current_obs, dtype=torch.float32)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
        goal_obs_tensor = torch.tensor(goal_obs, dtype=torch.float32)
        
        return (current_obs_tensor, next_obs_tensor, goal_obs_tensor)
    
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
# HILPDataset: {(s, z, s', R, terminal)_0^T}
# =============================================================================
class HILPDataset(Dataset):
    def __init__(self, seed, gamma, encoder, cfg):
        self.seed = seed
        seed_all(self.seed)
        self.val_ratio = cfg.val_ratio
        self.gamma = gamma
        self.encoder = encoder
        self.algo = cfg.algo
        
        self.index_list = list()
        hdf5_paths = []
        for t in self.town:
            hp = glob.glob(os.path.join(os.path.dirname(os.getcwd()), 
                                        f'datasets/{cfg.data_algo}/{cfg.data_benchmark}/{t.lower()}_*.hdf5'))
            hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                step_keys = sorted([k for k in f.keys() if k.startswith("step_")], 
                                   key=lambda x: int(x.split("_")[1]))
                epi_length = len(step_keys)
                
                for i in range(epi_length):
                    if i in range(epi_length - cfg.length + 1, epi_length):
                        next_idx = epi_length - 1
                        terminal = True
                    else:
                        next_idx = i + cfg.length - 1
                        terminal = False
                    current_step_key = step_keys[i]
                    next_step_key = step_keys[next_idx]
                    self.index_list.append((hdf5_path, current_step_key, next_step_key, terminal, i, next_idx, step_keys))
    
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        file_path, current_key, end_key, terminal, start_idx, end_idx, step_keys = self.index_list[index]
        with h5py.File(file_path, 'r') as f:
            current_obs = f[current_key]['obs']['birdview']['birdview'][:]
            next_obs = f[end_key]['obs']['birdview']['birdview'][:]
            
            if self.algo == "hilp":
                with torch.no_grad():
                    z = self.encoder(current_obs)
                    z_next = self.encoder(next_obs)
                    vec = z_next - z
                    norm = np.linalg.norm(vec)
                    z = vec / (norm + 1e-6)
            else:
                with torch.no_grad():
                    traj = []
                    for j in range(start_idx, end_idx):
                        s = f[step_keys[j]]['obs']['birdview']['birdview'][:]
                        a = f[step_keys[j]]['supervision']['action'][:]
                        s_next = f[step_keys[min(j+1, end_idx)]]['obs']['birdview']['birdview'][:]
                        r = f[step_keys[j]]['reward'][()]
                        terminal = f[step_keys[j]]['terminal'][()]
                        traj.append([s, a, s_next, r, terminal, False])
                    s, a, s_n, r, terminal, timeout = zip(*traj)
                    s = torch.tensor(np.array(s).transpose(0, 3, 1, 2), 
                            dtype=torch.float32).contiguous().clone()
                    a = torch.tensor(np.array(a), 
                             dtype=torch.float32).contiguous().clone()
                    s_n = torch.tensor(np.array(s_n).transpose(0, 3, 1, 2), 
                                 dtype=torch.float32).contiguous().clone()
                    r = torch.tensor(np.array(r), 
                             dtype=torch.float32).unsqueeze(1).contiguous().clone()
                    terminal = torch.tensor(np.array(terminal), 
                               dtype=torch.float32).unsqueeze(1).contiguous().clone()
                    timeout = torch.tensor(np.array(timeout), 
                              dtype=torch.float32).unsqueeze(1).contiguous().clone()
                    subtraj = (s, a, s_n, r, terminal, timeout)

                    z = self.encoder(subtraj)
                
            R = 0.0
            for j in range(start_idx, end_idx):
                r_j = f[step_keys[j]]['reward'][()]
                R += (self.gamma ** (j-start_idx)) * r_j
        
        return (current_obs, z, next_obs, R, terminal, False)
         
    
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.index_list[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.index_list[i] for i in val_idxs]
        
        return train_dataset, val_dataset
    
    
###############################################################################
# SubTrajDataset (OPAL, HsO-VP)
###############################################################################
class SubTrajDataset(Dataset):
    def __init__(self, seed, cfg):
        self.seed = seed
        seed_all(self.seed)
        self.val_ratio = cfg.val_ratio
        hdf5_paths = list()
        for t in cfg.data_town:
            hp = glob.glob(os.path.join(os.path.dirname(os.getcwd()),
                                          f'datasets/{cfg.data_algo}/{cfg.data_benchmark}/{t.lower()}_*.hdf5'))
            hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        self.index_list = list()
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                step_keys = sorted([k for k in f.keys() if k.startswith("step_")],
                                   key=lambda x: int(x.split("_")[1]))
                epi_length = len(step_keys)
                last_sequence_start = epi_length - cfg.length
                for i in range(last_sequence_start + 1):
                    next_idx = i + cfg.length - 1
                    terminal = (next_idx == epi_length - 1)
                    self.index_list.append((hdf5_path, step_keys, i, next_idx, terminal))
        
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        file_path, step_keys, start_idx, end_idx, terminal = self.index_list[index]
        trajectories = []
        with h5py.File(file_path, 'r') as f:
            for j in range(start_idx, end_idx + 1):
                obs = f[step_keys[j]]['obs']['birdview']['birdview'][:]
                action = f[step_keys[j]]['supervision']['action'][:]
                next_obs = f[step_keys[min(j+1, end_idx)]]['obs']['birdview']['birdview'][:]
                reward = f[step_keys[j]]['reward'][()]
                trajectories.append([obs, action, next_obs, reward, terminal, False])
        
        s, actions, s_n, rewards, terminals, timeouts = zip(*trajectories)

        states = torch.tensor(np.array(s).transpose(0, 3, 1, 2), 
                            dtype=torch.float32).contiguous().clone()
        actions = torch.tensor(np.array(actions), 
                             dtype=torch.float32).contiguous().clone()
        next_states = torch.tensor(np.array(s_n).transpose(0, 3, 1, 2), 
                                 dtype=torch.float32).contiguous().clone()
        rewards = torch.tensor(np.array(rewards), 
                             dtype=torch.float32).unsqueeze(1).contiguous().clone()
        terminals = torch.tensor(np.array(terminals), 
                               dtype=torch.float32).unsqueeze(1).contiguous().clone()
        timeouts = torch.tensor(np.array(timeouts), 
                              dtype=torch.float32).unsqueeze(1).contiguous().clone()
        
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
    
        
class HighLevelDataset(Dataset):
    def __init__(self, encoder, subtraj_dataset, gamma=0.99):
        self.encoder = encoder
        self.subtraj_dataset = subtraj_dataset
        self.gamma = gamma
        self.encoder.eval()  # evaluation 모드로 설정
        
        # 미리 z값들을 계산해서 저장
        self.high_level_data = []
        with torch.no_grad():
            for i in range(len(subtraj_dataset)):
                states, actions, next_states, rewards, terminals, timeouts = subtraj_dataset[i]
                
                # states와 actions를 시퀀스로 준비 (s:s_c, a:a_c)
                current_states = states[:-1]  # s:s_c
                current_actions = actions[:-1]  # a:a_c
                
                # encoder를 통해 z 계산
                z_mu, z_logstd = self.encoder(current_states.unsqueeze(0), 
                                        current_actions.unsqueeze(0))
                z = Normal(z_mu, torch.exp(z_logstd)).rsample()
                R = 0.0
                for r in reversed(rewards):
                    R = r + self.gamma * R
                    
                self.high_level_data.append({
                    'initial_state': states[0],
                    'z': z.squeeze(0),
                    'final_state': states[-1],
                    'total_reward': R,
                    'terminal': terminals[-1],
                    'timeout': timeouts[-1]
                })
    
    def __len__(self):
        return len(self.high_level_data)
    
    def __getitem__(self, idx):
        data = self.high_level_data[idx]
        return (
            data['initial_state'],  # s
            data['z'],             # z
            data['final_state'],   # s_c
            data['total_reward'],  # R
            data['terminal'],      # terminal
            data['timeout']        # timeout
        )
    
    def split_train_val(self, val_ratio=0.2, seed=0):
        val_mask = get_val_mask(len(self), val_ratio, seed)
        
        train_data = [self.high_level_data[i] for i, m in enumerate(val_mask) if not m]
        val_data = [self.high_level_data[i] for i, m in enumerate(val_mask) if m]
        
        train_dataset = HighLevelDataset.from_data(self.encoder, train_data)
        val_dataset = HighLevelDataset.from_data(self.encoder, val_data)
        
        return train_dataset, val_dataset
    
    @classmethod
    def from_data(cls, encoder, data):
        dataset = cls.__new__(cls)
        dataset.encoder = encoder
        dataset.high_level_data = data
        return dataset
    
    
    
class LowLevelDataset(SubTrajDataset):
    def __init__(self, algo, benchmark, town, length, encoder, seed=42):
        super().__init__(algo, benchmark, town, length, seed)
        self.encoder = encoder
    
    def __getitem__(self, index):
        obs, actions, next_obs, reward, terminal, timeout = super().__getitem__(index)

        states = obs.unsqueeze(0)
        acts = actions.unsqueeze(0)
        
        with torch.no_grad():
            latent_mu, latent_logstd = self.encoder(states, acts)
            std = torch.exp(0.5 * latent_logstd)
            z = Normal(latent_mu, std).rsample()
            
        return (obs, actions, z)



###############################################################################
# FilteredDataset (HsO-VP): labeling sub-trajectories using K-means clustering
###############################################################################

class FilteredDataset(Dataset):
    def __init__(self, seed, cfg):
        self.seed = seed
        seed_all(self.seed)
        self.cfg = cfg
        
        hdf5_paths = list()
        for t in self.cfg.data_town:
            hp = glob.glob(os.path.join(os.path.dirname(os.getcwd()),
                                        f'datasets/{self.cfg.data_algo}/{self.cfg.data_benchmark}/{t.lower()}_*.hdf5'))
            hdf5_paths.extend(hp)
        hdf5_paths = sorted(hdf5_paths)
        
        self.index_list = list()
        action_feature_list = list()
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                step_keys = sorted([k for k in f.keys() if k.startswith("step_")],
                                   key=lambda x: int(x.split("_")[1]))
                epi_length = len(step_keys)
                last_sequence_start = epi_length - cfg.length
                for i in range(last_sequence_start + 1):
                    next_idx = i + cfg.length - 1
                    terminal = (next_idx == epi_length - 1)
                    self.index_list.append((hdf5_path, step_keys, i, next_idx, terminal))
                    
                    actions = ()
                    for j in range(i, next_idx + 1):
                        action = f[step_keys[j]]['supervision']['action'][:]
                        actions.append(actions.flatten())
                    concat_action = np.concatenate(actions, axis=0)
                    action_feature_list.append(concat_action)
        self.action_features = np.array(action_feature_list)
        
        self.num_cluster = cfg.discrete_option if hasattr(cfg, "discrete_option") else 10
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=self.seed)
        self.cluster_labels = kmeans.fit_predict(self.action_features)
        self.cluster_centers = kmeans.cluster_centers_
        
        clusters = dict()
        for cluster_id in range(self.num_cluster):
            clusters[cluster_id] = np.where(self.cluster_labels == cluster_id)[0]
            
        min_cluster_size = min(len(indices) for indices in clusters.values())
        selected_indices = []
        for cluster_id, indices in clusters.items():
            center = self.cluster_centers[cluster_id]
            actions_in_cluster = self.action_features[indices]
            distances = np.linalg.norm(actions_in_cluster - center, axis=1)
            sorted_order = np.argsort(distances)
            selected = indices[sorted_order[:min_cluster_size]]
            selected_indices.extend(selected.tolist())
        
        self.filtered_index_list = [self.index_list[i] for i in selected_indices]
        self.filtered_cluster_labels = self.cluster_labels[selected_indices]
        
    def __len__(self):
        return len(self.filtered_index_list)
    
    def __getitem__(self, index):
        file_path, step_keys, start_idx, end_idx, terminal = self.filtered_index_list[index]
        cluster_label = int(self.filtered_cluster_labels[index])
        trajectories = list()
        with h5py.File(file_path, 'r') as f:
            for j in range(start_idx, end_idx + 1):
                obs = f[step_keys[j]]['obs']['birdview']['birdview'][:]
                action = f[step_keys[j]]['supervision']['action'][:]
                next_obs = f[step_keys[min(j+1, end_idx)]]['obs']['birdview']['birdview'][:]
                reward = f[step_keys[j]]['reward'][()]
                trajectories.append([obs, action, next_obs, reward, terminal, False])
                
            s, a, s_n, r, terminal, timeout = zip(*trajectories)
            s = torch.tensor(np.array(s).transpose(0, 3, 1, 2), dtype=torch.float32).contiguous().clone()
            a = torch.tensor(np.array(a), dtype=torch.float32).contiguous().clone()
            s_n = torch.tensor(np.array(s_n).transpose(0, 3, 1, 2), dtype=torch.float32).contiguous().clone()
            r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1).contiguous().clone()
            terminal = torch.tensor(np.array(terminal), dtype=torch.float32).unsqueeze(1).contiguous().clone()
            timeout = torch.tensor(np.array(timeout), dtype=torch.float32).unsqueeze(1).contiguous().clone()
            
            return (s, a, s_n, r, terminal, timeout, cluster_label)
        
    def split_train_val(self):
        val_mask = get_val_mask(len(self), self.val_ratio, self.seed)
        train_idxs = [i for i, m in enumerate(val_mask) if not m]
        val_idxs = [i for i, m in enumerate(val_mask) if m]
        
        train_dataset = copy.copy(self)
        train_dataset.index_list = [self.filtered_index_list[i] for i in train_idxs]
        train_dataset.filtered_cluster_labels = [self.filtered_cluster_labels[i] for i in train_idxs]
        
        val_dataset = copy.copy(self)
        val_dataset.index_list = [self.filtered_index_list[i] for i in val_idxs]
        val_dataset.filtered_cluster_labels = [self.filtered_cluster_labels[i] for i in val_idxs]
        
        return train_dataset, val_dataset