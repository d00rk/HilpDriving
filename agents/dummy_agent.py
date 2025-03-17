import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import random
import torch
import carla
import numpy as np
import logging
from omegaconf import OmegaConf

class DummySkillAgent:
    def __init__(self, skill_dim):
        self.skill_dim = skill_dim
    
    def act(self, obs):
        return torch.randn(self.skill_dim)

class DummyAgent:
    def __init__(self, path_to_conf_file='config_agent.yaml'):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.supervision_dict = None
        self.setup(path_to_conf_file)
        self.inference_times = []

    def setup(self, path_to_conf_file):
        cfg = OmegaConf.load(path_to_conf_file)
        self._obs_configs = cfg['obs_configs']

    def run_step(self, input_data, timestamp):
        # 랜덤 행동 생성
        throttle = np.random.uniform(0.0, 1.0)
        steer = np.random.uniform(-1.0, 1.0)
        brake = np.random.uniform(0.0, 1.0)
        
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        
        return control

    def reset(self, log_file_path):
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)
        self.inference_times = []

    def render(self, reward_debug, terminal_debug):
        return np.zeros((192, 192, 3), dtype=np.uint8)  # 더미 이미지 반환

    @property
    def obs_configs(self):
        return self._obs_configs