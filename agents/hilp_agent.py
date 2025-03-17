import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import logging
import numpy as np
import torch
import carla
from omegaconf import OmegaConf
import copy
import cv2
from carla_gym.utils.config_utils import load_entry_point
from opal.model.policy import GaussianPolicyforHilbert, GaussianPolicy

class HilpAgent:
    def __init__(self,
                 path_to_conf_file='config_agent.yaml'):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.supervision_dict = None
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        cfg = OmegaConf.load(path_to_conf_file)
        
        # model
        # high policy: pi(z|s)
        self.high_policy = GaussianPolicy(cfg.model.state_dim, cfg.model.latent_dim)
        # low policy: pi(a|s, z)
        self.low_policy = GaussianPolicyforHilbert(cfg.model.state_dim, cfg.model.latent_dim, cfg.model.action_dim)
        
        # load checkpoint
        high_policy_ckpt = torch.load(cfg.model.high_policy_checkpoint_path)
        low_policy_ckpt = torch.load(cfg.model.low_policy_checkpoint_path)
        self.high_policy.load_state_dict(high_policy_ckpt['policy_state_dict'])
        self.low_policy.load_state_dict(low_policy_ckpt['policy_state_dict'])
        self.high_policy.eval()
        self.low_policy.eval()
        
        self._obs_configs = cfg['obs_configs']
        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

    def run_high_step(self, input_data, timestamp):
        input_data = copy.deepcopy(input_data)
        policy_input = self._wrapper_class.process_obs(input_data, self._wrapper_kwargs['input_states'], train=False)
        with torch.no_grad():
            print(policy_input['birdview'])
            birdview = policy_input['birdview']
            birdview = torch.FloatTensor(birdview).unsqueeze(0)
            z = self.high_policy.act(birdview)
        return z
    
    def run_low_step(self, input_data, z, timestamp):
        input_data = copy.deepcopy(input_data)
        policy_input = self._wrapper_class.process_obs(input_data, self._wrapper_kwargs['input_states'], train=False)
        with torch.no_grad():
            birdview = policy_input['birdview']
            birdview = torch.FloatTensor(birdview).unsqueeze(0)
            a_mu, a_logstd = self.low_policy(birdview, z)
            a_std = torch.exp(torch.clamp(a_logstd, min=-10, max=2))
            
            dist = torch.distributions.Normal(a_mu, a_std)
            action = torch.tanh(dist.sample())
            
            throttle, steer, brake = action.cpu().numpy()
            throttle = np.clip(throttle, 0, 1)
            steer = np.clip(steer, -1, 1)
            brake = np.clip(brake, 0, 1)
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            
            self.supervision_dict = {
                'action': np.array([control.throttle, control.steer, control.brake], dtype=np.float32),
                'value': 0.0,
                'action_mu': a_mu.cpu().numpy(),
                'action_sigma': a_std.cpu().numpy(),
                'features': None,
                'speed': input_data['speed']['forward_speed']
            }
            self.supervision_dict = copy.deepcopy(self.supervision_dict)
            
            self._render_dict = {
                'timestamp': timestamp,
                'obs': input_data,
                'im_render': input_data['birdview']['rendered'],
                'action': action.cpu().numpy(),
                'action_value': 0.0,
                'action_log_probs': 0.0,
                'action_mu': a_mu.cpu().numpy(),
                'action_sigma': a_std.cpu().numpy(),
            }
            self._render_dict = copy.deepcopy(self._render_dict)
        return control

    def reset(self, log_file_path):
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

    def render(self, reward_debug, terminal_debug):
        self._render_dict['reward_debug'] = reward_debug
        self._render_dict['terminal_debug'] = terminal_debug

        return self.im_render(self._render_dict)
    
    def im_render(self, render_dict):
        im_birdview = render_dict['im_render']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)
        mu_str = np.array2string(render_dict['action_mu'], precision=2, separator=',', suppress_small=True)
        sigma_str = np.array2string(render_dict['action_sigma'], precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(render_dict['obs']['state'], precision=2, separator=',', suppress_small=True)

        txt_t = f'step:{render_dict["timestamp"]["step"]:5}, frame:{render_dict["timestamp"]["frame"]:5}'
        im = cv2.putText(im, txt_t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_1 = f'a{action_str} v:{render_dict["action_value"]:5.2f} p:{render_dict["action_log_probs"]:5.2f}'
        im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt_2 = f's{state_str}'
        im = cv2.putText(im, txt_2, (3, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        txt_3 = f'a{mu_str} b{sigma_str}'
        im = cv2.putText(im, txt_3, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for i, txt in enumerate(render_dict['reward_debug']['debug_texts'] +
                                render_dict['terminal_debug']['debug_texts']):
            im = cv2.putText(im, txt, (w, (i+2)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im

    @property
    def obs_configs(self):
        return self._obs_configs