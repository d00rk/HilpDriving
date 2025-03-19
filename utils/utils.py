import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import numpy as np

def kl_divergence(mu1, logstd1, mu2, logstd2):
    kl = 0.5 * (logstd2 - logstd1 + (torch.exp(logstd1) + (mu1 - mu2).pow(2)) / torch.exp(logstd2) - 1)
    return kl.sum()

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def l2_expectile_loss(x, tau):
    return torch.mean(torch.abs(tau - (x < 0).float() * (x ** 2)))

def update_exponential_moving_average(target, source, alpha):
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)
        
def log_sum_exp(tensor, dim=1, keepdim=False):
    m, _ = torch.max(tensor, dim=dim, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(tensor - m), dim=dim, keepdim=True))

def normalize_control(control):
    return float(np.clip(control, -1.0, 1.0))

def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRU) or isinstance(m, nn.GRUCell):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

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