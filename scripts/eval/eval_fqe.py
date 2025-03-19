import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model.value_function import *
from model.policy import *


# FQE로 학습한 Q 네트워크를 이용해 policy의 성능 평가 (평균 Q value 추정)
def evaluate_policy(q_network, policy_fn, dataloader, device='cpu'):
    q_network.eval()
    total_q = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            state, _, _, _, _ = batch  # 평가 시 state에서 policy를 통해 행동 선택
            state = state.to(device)
            action = policy_fn(state)
            q_values = q_network(state, action)
            total_q += q_values.sum().item()
            count += state.size(0)
    avg_q = total_q / count
    return avg_q


@click.command()
@click.option("--config", type=str, default="eval_fqe", help="config file name")
def main(config):
    cfg = OmegaConf.load(f"opal/config/{config}.yaml")
    
    q_function = TwinQ(cfg.model)
    policy = GaussianPolicy(cfg.model)
    
    if cfg.algo in ["hilp", "opal", "hsovp"]:
        dataset = LatentDataset(cfg.seed, cfg.gamma, hilbert_representation, cfg.data)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 더미 데이터셋 생성: (state, action, reward, next_state, done)
    num_samples = 1000
    state_dim = 4     # 예: CartPole 같은 환경의 상태 차원
    action_dim = 1    # 예: 연속 행동 (실제 적용 시 discrete 혹은 연속에 맞게 조정)
    
    # 임의의 데이터 생성 (실제 연구에서는 offline dataset 사용)
    states = np.random.rand(num_samples, state_dim).astype(np.float32)
    actions = np.random.rand(num_samples, action_dim).astype(np.float32)
    rewards = np.random.rand(num_samples, 1).astype(np.float32)
    next_states = np.random.rand(num_samples, state_dim).astype(np.float32)
    dones = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
    
    dataset = TensorDataset(torch.tensor(states),
                            torch.tensor(actions),
                            torch.tensor(rewards),
                            torch.tensor(next_states),
                            torch.tensor(dones))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Dummy policy 인스턴스 생성 (여기에 IQL/CQL로 학습한 policy 사용 가능)
    policy_model = DummyPolicy(state_dim, action_dim).to(device)
    policy_fn = lambda state: policy_model(state)
    
    # Q-network 생성 및 FQE training
    q_network = QNetwork(state_dim, action_dim)
    print("FQE Training 시작...")
    trained_q = train_fqe(q_network, policy_fn, dataloader, num_epochs=50, device=device)
    
    # FQE로 학습한 Q-network로 policy 평가
    print("Policy 평가 중...")
    avg_q_value = evaluate_policy(trained_q, policy_fn, dataloader, device=device)
    print("평가된 평균 Q value (policy 성능 추정):", avg_q_value)
