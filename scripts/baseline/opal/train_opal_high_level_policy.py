import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
import datetime as dt
from PIL import Image
import click
import tqdm
import shutil
import glob
from omegaconf import OmegaConf

from policy.iql import ImplicitQLearning
from policy.cql import ConservativeQLearning
from model.opal import Encoder
from model.value_function import *
from opal.model.policy import *
from dataset.dataset import HighLevelDataset, SubTrajDataset
from utils.seed import seed_all


def eval(algo,
         agent, 
         dataloader, 
         device, 
         num_evals):
    eval_loss = []
    for _ in range(num_evals):
        total_loss = 0.0
        for states, zs, next_states, rewards, terminals, _ in dataloader:
            states, zs, next_states, rewards, terminals = states.to(device), zs.to(device), next_states.to(device), rewards.to(device), terminals.to(device)
            loss_dict = agent.evaluate(states, zs, next_states, rewards, terminals)
            if algo == "iql":
                total_loss += loss_dict["v_loss"] + loss_dict["q_loss"] + loss_dict["policy_loss"]
            elif algo == "cql":
                total_loss += loss_dict["total_q_loss"] + loss_dict["policy_loss"]
        avg_loss = total_loss /len(dataloader)
        eval_loss.append(avg_loss)
    eval_loss = torch.tensor(eval_loss, dtype=torch.float32).cpu().numpy()
    return np.mean(eval_loss)
            
def train(algo,
          model, 
          q_function, 
          policy, 
          device, 
          num_epochs, 
          num_evals, 
          eval_frequency, 
          train_dataloader, 
          val_dataloader, 
          lr, 
          tau,
          beta,
          verbose,
          wb,
          checkpoint_dir,
          v_function=None):
    optim_factory = lambda params: torch.optim.Adam(params, lr=lr)
    if algo == "iql":
        agent = ImplicitQLearning(device=device, 
                            qf=q_function, 
                            vf=v_function, 
                            policy=policy, 
                            optimizer_factory=optim_factory, 
                            max_steps=10**6, 
                            tau=tau, 
                            beta=beta)
    elif algo == "cql":
        agent = ConservativeQLearning(device=device, 
                            qf=q_function, 
                            policy=policy, 
                            optimizer_factory=optim_factory, 
                            max_steps=10**6)
    else:
        raise ValueError(f"Invalid algorithm: {algo}")
    
    model = model.to(device)
    
    best_loss = np.inf
    
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if verbose:
        print(f"[torch] {device} is used.")
    
    global_step = 0
    for epoch in range(num_epochs):
        if algo == "iql":
            total_v_loss = 0.0
        if algo == "cql":
            total_cql_loss = 0.0
        total_q_loss = 0.0
        total_policy_loss = 0.0
        total_loss = 0.0
        
        progress_bar = tqdm.tqdm(enumerate(train_dataloader), 
                                 total=len(train_dataloader), 
                                 desc=f"[Epoch {epoch}/{num_epochs}]", 
                                 leave=True, 
                                 ncols=100)
        for i, (state, z, next_state, rewards, terminal, _) in progress_bar:
            state, z, next_state, rewards, terminal = state.to(device), z.to(device), next_state.to(device), rewards.to(device), terminal.to(device)  
            loss_dict = agent.update(state, z, next_state, rewards, terminal)
            if algo == "iql":
                t_loss = loss_dict["v_loss"] + loss_dict["q_loss"] + loss_dict["policy_loss"]
                total_loss += t_loss
                total_v_loss += loss_dict["v_loss"]
                total_q_loss += loss_dict["q_loss"]
                total_policy_loss += loss_dict["policy_loss"]
            elif algo == "cql":
                t_loss = loss_dict["total_q_loss"] + loss_dict["policy_loss"]
                total_loss += t_loss
                total_cql_loss += loss_dict["cql_loss_total"]
                total_q_loss += loss_dict["total_q_loss"]
                total_policy_loss += loss_dict["policy_loss"]
            global_step += 1
            progress_bar.set_postfix({
                "Total Loss": f"{t_loss:.4f}",
            })
        
        if algo == "iql":
            avg_v_loss = total_v_loss / len(train_dataloader)
        elif algo == "cql":
            avg_cql_loss = total_cql_loss / len(train_dataloader)
        avg_q_loss = total_q_loss / len(train_dataloader)
        avg_policy_loss = total_policy_loss / len(train_dataloader)
        avg_loss = total_loss / len(train_dataloader)
        
        if verbose and algo == "iql":
            print(f"[Train] Epoch [{epoch+1}/{num_epochs}]   |   Total V Loss: {avg_v_loss:.4f}   |   Total Q Loss: {avg_q_loss:.4f}   |   Total Policy Loss: {avg_policy_loss:.4f}   |   Total Loss: {avg_loss:.4f}")
        elif verbose and algo == "cql":
            print(f"[Train] Epoch [{epoch+1}/{num_epochs}]   |   Total CQL Loss: {avg_cql_loss:.4f}   |   Total Q Loss: {avg_q_loss:.4f}   |   Total Policy Loss: {avg_policy_loss:.4f}   |   Total Loss: {avg_loss:.4f}")
        
        if wb and algo == "iql":
            wandb.log({"train/epoch": epoch,
                       "train/global_step": global_step,
                       "train/loss": avg_loss,
                       "train/q_loss": avg_q_loss,
                       "train/v_loss": avg_v_loss,
                       "train/policy_loss": avg_policy_loss})
        elif wb and algo == "cql":
            wandb.log({"train/epoch": epoch,
                       "train/global_step": global_step,
                       "train/loss": avg_loss,
                       "train/q_loss": avg_q_loss,
                       "train/cql_loss": avg_cql_loss,
                       "train/policy_loss": avg_policy_loss})
        
        if (epoch) % eval_frequency == 0:
            if verbose:
                print(f"[Evaluation] {epoch} / {num_epochs}")
            
            eval_loss = eval(algo=algo,
                             agent=agent,
                             dataloader=val_dataloader, 
                             device=device, 
                             num_evals=num_evals)
            
            if verbose:
                print(f"[Evaluation] Loss: {eval_loss:.4f}")
            if wb:
                wandb.log({"eval/loss": eval_loss})
                
            if eval_loss <= best_loss:
                if verbose:
                    print(f"Save best model of epoch {epoch}")
                
                checkpoint_path = os.path.join(checkpoint_dir, f"high_level_policy_{algo}_{dt.datetime().now().replace(microsecond=0)}_epoch_{epoch}.pt")
                if algo == "iql":
                    torch.save({
                        'epoch': epoch,
                        'q_state_dict': q_function.state_dict(),
                        'v_state_dict': v_function.state_dict(),
                        'policy_state_dict': policy.state_dict(),
                        'loss': avg_loss,
                        'eval_loss': eval_loss  
                    }, checkpoint_path)
                elif algo == "cql":
                    torch.save({
                        'epoch': epoch,
                        'q_state_dict': q_function.state_dict(),
                        'policy_state_dict': policy.state_dict(),
                        'loss': avg_loss,
                        'eval_loss': eval_loss  
                    }, checkpoint_path)
                best_loss = eval_loss
        
    if verbose:
        print(f"[Train] Finished at {dt.datetime.now().replace(microsecond=0)}")
        
@click.command()
@click.option("-c", "--config", type=str, required=True, default='train_opal_high_level_policy', help="config file name")
def main(config):
    CONFIG_FILE = os.path.join(os.path.dirname(os.getcwd()), f'opal/config/{config}.yaml')
    conf = OmegaConf.load(CONFIG_FILE)
    
    if conf.train.resume:
        conf = OmegaConf.load(os.path.join(os.path.dirname(os.getcwd()), f'opal/outputs/high_level_policy/{conf.resume_ckpt_dir}/{config}.yaml'))
    
    seed = conf.seed
    seed_all(seed)
    
    data_algo = conf.data_algo
    data_benchmark = conf.data_benchmark
    data_town = conf.data_town
    
    encoder_dir = conf.encoder_dir
    encoder_dict_name = conf.encoder_dict_name
    ENCODER_DICT_PATH = os.path.join(os.path.dirname(os.getcwd()), f'opal/outputs/opal/{encoder_dir}/{encoder_dict_name}.pt')
    
    state_dim = conf.model.state_dim
    action_dim = conf.model.action_dim
    latent_dim = conf.model.latent_dim
    
    val_ratio = conf.dataset.val_ratio
    num_workers = conf.dataset.num_workers
    train_batch_size = conf.dataset.train_batch_size
    val_batch_size = conf.dataset.val_batch_size
    trajectory_length = conf.dataset.trajectory_length
    
    algo = conf.train.algo
    device = conf.train.device
    num_epochs = conf.train.num_epochs
    num_evals = conf.train.num_evals
    eval_frequency = conf.train.eval_frequency
    lr = conf.train.lr
    tau = conf.train.tau
    beta = conf.train.beta  
    resume = conf.train.resume
    
    verbose = conf.verbose
    wb = conf.wb
    wandb_project = conf.wandb_project
    wandb_name = conf.wandb_name
    wandb_tag = conf.wandb_tag
    
    if wb:
        wandb.init(project=wandb_project,
                   config=OmegaConf.to_container(conf, resolve=True))
        wandb.run.tags = wandb_tag
        wandb.run.name = f"{wandb_name}-{dt.datetime.now().replace(microsecond=0)}"
    
    # pretrained encoder
    encoder = Encoder(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim)
    ckpt = torch.load(ENCODER_DICT_PATH)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    
    if verbose:
        print("Create dataset")
    subtraj_dataset = SubTrajDataset(algo=data_algo, benchmark=data_benchmark, town=data_town, length=trajectory_length)
    dataset = HighLevelDataset(encoder, subtraj_dataset)
    train_dataset, val_dataset = dataset.split_train_val(val_ratio=val_ratio, seed=seed)
    if verbose:
        print("Created Dataset.")
        
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=num_workers)
    
    q_func = TwinQ(state_dim, latent_dim)
    policy = GaussianPolicy(state_dim, latent_dim)
    
    if resume:
        resume_ckpt_dir = conf.resume_ckpt_dir
        ckpts = sorted(glob.glob(os.path.join(os.path.dirname(os.getcwd()), f"opal/outputs/opal/{resume_ckpt_dir}", f"opal_high_level_policy_{algo}_*.pt")))
        ckpt = torch.load(ckpts[-1])
        q_func.load_state_dict(ckpt['q_state_dict'])
        policy.load_state_dict(ckpt['policy_state_dict'])
    
    if algo == "iql":
        v_func = ValueFunction(state_dim)
        if resume:
            v_func.load_state_dict(ckpt['v_state_dict'])
    else:
        v_func = None
        
    checkpoint_dir = os.path.join(os.path.dirname(os.getcwd()), f'outputs/opal/{dt.datetime.now().replace(microsecond=0)}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    shutil.copy(CONFIG_FILE, os.path.join(checkpoint_dir, f"{config}.yaml"))
        
    train(algo=algo,
          model=encoder, 
          q_function=q_func, 
          v_function=v_func, 
          policy=policy, 
          device=device, 
          num_epochs=num_epochs, 
          num_evals=num_evals, 
          eval_frequency=eval_frequency, 
          train_dataloader=train_dataloader, 
          val_dataloader=val_dataloader, 
          lr=lr, 
          tau=tau,
          beta=beta,
          verbose=verbose,
          wb=wb,
          checkpoint_dir=checkpoint_dir)
    

if __name__=="__main__":
    main()