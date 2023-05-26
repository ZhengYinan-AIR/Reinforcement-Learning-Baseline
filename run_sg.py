import os
import time

import gym
import safety_gym

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
from tqdm import tqdm

import wandb

np.set_printoptions(precision=3, suppress=True)

def run(config):

    # Verify experiment
    robot_list = ['Point', 'Car', 'Doggo']
    task_list = ['Goal1', 'Goal2', 'Button1', 'Button2', 'Push1', 'Push2']

    algo = config['algo']
    task = config['task']
    robot = config['robot']
    assert task in task_list, "Invalid task"
    assert robot in robot_list, "Invalid robot"
    exp_name = algo + '_' + robot + task

    env_name = 'Safexp-' + robot + task + '-v0'
    
    eval_env = gym.make(env_name)

    # Seeding
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed']) 
    eval_env.seed(config['seed'])

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    print('state_dim:', state_dim, '\t action_dim:', action_dim)
    

if __name__ == "__main__":
    from utils.config import get_parser
    args = get_parser().parse_args() 
    
    run(vars(args))
