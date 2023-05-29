import os
import gym
import safety_gym
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal
import utils
from utils import boolean
from env.sg.sg import SafetyGymWrapper
from run_sac import SAC, ReplayBuffer
import wandb

class Cost_Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width, use_softplus=False):
        super(Cost_Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

        self.use_softplus = use_softplus

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        if self.use_softplus:
            q1 = F.softplus(self.l3(q1))
        else:
            q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        if self.use_softplus:
            q2 = F.softplus(self.l6(q2))
        else:
            q2 = self.l6(q2)
        
        return q1, q2

class SACL(SAC):
    def __init__(self, state_dim, action_dim, max_action, config):
        super().__init__(state_dim, action_dim, max_action, config)

        self.cost_critic = Cost_Critic(state_dim, action_dim, self.hidden_width, use_softplus=config['use_softplus']).to(self.device)
        self.cost_critic_target = copy.deepcopy(self.cost_critic).to(self.device)

        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.lr)

        


def evaluate_policy(env, agent, env_name):
    times = 10  # Perform evaluations and calculate the average
    evaluate_reward = 0
    evaluate_cost = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, info = env.step(a)
            if env_name == 'simple_sg':
                c = info['violation']
            elif env_name == 'safety_gym':
                c = info['cost']
        
            episode_reward += r
            episode_cost += c

            s = s_
        evaluate_reward += episode_reward
        evaluate_cost += episode_cost

    return int(evaluate_reward / times), int(evaluate_cost / times)

def run(config):
    if config['environment'] == 'simple_sg':
        env_list = ['point', 'car']
        env_name = config['env_name']
        assert env_name in env_list, "Invalid Env"
        if config['violation_done']:
            env = SafetyGymWrapper(
                robot_type=env_name,
                id=None,    # train: done on violation; eval: false
            )
        else:
            env = SafetyGymWrapper(
                robot_type=env_name,
                id=1,    # train: done on violation; eval: false
            )

        env_evaluate = SafetyGymWrapper(
            robot_type=env_name,
            id=1,    # train: done on violation; eval: false
        )
        exp_name = config['algo'] + '_' + env_name

    elif config['environment'] == 'safety_gym':
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
    
        env = gym.make(env_name)
        env_evaluate = gym.make(env_name)

    seed = config['seed']
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    if config['environment'] == 'safety_gym':
        max_episode_steps = 1000
    else:
        max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))
    config.update({
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'max_episode_steps': max_episode_steps
    })

    
    if config['wandb'] == True:
        wandb.init(project=exp_name, name='test_s'+str(config['seed']), reinit=True, mode='online')

    # Prepare Logger
    logger_kwargs = utils.setup_logger_kwargs(exp_name, config['seed'])

    logger = utils.Logger(**logger_kwargs)
    logger.save_config(config)

    agent = SACL(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(state_dim, action_dim, config['device'])

    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training
    result = {}

    while total_steps < config['max_train_steps']:
        s = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            if total_steps < config['random_steps']:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                a = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            if config['goal_met_done']:
                info_keys = info.keys()
                goal_met = ('goal_met' in info_keys)
                done = done or goal_met # collision not done, reach goal done and terminal done
            
            if config['environment'] == 'safety_gym':
                c = info['cost']
                cv = c
            else:
                c = info['violation']
                cv = torch.tensor(info['constraint_value'], dtype=torch.float)

            replay_buffer.store(s, a, r, s_, done, c, cv)  # Store the transition
            s = s_

            if total_steps >= config['random_steps']:
                result = agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % config['evaluate_freq'] == 0:
                evaluate_num += 1
                evaluate_reward, evaluate_cost = evaluate_policy(env_evaluate, agent, config['environment'])
                result.update({'ep_r': evaluate_reward, 'ep_c': evaluate_cost})
                for k, v in sorted(result.items()):
                    print(f'- {k:23s}:{v:15.10f}')
                print(f'iteration={total_steps}')
                if config['wandb']:
                    wandb.log(result)

                # Save model
                dir = logger.get_dir()
                agent.save(dir)
                print('model saved')


            total_steps += 1

        print('one episode finish, episode_steps={}'.format(episode_steps))

    # Save model
    dir = logger.get_dir()
    agent.save(dir)
    print('model saved')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='SAC', type=str)
    parser.add_argument('--environment', default='safety_gym', type=str) # simple_sg or safety_gym
    parser.add_argument('--robot', default='Point', type=str)
    parser.add_argument('--task', default='Goal1', type=str)
    parser.add_argument('--env_name', default='point', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--wandb', default=False, type=boolean)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--max_train_steps', default=int(3e6), type=int)
    parser.add_argument('--evaluate_freq', default=int(5e3), type=int)
    parser.add_argument('--random_steps', default=int(5e3), type=int)

    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--adaptive_alpha', default=True, type=boolean)
    parser.add_argument('--hidden_sizes', default=256)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=3e-5, type=float)
    parser.add_argument('--penalty_lr', default=3e-5, type=float)

    parser.add_argument('--goal_met_done', default=False, type=boolean)
    parser.add_argument('--violation_done', default=False, type=boolean)

    parser.add_argument('--use_softplus', default=True, type=boolean)

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args() 
    run(vars(args))
