import os
import gym
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

import wandb

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device):
        self.max_size = int(1e6)
        self.device = device
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))
        self.c = np.zeros((self.max_size, 1)) # cost
        self.cv = np.zeros((self.max_size, 1)) # constraint violation

    def store(self, s, a, r, s_, dw, c, cv):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.c[self.count] = c
        self.cv[self.count] = cv
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(self.device)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(self.device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(self.device)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(self.device)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(self.device)
        batch_c = torch.tensor(self.c[index], dtype=torch.float).to(self.device)
        batch_cv = torch.tensor(self.cv[index], dtype=torch.float).to(self.device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_c, batch_cv


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, config):
        self.max_action = max_action
        self.hidden_width = config['hidden_sizes']  # The number of neurons in hidden layers of the neural network
        self.device = config['device']
        self.batch_size = config['batch_size']  # batch size
        self.GAMMA = config['gamma']  # discount factor
        self.TAU = config['tau']  # Softly update the target network
        self.lr = config['lr']  # learning rate
        self.actor_lr = config['actor_lr']
        self.adaptive_alpha = config['adaptive_alpha']  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, dtype=torch.float).to(self.device).requires_grad_(True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.actor_lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.cpu().data.numpy().flatten()

    def learn(self, relay_buffer):
        result = {}
        batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_c, batch_cv = relay_buffer.sample(self.batch_size)  # Sample a batch

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Update alpha
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
        
        result.update({
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.log_alpha.exp().detach().mean(),
            'current_q': current_Q1.detach().mean(),
            'target_q': target_Q.detach().mean()
        })
        return result
    
    def save(self, filedir):
        torch.save(self.actor.state_dict(), os.path.join(filedir, 'policy_network.pth'))

    def load(self, filedir):
        self.actor.load_state_dict(torch.load(os.path.join(filedir, 'policy_network.pth')))


def evaluate_policy(env, agent):
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

            c = info['violation']

            episode_reward += r
            episode_cost += c

            s = s_
        evaluate_reward += episode_reward
        evaluate_cost += episode_cost

    return int(evaluate_reward / times), int(evaluate_cost / times)

def run(config):
    
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

    exp_name = config['algo'] + '_' + env_name
    if config['wandb'] == True:
        wandb.init(project=exp_name, name='test_s'+str(config['seed']), reinit=True, mode='online')

    # Prepare Logger
    logger_kwargs = utils.setup_logger_kwargs(exp_name, config['seed'])

    logger = utils.Logger(**logger_kwargs)
    logger.save_config(config)

    agent = SAC(state_dim, action_dim, max_action, config)
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
            c = info['violation']
            cv = torch.tensor(info['constraint_value'], dtype=torch.float)

            replay_buffer.store(s, a, r, s_, done, c, cv)  # Store the transition
            s = s_

            if total_steps >= config['random_steps']:
                result = agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % config['evaluate_freq'] == 0:
                evaluate_num += 1
                evaluate_reward, evaluate_cost = evaluate_policy(env_evaluate, agent)
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

            #print(total_steps)

    # Save model
    dir = logger.get_dir()
    agent.save(dir)
    print('model saved')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='SAC', type=str)
    parser.add_argument('--env_name', default='point', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--wandb', default=False, type=boolean)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--max_train_steps', default=int(3e6), type=int)
    parser.add_argument('--evaluate_freq', default=int(5e3), type=int)
    parser.add_argument('--random_steps', default=int(5e3), type=int)

    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=5.0, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--adaptive_alpha', default=True, type=boolean)
    parser.add_argument('--hidden_sizes', default=256)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=3e-5, type=float)

    parser.add_argument('--goal_met_done', default=False, type=boolean)
    parser.add_argument('--violation_done', default=False, type=boolean)


    return parser

if __name__ == '__main__':
    args = get_parser().parse_args() 
    run(vars(args))
