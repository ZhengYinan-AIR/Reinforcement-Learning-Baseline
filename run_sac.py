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

import wandb
from tqdm import tqdm

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
    
"""
alpha auto tuning
"""
class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.init_value = init_value
        self.constant = nn.Parameter(
            torch.tensor(self.init_value, requires_grad=True)
        )

    def forward(self):
        return self.constant


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
        self.lr_end = config['lr_end']
        self.actor_lr_end = config['actor_lr_end']
        self.actor_update_interval = config['actor_update_interval']
        self.grad_norm = config['grad_norm']

        self.updates_per_training = config['train_steps']
        self.actor_updates_num = int(self.updates_per_training / self.actor_update_interval)

        self.adaptive_alpha = config['adaptive_alpha']  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = Scalar(init_value=config['alpha_init']).to(self.device)
            self.alpha = self.log_alpha().exp()
            self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=self.actor_lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.hidden_width).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=self.actor_updates_num,
            eta_min=self.actor_lr_end
        )
        self.critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            T_max=self.updates_per_training,
            eta_min=self.lr_end
        )

        self.num_update = 0

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.cpu().data.numpy().flatten()

    def update_critic(self, batch_s, batch_a, batch_r, batch_s_, batch_dw, result):
        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        result.update({
            'critic_loss': critic_loss,
            'current_q': current_Q1.detach().mean(),
            'target_q': target_Q.detach().mean(),
            'r': batch_r.detach().mean(),
        })

        return result
    
    def update_actor(self, batch_s, result):

        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Compute alpha loss
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha().exp() * (log_pi + self.target_entropy).detach()).mean()

        result.update({
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.log_alpha().exp().detach().mean(),
        })

        return result

    def learn(self, batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_c, batch_cv, result={}):
        # Compute critic loss
        result = self.update_critic(batch_s, batch_a, batch_r, batch_s_, batch_dw, result)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        result['critic_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm)
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()

        if self.num_update % self.actor_update_interval == 0:
            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            result = self.update_actor(batch_s, result)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            result['actor_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm)
            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()

            # Optimize the alpha
            if self.adaptive_alpha:
                self.alpha_optimizer.zero_grad()
                result['alpha_loss'].backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha().exp()

            # Unfreeze critic networks
            for params in self.critic.parameters():
                params.requires_grad = True

        self.num_update += 1

        return result
    
    def save(self, filedir):
        torch.save(self.actor.state_dict(), os.path.join(filedir, 'policy_network.pth'))

        return 'policy_network.pth'

    def load(self, filedir):
        self.actor.load_state_dict(torch.load(os.path.join(filedir, 'policy_network.pth')))


def evaluate_policy(env, agent, env_name):
    times = 10  # Perform evaluations and calculate the average
    evaluate_reward = 0
    evaluate_cost = 0
    for iteration in tqdm(range(0, times), ncols=70, desc='Evaluation', initial=1, total=times, ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
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

    train_steps = config['max_train_steps'] - config['random_steps'] # the total train steps

    config.update({
        'env_name': env_name,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'max_episode_steps': max_episode_steps,
        'train_steps': train_steps,
    })

    
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
    result_logs = {}
    red_list = ['ep_r', 'ep_c']
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
                result = agent.learn(*replay_buffer.sample(config['batch_size']))

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % config['evaluate_freq'] == 0:

                logger.log('\n==========Start evaluation==========')
                evaluate_num += 1
                evaluate_reward, evaluate_cost = evaluate_policy(env_evaluate, agent, config['environment'])
                result.update({'ep_r': evaluate_reward, 'ep_c': evaluate_cost})
                for k, v in sorted(result.items()):
                    if k in red_list:
                        logger.log(f'- {k:15s}:{v:5.5f}', color='red')
                    else:
                        print(f'- {k:15s}:{v:5.5f}')

                if config['wandb']:
                    wandb.log(result)

                result_log = {'log': result, 'step': total_steps}
                result_logs[str(total_steps)] = result_log

                # Save results
                logger.save_result_logs(result_logs)

                # Save model
                dir = logger.get_dir()
                file_name = agent.save(dir)
                logger.log('Logging model to ' + file_name)
                logger.log('\n==========Start training==========')


            total_steps += 1

        logger.log('Complete one episode with {} steps'.format(total_steps), color='magenta')

    # Save model
    dir = logger.get_dir()
    file_name = agent.save(dir)
    logger.log('Logging model to ' + dir)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='SAC', type=str)
    parser.add_argument('--environment', default='safety_gym', type=str) # simple_sg or safety_gym
    # parser.add_argument('--environment', default='simple_sg', type=str)
    parser.add_argument('--robot', default='Point', type=str)
    parser.add_argument('--task', default='Goal1', type=str)
    parser.add_argument('--env_name', default='point', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--wandb', default=True, type=boolean)
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
    parser.add_argument('--lr_end', default=8e-5, type=float)
    parser.add_argument('--actor_lr', default=8e-5, type=float)
    parser.add_argument('--actor_lr_end', default=4e-5, type=float)
    parser.add_argument('--actor_update_interval', default=int(2), type=int)
    parser.add_argument('--grad_norm', default=5., type=float)
    parser.add_argument('--alpha_init', default=0., type=float)


    parser.add_argument('--goal_met_done', default=False, type=boolean)
    parser.add_argument('--violation_done', default=False, type=boolean)


    return parser

if __name__ == '__main__':
    args = get_parser().parse_args() 
    run(vars(args))
