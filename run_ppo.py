import os
import torch
import numpy as np
import gym
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal

import utils
from utils import boolean

import wandb

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

def evaluate_policy(config, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if config['use_state_norm']:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if config['policy_dist'] == "Beta":
                action = 2 * (a - 0.5) * config['max_action']  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if config['use_state_norm']:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times

class ReplayBuffer:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.device = config['device']

        self.s = np.zeros((self.batch_size, self.state_dim))
        self.a = np.zeros((self.batch_size, self.action_dim))
        self.a_logprob = np.zeros((self.batch_size, self.action_dim))
        self.r = np.zeros((self.batch_size, 1))
        self.s_ = np.zeros((self.batch_size, self.state_dim))
        self.dw = np.zeros((self.batch_size, 1))
        self.done = np.zeros((self.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.a, dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return s, a, a_logprob, r, s_, dw, done

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, config):
        super(Actor_Beta, self).__init__()
        self.state_dim = config['state_dim']
        self.hidden_width = config['hidden_width']
        self.action_dim = config['action_dim']
        self.fc1 = nn.Linear(self.state_dim, self.hidden_width)
        self.fc2 = nn.Linear(self.hidden_width, self.hidden_width)
        self.alpha_layer = nn.Linear(self.hidden_width, self.action_dim)
        self.beta_layer = nn.Linear(self.hidden_width, self.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][config['use_tanh']]  # Trick10: use tanh

        if config['use_orthogonal_init']:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, config):
        super(Actor_Gaussian, self).__init__()
        self.state_dim = config['state_dim']
        self.hidden_width = config['hidden_width']
        self.action_dim = config['action_dim']
        self.max_action = config['max_action']
        self.fc1 = nn.Linear(self.state_dim, self.hidden_width)
        self.fc2 = nn.Linear(self.hidden_width, self.hidden_width)
        self.mean_layer = nn.Linear(self.hidden_width, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][config['use_tanh']]  # Trick10: use tanh

        if config['use_orthogonal_init']:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.state_dim = config['state_dim']
        self.hidden_width = config['hidden_width']
        self.fc1 = nn.Linear(self.state_dim, self.hidden_width)
        self.fc2 = nn.Linear(self.hidden_width, self.hidden_width)
        self.fc3 = nn.Linear(self.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][config['use_tanh']]  # Trick10: use tanh

        if config['use_orthogonal_init']:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous():
    def __init__(self, config):
        self.policy_dist = config['policy_dist']
        self.max_action = config['max_action']
        self.batch_size = config['batch_size']
        self.mini_batch_size = config['mini_batch_size']
        self.max_train_steps = config['max_train_steps']
        self.lr_a = config['lr_a']  # Learning rate of actor
        self.lr_c = config['lr_c']  # Learning rate of critic
        self.gamma = config['gamma'] # Discount factor
        self.lamda = config['lamda']  # GAE parameter
        self.epsilon = config['epsilon']  # PPO clip parameter
        self.K_epochs = config['K_epochs']  # PPO parameter
        self.entropy_coef = config['entropy_coef']  # Entropy coefficient
        self.set_adam_eps = config['set_adam_eps']
        self.use_grad_clip = config['use_grad_clip']
        self.use_lr_decay = config['use_lr_decay']
        self.use_adv_norm = config['use_adv_norm']

        self.device = config['device']

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(config).to(self.device)
        else:
            self.actor = Actor_Gaussian(config).to(self.device)
        self.critic = Critic(config).to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().cpu().numpy().flatten()
        else:
            a = self.actor(s).detach().cpu().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps):
        result = {}
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
        
        result.update({
            'actor_loss': actor_loss.mean(),
            'critic_loss': critic_loss,
            'v_s': v_s.detach().mean(),
            'v_target': v_target[index].detach().mean()
        })

        return result

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def save(self, filedir):
        torch.save(self.actor.state_dict(), os.path.join(filedir, 'policy_network.pth'))

            
def run(config):
    env_list = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    env_name = config['env_name']
    assert env_name in env_list, "Invalid Env"

    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment

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

    evaluate_num = 0  # Record the number of evaluations
    result = {}
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(config)
    agent = PPO_continuous(config)


    state_norm = Normalization(shape=state_dim)  # Trick 2:state normalization
    if config['use_reward_norm']:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif config['use_reward_scaling']:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=config['gamma'])

    while total_steps < config['max_train_steps']:
        s = env.reset()
        if config['use_state_norm']:
            s = state_norm(s)
        if config['use_reward_scaling']:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if config['policy_dist'] == "Beta":
                action = 2 * (a - 0.5) * config['max_action']  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, _ = env.step(action)

            if config['use_state_norm']:
                s_ = state_norm(s_)
            if config['use_reward_norm']:
                r = reward_norm(r)
            elif config['use_reward_scaling']:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == config['batch_size']:
                result = agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % config['evaluate_freq'] == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(config, env_evaluate, agent, state_norm)
                result.update({'ep_r': evaluate_reward})
                for k, v in sorted(result.items()):
                    print(f'- {k:23s}:{v:15.10f}')
                print(f'iteration={total_steps}')
                if config['wandb']:
                    wandb.log(result)

                # Save model
                dir = logger.get_dir()
                agent.save(dir)
                print('model saved')
    # Save model
    dir = logger.get_dir()
    agent.save(dir)
    print('model saved')



def get_parser():
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=boolean, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=boolean, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=boolean, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=boolean, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=boolean, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=boolean, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=boolean, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=boolean, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=boolean, default=True, help="Trick 10: tanh activation function")

    parser.add_argument('--algo', default='PPO', type=str)
    parser.add_argument('--env_name', default='Hopper-v2', type=str)
    parser.add_argument('--device', default='cuda:2', type=str)
    parser.add_argument('--wandb', default=True, type=boolean)
    parser.add_argument('--seed', default=0, type=int)

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args() 
    run(vars(args))
