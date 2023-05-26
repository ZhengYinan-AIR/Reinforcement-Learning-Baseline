import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

MEAN_MIN = -7.24
MEAN_MAX = 7.25
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-6

# CQL
def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

class Actor(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        """
        Stochastic Actor network
        :param num_state: dimension of the state
        :param num_action: dimension of the action
        :param num_hidden: number of the units of hidden layer
        :param device: cuda or cpu
        """
        super(Actor, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

    def forward(self, x, repeat=None):
        if repeat is not None:
            x = extend_and_repeat(x, 1, repeat)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()


        logp_pi = a_distribution.log_prob(action).sum(axis=-1)

        logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)

        action = torch.tanh(action)
        return action, logp_pi

    def get_log_density(self, x, y):
        """
        calculate the log probability of the action conditioned on state
        :param x: state
        :param y: action
        :return: log(P(action|state))
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clamp(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)
        return logp_pi

    def get_action(self, x):
        """
        generate actions according to the state
        :param x: state
        :return: action
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action

    def get_deterministic_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        mu = torch.clamp(mu, MEAN_MIN, MEAN_MAX)
        mu = torch.tanh(mu)
        return mu

class Double_Critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, softplus=False):
        super(Double_Critic, self).__init__()
        self.device = device
        self.softplus = softplus

        # Q1 architecture
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(num_state + num_action, num_hidden)
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.fc6 = nn.Linear(num_hidden, 1)

    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float).to(self.device)
        x = torch.cat([obs, action], dim=-1)

        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        if self.softplus:
            q1 = F.softplus(self.fc3(q1))
        else:
            q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        if self.softplus:
            q2 = F.softplus(self.fc6(q2))
        else:
            q2 = self.fc6(q2)
        return q1, q2
    
    def q1(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float).to(self.device)   

        multiple_actions = False
        batch_size = obs.shape[0]
        if action.ndim == 3 and obs.ndim == 2:
            multiple_actions = True
            obs = extend_and_repeat(obs, 1, action.shape[1]).reshape(-1, obs.shape[-1])
            action = action.reshape(-1, action.shape[-1])

        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.softplus:
            q1 = F.softplus(self.fc3(x))
        else:
            q1 = self.fc3(x)
        if multiple_actions == True:
            q1 = q1.reshape(batch_size, -1)
        return q1

    def q2(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float).to(self.device)   

        multiple_actions = False
        batch_size = obs.shape[0]
        if action.ndim == 3 and obs.ndim == 2:
            multiple_actions = True
            obs = extend_and_repeat(obs, 1, action.shape[1]).reshape(-1, obs.shape[-1])
            action = action.reshape(-1, action.shape[-1])

        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        if self.softplus:
            q2 = F.softplus(self.fc6(x))
        else:
            q2 = self.fc6(x)
        if multiple_actions == True:
            q2 = q2.reshape(batch_size, -1)
        return q2


class V_critic(nn.Module):
    def __init__(self, num_state, num_hidden, device, softplus=False):
        super(V_critic, self).__init__()
        self.device = device
        self.softplus = softplus

        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.softplus:
            v = F.softplus(self.state_value(x))
        else:
            v = self.state_value(x)
        return v


class Q_critic(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, softplus=False):
        super(Q_critic, self).__init__()
        self.device = device
        self.softplus = softplus

        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float).to(self.device) 

        multiple_actions = False
        batch_size = obs.shape[0]
        if action.ndim == 3 and obs.ndim == 2:
            multiple_actions = True
            obs = extend_and_repeat(obs, 1, action.shape[1]).reshape(-1, obs.shape[-1])
            action = action.reshape(-1, action.shape[-1])

        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.softplus:
            q1 = F.softplus(self.state_value(x))
        else:
            q1 = self.state_value(x)
        if multiple_actions == True:
            q1 = q1.reshape(batch_size, -1)
        return q1
    
"""
Vanilla Variational Auto-Encoder 
"""
class VAE(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        super(VAE, self).__init__()
        self.device = device
        self.latent_dim = num_action * 2
        self.num_state = num_state
        self.hyper_hidden_dim = num_hidden
        self.num_action = num_action

        self.e1 = nn.Linear(num_state + num_action, num_hidden)
        self.e2 = nn.Linear(num_hidden, num_hidden)
        self.mean = nn.Linear(num_hidden, self.latent_dim)
        self.log_std = nn.Linear(num_hidden, self.latent_dim)
        
        self.d1 = nn.Linear(num_state + self.latent_dim, num_hidden)
        self.d2 = nn.Linear(num_hidden, num_hidden)
        self.d3 = nn.Linear(num_hidden, num_action)


    def forward(self, obs, action):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float).to(self.device) 
            
        z = F.relu(self.e1(torch.cat((obs, action), dim=-1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(obs, z)

        return u, mean, std
    
    def decode(self, obs, z=None):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            
        if z is None:
            z = torch.randn((obs.shape[0], obs.shape[1], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
        a = F.relu(self.d1(torch.cat((obs, z), dim=-1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))
    
"""
alpha auto tuning
"""
class Scalar(nn.Module):
    def __init__(self, init_value, device):
        super().__init__()
        self.device = device
        self.init_value = init_value
        self.constant = nn.Parameter(
            torch.tensor(self.init_value, requires_grad=True)
        )

    def forward(self):
        return self.constant