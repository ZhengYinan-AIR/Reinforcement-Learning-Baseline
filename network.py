import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

LOG_STD_MIN = -5
LOG_STD_MAX = 2

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
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
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
    
class Double_Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width, use_softplus=False):
        super(Double_Critic, self).__init__()
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
    
    def critic_num(self):
        return 2

class Scalar_Multiplier(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.init_value = init_value
        self.constant = nn.Parameter(
            torch.tensor(self.init_value, requires_grad=True)
        )

    def forward(self):
        return F.softplus(self.constant)
    
    def func_type(self):
        return 'scalar'

class Critic(nn.Module):  # single critic network
    def __init__(self, state_dim, action_dim, hidden_width, use_softplus=False):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

        self.use_softplus = use_softplus

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        if self.use_softplus:
            q1 = F.softplus(self.l3(q1))
        else:
            q1 = self.l3(q1)
        
        return q1
    
    def critic_num(self):
        return 1
    
class V_critic(nn.Module):
    def __init__(self, num_state, num_hidden):
        super(V_critic, self).__init__()
        
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.state_value(x)
        return v
    
class MLP_Multiplier(nn.Module):
    def __init__(self, state_dim, num_hidden, upper_bound):
        super(MLP_Multiplier, self).__init__()

        self.fc1 = nn.Linear(state_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

        self.upper_bound = upper_bound

    def forward(self, state):
        state = torch.tanh(self.fc1(state))
        state = torch.tanh(self.fc2(state))
        state = self.fc3(state)
        lam = self.upper_bound/2. * \
            (1. + torch.tanh(state/self.upper_bound*2 ))
        return lam
    
    def func_type(self):
        return 'mlp'
    