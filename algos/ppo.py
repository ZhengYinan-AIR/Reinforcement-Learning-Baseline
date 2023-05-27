import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.network import Actor, V_critic
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class PPO(object):

    def __init__(self, observation_spec, action_spec, config):
        self._K_epochs = config['K_epochs']
        self._lamda = config['lamda']
        self._epsilon = config['epsilon']
        self._entropy_coef = config['entropy_coef']
        self._gamma = config['gamma']
        self._hidden_sizes = config['hidden_sizes']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self._actor_lr = config['actor_lr']

        self._max_train_steps = config['max_train_steps']

        self._use_adv_norm = config['use_adv_norm']
        self._use_grad_clip = config['use_grad_clip']
        self._use_lr_decay = config['use_lr_decay']
        self._device = config['device']

        self._critic_network = V_critic(observation_spec, action_spec, self._hidden_sizes, self._device, 
                                        use_tanh=config['use_tanh'], use_orthogonal=config['use_orthogonal_init']).to(self._device)
        self._policy_network = Actor(observation_spec, action_spec, self._hidden_sizes, self._device, 
                                     use_tanh=config['use_tanh'], use_orthogonal=config['use_orthogonal_init']).to(self._device)
        
        if config['set_adam_eps']:
            self._critic_optimizer = torch.optim.Adam(self._critic_network.parameters(), self._lr, eps=1e-5)
            self._policy_optimizer = torch.optim.Adam(self._policy_network.parameters(), self._actor_lr, eps=1e-5)
        else:
            self._critic_optimizer = torch.optim.Adam(self._critic_network.parameters(), self._lr)
            self._policy_optimizer = torch.optim.Adam(self._policy_network.parameters(), self._actor_lr)

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
            vs = self._critic_network(s)
            vs_ = self._critic_network(s_)
            deltas = r + self._gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self._gamma * self._lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self._device)
            v_target = adv + vs
            # advantage normalization
            if self._use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self._K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(replay_buffer.size)), self._batch_size, False):
                dist_now = self._policy_network.get_distribution(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = self._policy_network.get_log_density(s[index], a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self._epsilon, 1 + self._epsilon) * adv[index]
                actor_loss = torch.mean(-torch.min(surr1, surr2) - self._entropy_coef * dist_entropy)  # Trick 5: policy entropy

                # Update actor
                self._policy_optimizer.zero_grad()
                actor_loss.backward()
                if self._use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 0.5)
                self._policy_optimizer.step()

                v_s = self._critic_network(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self._critic_optimizer.zero_grad()
                critic_loss.backward()
                if self._use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self._critic_network.parameters(), 0.5)
                self._critic_optimizer.step()

        if self._use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        result.update({
            'policy_loss': actor_loss,
            'critic_loss': critic_loss,
            'state_values': v_s.detach().mean(),
            'state_target_values': v_target.detach().mean(),
            'reward': r.detach().mean()
        })
        return result
    
    def lr_decay(self, total_steps):
        actor_lr_now = self._actor_lr * (1 - total_steps / self._max_train_steps)
        lr_now = self._lr * (1 - total_steps / self._max_train_steps)
        for p in self._policy_optimizer.param_groups:
            p['lr'] = actor_lr_now
        for p in self._critic_optimizer.param_groups:
            p['lr'] = lr_now

    def step(self, o):
        o = torch.from_numpy(o).float().to(self._device)
        action = self._policy_network.get_action(o)

        return action.detach().cpu().numpy()
    
    def choose_action(self, o):
        o = torch.from_numpy(o).float().to(self._device)
        action, log_prob = self._policy_network(o)

        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def save(self, filedir):
        torch.save(self._policy_network.state_dict(), os.path.join(filedir, 'policy_network.pth'))
        # torch.save(self._optimizers['policy'].state_dict(), os.path.join(modeldir, 'policy_network_optimizer.pth'))

    def load(self, filedir):
        modeldir = filedir
        self._policy_network.load_state_dict(torch.load(os.path.join(modeldir, 'policy_network.pth')))
        # self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)