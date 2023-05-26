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
        self._tau = config['tau']
        self._hidden_sizes = config['hidden_sizes']
        self._batch_size = config['batch_size']
        self._lr = config['lr']
        self._actor_lr = config['actor_lr']
        self._grad_norm_clip = config['grad_norm_clip']

        self._device = config['device']

        self._critic_network = V_critic(observation_spec, action_spec, self._hidden_sizes, self._device, use_tanh=True).to(self._device)
        self._critic_optimizer = torch.optim.Adam(self._critic_network.parameters(), self._lr)
        # policy-network
        self._policy_network = Actor(observation_spec, action_spec, self._hidden_sizes, self._device, use_tanh=True).to(self._device)
        self._policy_optimizer = torch.optim.Adam(self._policy_network.parameters(), self._actor_lr)

        self._num_training = 1

    def update(self, replay_buffer):
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
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self._gamma * self._lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            
            # advantage normalization
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self._K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(replay_buffer.size)), self._batch_size, False):
                dist_now = self._policy_network.get_distribution(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self._epsilon, 1 + self._epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self._entropy_coef * dist_entropy  # Trick 5: policy entropy

                # Update actor
                self._policy_optimizer.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), 0.5)
                self._policy_optimizer.step()

                v_s = self._critic_network(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self._critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._critic_network.parameters(), 0.5)
                self._critic_optimizer.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)


        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self._buffer.rewards), reversed(self.buffer._is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self._gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self._device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self._buffer.states, dim=0)).detach().to(self._device)
        old_actions = torch.squeeze(torch.stack(self._buffer.actions, dim=0)).detach().to(self._device)
        old_logprobs = torch.squeeze(torch.stack(self._buffer.logprobs, dim=0)).detach().to(self._device)
        old_state_values = torch.squeeze(torch.stack(self._buffer.state_values, dim=0)).detach().to(self._device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self._K_epochs):

            # Evaluating old actions and values
            logprobs = self._policy_network.get_log_density(old_states, old_actions)
            state_values = self._critic_network(old_states)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss = -torch.min(surr1, surr2)
            critic_loss = F.mse_loss(state_values, rewards)

            self._optimizers['policy'].zero_grad()
            policy_loss.backward()
            self._optimizers['policy'].step()

            self._optimizers['critic'].zero_grad()
            critic_loss.backward()
            self._optimizers['critic'].step()

        # Copy new weights into old policy
        self._old_policy_network.load_state_dict(self._policy_network.state_dict())

        result.update({
            'policy_loss': policy_loss,
            'critic_loss': critic_loss,
            'state_values': state_values.detach().mean(),
        })
        return result

    def step(self, o):
        o = torch.from_numpy(o).float().to(self._device)
        action = self._policy_network.get_action(o)

        return action.detach().cpu().numpy()

    def save(self, filedir):
        modeldir = os.path.join(filedir, 'model')
        os.makedirs(modeldir)

        torch.save(self._policy_network.state_dict(), os.path.join(modeldir, 'policy_network.pth'))
        # torch.save(self._optimizers['policy'].state_dict(), os.path.join(modeldir, 'policy_network_optimizer.pth'))

    def load(self, filedir):
        modeldir = os.path.join(filedir, 'model')
        self._policy_network.load_state_dict(torch.load(os.path.join(modeldir, 'policy_network.pth')))
        self._optimizers['policy'] = torch.optim.Adam(self._policy_network.parameters(), self._lr)