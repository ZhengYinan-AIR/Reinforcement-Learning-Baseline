import h5py
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, size, device):
        self.size = size
        self.s = np.zeros((size, state_dim))
        self.a = np.zeros((size, action_dim))
        self.a_logprob = np.zeros((size, action_dim))
        self.r = np.zeros((size, 1))
        self.s_ = np.zeros((size, state_dim))
        self.dw = np.zeros((size, 1))
        self.done = np.zeros((size, 1))
        self.count = 0

        self._device = device

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
        s = torch.tensor(self.s, dtype=torch.float).to(self._device)
        a = torch.tensor(self.a, dtype=torch.float).to(self._device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self._device)
        r = torch.tensor(self.r, dtype=torch.float).to(self._device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self._device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self._device)
        done = torch.tensor(self.done, dtype=torch.float).to(self._device)

        return s, a, a_logprob, r, s_, dw, done