import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions, device):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(in_states, h1_nodes)
        self.linear2 = nn.Linear(h1_nodes, out_actions)

        # self.fcs = nn.ModuleList()
        # prev_nodes = in_states
        # for h1_node in h1_nodes:
        #     self.fcs.append(nn.Linear(prev_nodes, h1_node))
        #     prev_nodes = h1_node
        # self.out = nn.Linear(prev_nodes, out_actions)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

        # for fc in self.fcs:
        #     x = F.relu(fc(x))
        # x = self.out(x)
        # return x


# Define memory for Exerience
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class SnakeDQL():
    def __init__(self, model, lr, discount_rate):
        self.lr = lr
        self.discount_rate = discount_rate
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.plot_losses_total = 0
        self.num_steps = 0

    def train_step(self, state, action, reward, next_state, dones):
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:  # Training on a single experience
            # Need to unsqueeze to ensure consistent dimensions with batch optimization
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            dones = (dones, )

        pred = self.model(state)

        target = pred.clone()
        for i in range(len(dones)):
            Q_new = reward[i] + self.discount_rate * torch.max(self.model(next_state[i])) * (1 - dones[i])
            target[i][action[i]] = Q_new

        # Optimize through PyTorch
        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        self.plot_losses_total += loss
        self.num_steps += 1
        loss.backward()

        self.optimizer.step()