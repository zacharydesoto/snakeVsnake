import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, in_states, out_actions, device, using_cnn=True):
        super().__init__()
        self.device = device
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )

        self.layer_stack = nn.Sequential(
            nn.Flatten(), # Flatten inputs to a single vector
            nn.Linear(in_features=10*23*23, out_features=25) # Change in_features 
        )

        self.linear = nn.Linear(in_features=25+8, out_features=out_actions)

    def forward(self, x):
        blind_inp = x[0]
        conv_inp = x[1].unsqueeze(0).unsqueeze(0)

        grid_features = self.conv_block1(conv_inp)
        # grid_features = self.conv_block2(grid_features)
        grid_features = self.layer_stack(grid_features).squeeze(0)

        combined_features = torch.cat((blind_inp, grid_features), dim=0)
        out = self.linear(combined_features)
        return out


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

        if type(dones) != list:  # Training on a single experience
            # Need to unsqueeze to ensure consistent dimensions with batch optimization
            state = [state]
            next_state = [next_state]
            action = [action]
            reward = [reward]
            dones = [dones]

        pred = torch.zeros(len(dones), 4)
        for i in range(len(dones)):
            pred[i] = self.model(state[i])
        
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