import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt


class DQN(nn.module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


# Define memory for Exerience
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)


class SnakeDQL():
    def __init__(self, learning_rate=1e-3, discount_factor=0.9,
                 network_sync_rate=10,
                 replay_memory_size=1000, mini_batch_size=32,
                 num_hidden_nodes=500):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.network_sync_rate = network_sync_rate
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.num_hidden_nodes = num_hidden_nodes
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def train(self, episodes, env):
        state = env.reset()
        num_input_params = len(state)
        num_actions = len(env.actions)

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policyDQN = DQN(num_input_params, self.num_hidden_nodes, num_actions)
        targetDQN = DQN(num_input_params, self.num_hidden_nodes, num_actions)

        targetDQN.load_state_dict(policyDQN.state_dict())

        self.optimizer = torch.optim.AdamW(
            policyDQN.parameters(), lr=self.learning_rate)
        rewards_per_episode = np.zeros(episodes)

        step_count = 0

        for i in range(episodes):
            terminated, truncated = False, False
            episode_reward = 0

            while not terminated and not truncated:
                if random.random < epsilon:  # Choose randomly
                    action = random.choice(env.actions)
                else:
                    with torch.no_grad():
                        action = env.actions[policyDQN(state).argmax()]
                
                new_state, reward, terminated, truncated = env.step(action)
                episode_reward += reward

                memory.append((state, action, new_state, reward, terminated))

                step_count += 1
            
            rewards_per_episode[i] = episode_reward

            if i % 100 == 0:
                print(f'Epoch {i} Rewards: {episode_reward}')
            
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policyDQN, targetDQN)

                # Decay epsilon
                epsilon = epsilon - 1/episodes

                if step_count > self.network_sync_rate:
                    targetDQN.load_state_dict(policyDQN.state_dict())
                    step_count = 0
            
            torch.save(policyDQN.state_dict(), 'model.pth')

            plt.plot(rewards_per_episode)
        


