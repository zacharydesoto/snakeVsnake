import torch
import random
from collections import deque
from dqn import DQN, SnakeDQL
from environment import *
import os

class Agent:

    def __init__(self, path, device=torch.device('cpu')):
        self.n_games = 0
        self.epsilon = 0
        self.discount_rate = 0.9
        self.batch_size = 1024
        self.lr = 1e-3
        self.max_memory = 100_000
        self.memory = deque(maxlen=self.max_memory)
        self.model = DQN(8, 256, 4, device)
        if os.path.isfile(path):
            print(f'Loading DQN from {path}')
            self.model.load_state_dict(torch.load(path))
        self.trainer = SnakeDQL(self.model, lr=self.lr, discount_rate=self.discount_rate)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # Automatically pops left if max_memory is reached

    def train_long_memory(self):
        # Get batch from experience replay buffer
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        # Repackage data
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = torch.stack(states)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, train=True):
        # Decay epsilon linearly
        self.epsilon = max(0, 0.4 - self.n_games / 250)

        # Always choose the best move if not training
        if not train:
            self.epsilon = 0
        
        # Epsilon-greedy
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
        else:
            move = self.model(state).argmax().item()

        return move

    def get_state_dict(self):
        return self.model.state_dict()


def train(config, path, best_rewards=0):
    agent = Agent(path)
    env = SnakeEnvironment(config)
    total_rewards1, total_rewards2 = 0, 0
    while True:
        # Get old states
        state1 = env.get_network_state(is_snake1=True)
        state2 = env.get_network_state(is_snake1=False)

        # Get agents' actions
        action1 = agent.get_action(state1)
        action2 = agent.get_action(state2)

        # Perform actions and get new state
        new_state1, reward1, reward2, done, truncated = env.step(action1, action2)
        new_state2 = env.get_network_state(is_snake1=False)
        total_rewards1 += reward1
        total_rewards2 += reward2

        # Train based on experience
        agent.train_short_memory(state1, action1, reward1, new_state1, done)
        agent.train_short_memory(state2, action2, reward2, new_state2, done)

        # Save experience for replay later
        agent.remember(state1, action1, reward1, new_state1, done)
        agent.remember(state2, action2, reward2, new_state2, done)


        if done or truncated: # Episode over
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()
            print(f'Episode {agent.n_games}, Reward 1: {total_rewards1}, Reward 2: {total_rewards2}, Epsilon: {agent.epsilon}')
            if total_rewards1 + total_rewards2 > best_rewards or agent.n_games % 100 == 99:
                best_rewards = total_rewards1 + total_rewards2
                print('Saving new model')
                torch.save(agent.get_state_dict(), path)
            total_rewards1, total_rewards2 = 0, 0
