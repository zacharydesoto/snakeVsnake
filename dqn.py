import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt


class DQN(nn.Module):
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

    def train(self, episodes, env, policy_save_path, policy_load_path=None):
        '''Trains a policy DQN given an environment.
    
        Args:
            episodes (int): Number of episodes to train for.
            env (SnakeEnvironment): Environment for the agent to interact with.
            policy_save_path (string): File path to save the policy DQN to.
            policy_load_path (string): File path to load a previous policy DQN from.
        
        Returns:
            np array: Rewards earned each episode
        '''
        state = env.reset()
        num_input_params = env.get_network_state().shape[0]
        num_actions = len(env.actions)

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_input_params, self.num_hidden_nodes, num_actions)
        if policy_load_path != None:
            policy_dqn.load_state_dict(torch.load(policy_load_path))
        target_dqn = DQN(num_input_params, self.num_hidden_nodes, num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.AdamW(
            policy_dqn.parameters(), lr=self.learning_rate)
        rewards_per_episode_1 = np.zeros(episodes)
        rewards_per_episode_2 = np.zeros(episodes)

        step_count = 0

        for i in range(episodes):
            terminated, truncated = False, False
            episode_reward_1, episode_reward_2 = 0, 0

            while not terminated and not truncated:
                if random.random() < epsilon:  # Choose randomly
                    action1 = random.choice(env.actions)
                else:
                    with torch.no_grad():
                        action1 = policy_dqn(env.get_network_state(is_snake1=True)).argmax()  # Gets state for snake 1

                if random.random() < epsilon:
                    action2 = random.choice(env.actions)
                else:
                    with torch.no_grad():
                        action2 = policy_dqn(env.get_network_state(is_snake1=False)).argmax()  # Gets state for snake 2
                
                new_state, reward1, reward2, terminated, truncated = env.step(action1, action2)
                episode_reward_1 += reward1
                episode_reward_2 += reward2

                memory.append((state, action1, action2, new_state, reward1, reward2, terminated))

                state = new_state
                step_count += 1
            
            rewards_per_episode_1[i] = episode_reward_1
            rewards_per_episode_2[i] = episode_reward_2

            if i % 100 == 0:
                print(f'Epoch {i} Rewards: Snake 1: {episode_reward_1}, Snake 2: {episode_reward_2}')
                torch.save(policy_dqn.state_dict(), policy_save_path)
            
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = epsilon - 1/episodes

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0            

            plt.plot(rewards_per_episode_1)  # TODO: plotted for snake 1 only

            return rewards_per_episode_1, reward2
        
    def optimize(self, mini_batch, policy_dqn, target_dqn):  # FIXME: Make optimize function optimize on both snakes' rewards.
        # Could maybe do this by 
        '''Optimizes the policy DQN given a batch from the experience replay buffer.'''

        current_q_list = []
        target_q_list = []

        for state, action1, action2, new_state, reward1, reward2, terminated in mini_batch:
            if terminated:
                target = torch.tensor([reward1])
            else:
                target = torch.tensor(
                    reward1 + self.discount_factor * target_dqn(new_state).max()
                )

            loss = self.loss_fn(target, target_dqn(state))

            current_q = policy_dqn(state)
            current_q_list.append(current_q)

            target_q = target_dqn(state)
            target_q[action1] = target
            target_q_list.append(target_q)
        
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


