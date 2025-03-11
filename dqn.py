import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions, device):
        super().__init__()

        self.device = device

        self.linear1 = nn.Linear(in_states, h1_nodes)
        self.linear2 = nn.Linear(h1_nodes, out_actions)
        self.linear3 = nn.Linear(h1_nodes, out_actions)

        # self.fcs = nn.ModuleList()
        # prev_nodes = in_states
        # for h1_node in h1_nodes:
        #     self.fcs.append(nn.Linear(prev_nodes, h1_node))
        #     prev_nodes = h1_node
        # self.out = nn.Linear(prev_nodes, out_actions)

    def forward(self, x):
        x = x.to(self.device)

        x = F.relu(self.linear1(x))
        q1 = self.linear2(x)
        q2 = self.linear3(x)
        return (q1, q2)

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
    def __init__(self, learning_rate=1e-3, discount_factor=0.9,
                 network_sync_rate=10,
                 replay_memory_size=1000, mini_batch_size=32,
                 num_hidden_nodes=(500), device='cpu'):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.network_sync_rate = network_sync_rate
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.num_hidden_nodes = num_hidden_nodes
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.device = device

        print('Snake DQL Initiated')

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
        print("Beginning Training")
        state = env.reset(2, 2).to(self.device)
        num_input_params = env.get_network_state().shape[0]
        num_actions = len(env.actions)

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_input_params, self.num_hidden_nodes, num_actions, self.device).to(self.device)
        if policy_load_path != None:
            print(f'Loading Policy DQN from {policy_load_path}')
            policy_dqn.load_state_dict(torch.load(policy_load_path))
        target_dqn = DQN(num_input_params, self.num_hidden_nodes, num_actions, self.device).to(self.device)
        policy_dqn.train()
        target_dqn.train()

        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.AdamW(policy_dqn.parameters(), lr=self.learning_rate)
        rewards_per_episode_1 = np.zeros(episodes)
        rewards_per_episode_2 = np.zeros(episodes)
        losses = []

        step_count = 0

        print("Training setup completed")

        for i in range(episodes):
            # print(f'Starting episode {i}')
            terminated, truncated = False, False
            episode_reward_1, episode_reward_2 = 0, 0

            while not terminated and not truncated:
                if random.random() < epsilon:  # Choose randomly
                    action1 = random.choice(range(len(env.actions)))
                    action1_dir = env.actions[action1]
                else:
                    with torch.no_grad():
                        q1, _ = policy_dqn(env.get_network_state(is_snake1=True).to(self.device))
                        action1 = q1.argmax().item()  # Gets state for snake 1

                        action1_dir = env.actions[action1]

                if random.random() < epsilon:
                    action2 = random.choice(range(len(env.actions)))
                    action2_dir = env.actions[action2]
                else:
                    with torch.no_grad():
                        _, q2 = policy_dqn(env.get_network_state(is_snake1=False).to(self.device))
                        action2 = q2.argmax().item()  # Gets state for snake 2
                        action2_dir = env.actions[action2]
                
                new_state, reward1, reward2, terminated, truncated = env.step(action1_dir, action2_dir)
                new_state = new_state.to(self.device)
                episode_reward_1 += reward1
                episode_reward_2 += reward2

                memory.append((state, action1, action2, new_state, reward1, reward2, terminated))

                state = new_state
                step_count += 1
            
            state = env.reset(2, 2).to(self.device)

            rewards_per_episode_1[i] = episode_reward_1
            rewards_per_episode_2[i] = episode_reward_2

            # Decay epsilon
            epsilon_min, epsilon_max = 0.1, 1.0
            decay_rate = 3e-4
            epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * i)

            if i % 50 == 0:
                print(f'Episode {i} Rewards: Snake 1: {episode_reward_1}, Snake 2: {episode_reward_2}')
                print(f'Epsilon: {epsilon}')
                torch.save(policy_dqn.state_dict(), policy_save_path)
            
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                loss = self.optimize(mini_batch, policy_dqn, target_dqn)
                losses.append(loss)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        plt.plot(losses)  # TODO: plotted for snake 1 only
        plt.savefig('loss.png')

        plt.plot(rewards_per_episode_1)
        plt.savefig('rewards.png')

        return rewards_per_episode_1, rewards_per_episode_2
        
    def optimize(self, mini_batch, policy_dqn, target_dqn):  # FIXME: Make optimize function optimize on both snakes' rewards.
        '''Optimizes the policy DQN given a batch from the experience replay buffer.'''

        states, actions1, actions2, new_states, rewards1, rewards2, dones = [], [], [], [], [], [], []
        for state, action1, action2, new_state, reward1, reward2, terminated in mini_batch:
            states.append(state)
            actions1.append(action1)
            actions2.append(action2)
            rewards1.append(reward1)
            rewards2.append(reward2)
            new_states.append(new_state)
            dones.append(terminated)
        
        # Convert lists to tensors (make sure they are on the correct device)
        states = torch.stack(states).to(self.device)
        new_states = torch.stack(new_states).to(self.device)
        actions1 = torch.tensor(actions1, dtype=torch.long).to(self.device)
        actions2 = torch.tensor(actions2, dtype=torch.long).to(self.device)
        rewards1 = torch.tensor(rewards1, dtype=torch.float).to(self.device)
        rewards2 = torch.tensor(rewards2, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        
        # Compute current Q values using the policy network
        current_q1, current_q2 = policy_dqn(states)  # batch_size x num_actions
        # print(current_q.shape)
        # Gather the Q-values for the taken actions
        current_q1 = current_q1.gather(1, actions1.unsqueeze(1)).squeeze(1)  # Selects the q value for the selected action for each experience in the batch
        current_q2 = current_q2.gather(1, actions2.unsqueeze(1)).squeeze(1)
        
        # Compute the next Q values from the target network
        next_q1, next_q2 = target_dqn(new_states)
        next_q1 = next_q1.max(1)[0]
        next_q2 = next_q2.max(1)[0]
        # For terminal states, we want the next Q value to be zero
        expected_q1 = rewards1 + self.discount_factor * next_q1 * (1 - dones)  # If terminated, dones[i] = 1, so expected_q = rewards[i]
        expected_q2 = rewards2 + self.discount_factor * next_q2 * (1 - dones)
        
        loss1 = self.loss_fn(current_q1, expected_q1)
        loss2 = self.loss_fn(current_q2, expected_q2)
        # FIXME: should total loss be sum of individual losses
        loss = loss1 + loss2
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def test(self, episodes, env, policy_load_path, save_game_path):
        num_input_params = env.get_network_state().shape[0]
        num_actions = len(env.actions)

        policy_dqn = DQN(num_input_params, self.num_hidden_nodes, num_actions, self.device).to(self.device)
        print(f'Loading Policy DQN from {policy_load_path}')
        policy_dqn.load_state_dict(torch.load(policy_load_path))
        policy_dqn.eval()

        for i in range(episodes):
            print(f'Starting episode {i}')
            state = env.reset(2, 2).to(self.device)
            terminated, truncated = False, False
            
            with torch.no_grad():
                while not terminated and not truncated:
                    q1, _ = policy_dqn(env.get_network_state(is_snake1=True).to(self.device))
                    action1 = q1.argmax().item()  # Gets state for snake 1
                    action1_dir = env.actions[action1]

                    _, q2 = policy_dqn(env.get_network_state(is_snake1=False).to(self.device))
                    action2 = q2.argmax().item()  # Gets state for snake 2
                    action2_dir = env.actions[action2]
                    
                    new_state, reward1, reward2, terminated, truncated = env.step(action1_dir, action2_dir)
                    new_state = new_state.to(self.device)
            
            env.save_game(save_game_path)
