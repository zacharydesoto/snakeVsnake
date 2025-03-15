import torch
import random
from collections import deque
from dqn import DQN, SnakeDQL
from environment import *
import os
import matplotlib.pyplot as plt
from utils import check_out_bounds, check_collision

class Agent:

    def __init__(self, path, device=torch.device('cpu')):
        self.n_games = 0
        self.epsilon = 0
        self.discount_rate = 0.9
        self.batch_size = 1024
        self.lr = 1e-3
        self.max_memory = 100_000
        self.memory = deque(maxlen=self.max_memory)
        self.model = DQN(in_states=1, out_actions=4, device=device) # FIXME: does this need .to(device)
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
        state = state.squeeze(0)
        next_state = next_state.squeeze(0)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, train=True, env=None, get_baseline_action=False):
        if get_baseline_action:
            possible_actions = []
            for dir in [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN]:
                if not env.check_dies(direction=dir, is_snake1=False):
                    possible_actions.append(dir)
            
            if possible_actions:
                move = random.choice(possible_actions)
            else:
                move = 0
            print(f"Random move {move}")
            return move
        
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

        print(f'Actual move {move}')
        return move

    def get_state_dict(self):
        return self.model.state_dict()


def train(config, path, best_rewards=0, episodes=None):
    agent = Agent(path)
    env = SnakeEnvironment(config)
    total_rewards1, total_rewards2 = 0, 0
    plot_rewards_1, plot_rewards_2 = [], []
    avg_loss = 0
    plot_losses = []
    while True:
        if episodes and agent.n_games >= episodes:
            break

        # Get old states
        state1 = env.get_portion_grid(is_snake1=True) # (1, 3, 5, 5) to input in model 
        state2 = env.get_portion_grid(is_snake1=False)

        # Get agents' actions
        action1 = agent.get_action(state1)
        action2 = agent.get_action(state2)

        # Perform actions and get new state
        new_state1, reward1, reward2, done, truncated = env.step(action1, action2)
        new_state2 = env.get_portion_grid(is_snake1=False)
        total_rewards1 += reward1
        total_rewards2 += reward2

        new_state1, new_state2 = new_state1.squeeze(0), new_state2.squeeze(0)

        # Train based on experience
        agent.train_short_memory(state1, action1, reward1, new_state1, done)
        agent.train_short_memory(state2, action2, reward2, new_state2, done)

        # Save experience for replay later
        state1 = state1.squeeze(0)
        state2 = state2.squeeze(0)
        new_state1 = new_state1.squeeze(0)
        new_state2 = new_state2.squeeze(0)
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
            plot_rewards_1.append(total_rewards1)
            plot_rewards_2.append(total_rewards2)
            total_rewards1, total_rewards2 = 0, 0
            avg_loss = float(agent.trainer.plot_losses_total) / agent.trainer.num_steps
            plot_losses.append(avg_loss)
            avg_loss = 0
    
    plot_figures(agent.n_games, plot_rewards_1=plot_rewards_1, plot_rewards_2=plot_rewards_2, plot_losses=plot_losses)

def plot_figures(n_games, plot_rewards_1, plot_rewards_2, plot_losses):
    x_axis = range(1, n_games + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, plot_rewards_1, marker='o', linestyle='-', label='Snake 1')
    plt.plot(x_axis, plot_rewards_2, marker='o', linestyle='-', label='Snake 2')
    plt.xlabel("Number of Games")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig('new_rewards.png')

    plt.figure(figsize=(10,6))
    plt.plot(x_axis, plot_losses)
    plt.xlabel("Number of Games")
    plt.ylabel("Average Loss")
    plt.savefig('new_losses.png')

def test(config, path, baseline=False, episodes=100):
    agent = Agent(path)
    agent.n_games = 1000
    agent.model.eval()
    env = SnakeEnvironment(config)

    snake1_lengths = []
    snake2_lengths = []

    for i in range(episodes):
        print(f"Running episode {i}")
        while True:
            # Get old states
            state1 = env.get_portion_grid(is_snake1=True)
            state2 = env.get_portion_grid(is_snake1=False)

            # print(f'state1 shape {state1.shape}')
            # print(f'state2 shape {state2.shape}')

            # Get agents' actions
            action1 = agent.get_action(state1)
            # action2 = agent.get_action(state2)
            
            # For baseline (random) model
            action2 = agent.get_action(state2, env=env, get_baseline_action=True)

            # Perform actions and get new state
            _, _, _, done, truncated = env.step(action1, action2)

            if done or truncated: # Episode over
                snake1_lengths.append(env.snake1.length)
                snake2_lengths.append(env.snake2.length)

                env.reset()
                agent.n_games += 1
                break
    

    x_axis = range(1, episodes + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, snake1_lengths, marker='o', linestyle='-', label='Snake 1')
    plt.plot(x_axis, snake2_lengths, marker='o', linestyle='-', label='Snake 2')
    plt.xlabel("Number of Games")
    plt.ylabel("Lengths")
    plt.title('Local Vision Snake Lengths in Testing')
    plt.legend()
    plt.savefig('lengths.png')

    print(f'Snake 1 average length: {np.average(np.array(snake1_lengths))}')
    print(f'Snake 2 average length: {np.average(np.array(snake2_lengths))}')