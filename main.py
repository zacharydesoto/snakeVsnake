from game import two_player_game
from dqn import *
from environment import SnakeEnvironment

def main():
    train_model()

def train_model():
    model = SnakeDQL(learning_rate=1e-3, discount_factor=0.95, network_sync_rate=1000, replay_memory_size=1000, mini_batch_size=64, num_hidden_nodes=500)

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    env = SnakeEnvironment(config)

    model.train(100, env, 'model.pt')

main()