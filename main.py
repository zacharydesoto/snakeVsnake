from game import *
from dqn import *
from environment import SnakeEnvironment

def main():
    test_model()
    replay_game('test.pickle')
    # train_model()

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device {device}')

    model = SnakeDQL(learning_rate=1e-5, discount_factor=0.99, network_sync_rate=1000, replay_memory_size=1000, mini_batch_size=64, num_hidden_nodes=(1000, 500), device=device)

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    env = SnakeEnvironment(config)

    model.train(1000, env, 'model4.pt')    

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device {device}')

    model = SnakeDQL(learning_rate=1e-5, discount_factor=0.95, network_sync_rate=1000, replay_memory_size=1000, mini_batch_size=64, num_hidden_nodes=(1000, 500), device=device)

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }

    env = SnakeEnvironment(config, save=True)
    model.test(1, env, 'model4.pt', 'test.pickle')

main()