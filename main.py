from game import *
from dqn import *
from environment import SnakeEnvironment

def main():
    train_model()
    test_model()
    replay_game('test.pickle')

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device {device}')

    model = SnakeDQL(learning_rate=1e-6, discount_factor=0.99, network_sync_rate=1000, replay_memory_size=10000, mini_batch_size=128, num_hidden_nodes=500, device=device)

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    env = SnakeEnvironment(config, save=True)

    model.train(1000, env, 'model9.pt', 'model9.pt')    

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device {device}')

    model = SnakeDQL(num_hidden_nodes=500, device=device)

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }

    env = SnakeEnvironment(config, save=True)
    model.test(1, env, 'model9.pt', 'test.pickle')

    # Model 9: 500 hidden nodes

main()
