from game import *
from dqn import *
from environment import SnakeEnvironment

def main():
    # train_model()
    test_model()
    replay_game('test.pickle')

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device {device}')

    model = SnakeDQL(learning_rate=1e-4, discount_factor=0.95, network_sync_rate=50, replay_memory_size=100000, mini_batch_size=512, num_hidden_nodes=256, device=device)

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    env = SnakeEnvironment(config, save=True)

    model.train(10000, env, 'model8.pt', 'model8.pt')    

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device {device}')

    model = SnakeDQL(num_hidden_nodes=256, device=device)

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }

    env = SnakeEnvironment(config, save=True)
    model.test(1, env, 'model8.pt', 'test.pickle')

    # Model 9: 500 hidden nodes

main()
