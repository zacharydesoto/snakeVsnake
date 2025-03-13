from game import *
from dqn import *
from environment import SnakeEnvironment
from agent import Agent, train

def main():
    # game()
    # player_vs_ai('test_model.pt')
    # agent1 = Agent('biggest_smartest_vision_model.pt', 28, (512, 1024), 3)
    # agent2 = Agent('test_model.pt', 7, 256, 3)
    # game(ticks_per_s=30, snake1_model=agent1, snake2_model=agent2)
    # ai_vs_ai(, 'final_blind_model.pt', 28, (512, 1024), 3)

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    train(config, 'final_blind_model.pt', 7, 256, 3, best_rewards=0, episodes=300)

def player_vs_ai(path):
    agent = Agent(path, 7, 256, 3)

    game(snake1_model=agent)

def ai_vs_ai(path1, path2, n_in, n_hidden, n_out, ticks_per_s=30):
    agent1 = Agent(path1, n_in, n_hidden, n_out)
    agent2 = Agent(path2, n_in, n_hidden, n_out)

    game(ticks_per_s, snake1_model=agent1, snake2_model=agent2)

main()
