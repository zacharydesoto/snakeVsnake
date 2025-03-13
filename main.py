from game import *
from dqn import *
from environment import SnakeEnvironment
from agent import Agent, train

def main():
    game()
    # player_vs_ai('new_model1.pt')
    # ai_vs_ai('test_model.pt')

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    # train(config, 'test_model.pt', 7, 256, 3, best_rewards=0)

def player_vs_ai(path):
    agent = Agent(path, 8, 256, 3)

    game(snake1_model=agent)

def ai_vs_ai(path, ticks_per_s=30):
    agent1 = Agent(path, 7, 256, 3)
    agent2 = Agent(path, 7, 256, 3)

    game(ticks_per_s, snake1_model=agent1, snake2_model=agent2)

main()
