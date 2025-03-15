from game import *
from dqn import *
from environment import SnakeEnvironment
from agent import Agent, train, test

def main():
    # game()
    # player_vs_ai('new_model1.pt')

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    # train(config, 'cnn_combined_model.pt', best_rewards=0, episodes=450)
    # ai_vs_ai('cnn_combined_model.pt')
    get_agent_test(config, 'cnn_combined_model.pt')


def player_vs_ai(path):
    agent = Agent(path)

    game(snake1_model=agent)

def ai_vs_ai(path, ticks_per_s=30):
    agent1 = Agent(path)
    agent2 = Agent(path)

    game(ticks_per_s, snake1_model=agent1, snake2_model=agent2)

def get_agent_test(config, path):
    test(config, path)

main()
