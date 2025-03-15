from game import *
from dqn import *
from environment import SnakeEnvironment
from agent import Agent, train, test

def main():
    # player_vs_ai('local_vision_model.pt')

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    # train(config, 'local_vision_model.pt', best_rewards=0, episodes=450)
    # ai_vs_ai('local_vision_model.pt')
    get_agent_test(config, 'local_vision_model.pt', baseline=True)


def player_vs_ai(path):
    agent = Agent(path)

    game(snake1_model=agent)

def ai_vs_ai(path, ticks_per_s=30):
    agent = Agent(path)

    game(ticks_per_s, snake1_model=agent, snake2_model=agent)

def get_agent_test(config, path, baseline=False):
    test(config, path, baseline=baseline)

main()
