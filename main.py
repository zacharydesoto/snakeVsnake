from game import *
from dqn import *
from environment import SnakeEnvironment
from agent import Agent, train

def main():
    game()
    # player_vs_ai('new_model1.pt')
    # ai_vs_ai('final_blind_model.pt')

    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    # train(config, 'final_blind_model.pt', best_rewards=0)

def player_vs_ai(path):
    agent = Agent(path)

    game(snake1_model=agent)

def ai_vs_ai(path, ticks_per_s=30):
    agent1 = Agent(path)
    agent2 = Agent(path)

    game(ticks_per_s, snake1_model=agent1, snake2_model=agent2)

main()
