from game import game
from agent import Agent, train, test

def main():
    config = {
        'GRID_WIDTH': 20,
        'GRID_HEIGHT': 20
    }
    # train(config, 'local_vision_model.pt', best_rewards=0, episodes=450)
    # ai_vs_ai('local_vision_model.pt')
    player_vs_ai('local_vision_model.pt')
    # test_agent_against_baseline(config, 'local_vision_model.pt')


def player_vs_ai(path):
    '''Lets the player play against the model saved to path using the arrow keys'''
    agent = Agent(path)

    game(snake1_model=agent)

def ai_vs_ai(path, ticks_per_s=30):
    '''Makes two agents using the model saved to path play against each other'''
    agent = Agent(path)

    game(ticks_per_s, snake1_model=agent, snake2_model=agent)

def test_agent_against_baseline(config, path, episodes=100):
    '''Has an agent using the model saved to path play against the baseline model'''
    test(config, path, baseline=True, episodes=episodes)

main()
