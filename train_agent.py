from dqn_agent import DQNAgent
from pipeline import GameState

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    names = ['model1', 'model2', 'model3']
    results = dict()

    # train lol
    for name in names:
        params['name'] = name
        env = GameState()
        agent = DQNAgent(env, params)
        results['name'] = agent.train(episodes=150)