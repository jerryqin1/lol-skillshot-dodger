from dqn_agent import DQNAgent
from pipeline import GameState
import numpy as np
import pygame as pg
from pipeline import WIN_HEIGHT, WIN_WIDTH


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
    # for name in names:
    #     params['name'] = name
    #     env = GameState()
    #     agent = DQNAgent(env, params)
    #     results['name'] = agent.train(episodes=150)

    pg.init()
    screen = pg.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    game = GameState()
    np.random.seed(6)
    score = 0

    for i in range(1000):
        action = np.random.randint(0, 9)
        dqn = DQNAgent(game, params)
        dqn.train(1000)
        frame, reward, term, _ = game.step(action)
        score += reward
        if term: break

    print('Score', score)