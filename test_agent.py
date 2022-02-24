from dqn_agent import DQNAgent
from pipeline import GameState
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plots a single traning reward curve for a specific state space representation
def plot_single_training_reward_curve(model):
    df = pd.read_csv("../rewards/rewards_{}.csv".format(model))
    df[['reward', 'average reward', '10 episode average reward']].plot()
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.title('{} reward training curve'.format(model))
    plt.legend()
    plt.show()

# plots a single traning score curve for a specific state space representation
def plot_single_training_score_curve(model):
    df = pd.read_csv("../rewards/rewards_{}.csv".format(model))
    df[['score', 'average score', '10 episode average score']].plot()
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.title('{} score training curve'.format(model))
    plt.legend()
    plt.show()

# plots the 10 episode testing scores of each state space representation
def plot_testing_scores(test_results, episodes):
    x = [i + 1 for i in range(episodes)]
    for key in test_results:
        plt.plot(x, test_results[key][1], label=key)
        plt.ylabel('score')
        plt.xlabel('episode')
        plt.legend()
    plt.show()

# plots the 10 episode testing rewards of each state space representation
def plot_testing_rewards(test_results, episodes):
    x = [i + 1 for i in range(episodes)]
    for key in test_results:
        plt.plot(x, test_results[key][0], label=key)
        plt.ylabel('cumulative reward')
        plt.xlabel('episode')
        plt.legend()
    plt.show()

if __name__ == '__main__':
    params = dict()
    params['name'] = None
    params['epsilon'] = 0
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = 0
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    names = ['model1', 'model2', 'model3']
    testing_results = {}

    for name in names:
        params['name'] = name
        env = GameState()
        agent = DQNAgent(env, params)
        testing_results['model'] = agent.test(episodes=10)

    plot_testing_scores(testing_results, 10)

