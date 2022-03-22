from train import train_test
from train import createNetwork
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import random
import numpy as np
from collections import deque


testing = True
seed = 2
np.random.seed(seed)
random.seed(seed)
tf.compat.v1.set_random_seed(seed)

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

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input_layer, readout, hidden_fully_connected_1 = createNetwork()
    train_test(input_layer, readout, hidden_fully_connected_1, sess, testing, 10)

