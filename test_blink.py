from train_blink import train_test
from train_blink import createNetwork
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
def plot_single_training_reward_curve():
    df = pd.read_csv("rewards/training_reward_val.csv")
    df[['reward', 'average reward', '100 episode average reward']].plot()
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.title('reward training curve')
    plt.legend()
    plt.show()

# plots the 10 episode testing rewards of each state space representation
def plot_testing_rewards():
    df = pd.read_csv("rewards/testing_reward_val.csv")
    df[['reward', 'average reward', '100 episode average reward']].plot()
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.title('reward testing curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input_layer, readout, hidden_fully_connected_1 = createNetwork()
    train_test(input_layer, readout, hidden_fully_connected_1, sess, testing, 10)
    plot_single_training_reward_curve()
    plot_testing_rewards()

