from game_env import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from kerasrl import build_agent, build_model

env = GameEnv()
height, width, channels = env.observation_space.shape
actions = env.action_space.n

model_path = "models/"
model_name = "dqn_winsize2_5.hdf5"

num_episodes = 1000

model = build_model(height, width, channels, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-4))

dqn.load_weights(model_path + model_name)
scores = dqn.test(env, nb_episodes=num_episodes, visualize=False)
print(np.mean(scores.history['episode_reward']))