from game_env_small import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
# from kerasrl import build_agent, build_model

window_size = 3


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(window_size, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.,
                                  nb_steps=30000)
    memory = SequentialMemory(limit=30000, window_length=window_size)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                   )
    return dqn


env = GameEnv()
height, width, channels = env.observation_space.shape
actions = env.action_space.n

model_path = "models/"
model_name = "dqn9.hdf5"

# model_path = "models/"
# model_name = "dqn_winsize2_11.hdf5"
#
# model_path = "models/"
# model_name = "dqn_winsize4_6.hdf5"

num_episodes = 20

# Build models
model = build_model(height, width, channels, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-4))

# Load and test
dqn.load_weights(model_path + model_name)
scores = dqn.test(env, nb_episodes=num_episodes, visualize=False)
print(np.mean(scores.history['episode_reward']))
print(np.std(scores.history['episode_reward']))