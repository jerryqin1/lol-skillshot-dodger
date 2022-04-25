import os

gpu = True
if gpu:
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\")
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\libnvvp\\")
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.5\\extras\\CUPTI\\lib64\\")
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\include\\")
    os.add_dll_directory("C:\\Program Files\\tools\\cuda1\\bin")

from game_env import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy



def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(2, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.,
                                  nb_steps=30000)
    memory = SequentialMemory(limit=30000, window_length=2)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                   )
    return dqn


if __name__ == "__main__":

    env = GameEnv()
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(height, width, channels, actions)
    dqn = build_agent(model, actions)

    dqn.compile(Adam(learning_rate=1e-4))

    models_dir = "models/"
    read_file = "dqn_winsize2_6.hdf5"
    save_file = "dqn_winsize2_7.hdf5"

    if os.path.exists(models_dir + read_file):
        dqn.load_weights(models_dir + read_file)
        print("Loaded existing model")
    else:
        print("Could not find model at filepath")

    h = dqn.fit(env, nb_steps=600000, visualize=False, verbose=2)
    print('done training!')

    # show graph
    # dqn.save_weights(models_dir + save_file, overwrite=True)

    scores = dqn.test(env, nb_episodes=100, visualize=False)
    print(np.mean(scores.history['episode_reward']))
