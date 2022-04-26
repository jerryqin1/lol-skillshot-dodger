from kerasrl import build_agent, build_model
import os
from game_env import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
import multiprocessing
from gym.utils import play

def _seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		self.robot.np_random = self.np_random # use the same np_randomizer for robot as for env
		return [seed]

def modelPlay():
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

    # h = dqn.fit(env, nb_steps=600000, visualize=False, verbose=2)
    print('done training!')

    # show graph
    # dqn.save_weights(models_dir + save_file, overwrite=True)

    scores = dqn.test(env, nb_episodes=10, visualize=False)
    print(np.mean(scores.history['episode_reward']))


def humanPlay():
    env = GameEnv()
    # play.play(env)
    env.playGame(10)



if __name__ == "__main__":
    p1 = multiprocessing.Process(target=modelPlay)
    p2 = multiprocessing.Process(target=humanPlay)
    p1.start()
    p2.start()
