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


TRIALS = 10

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

    scores = dqn.test(env, nb_episodes=TRIALS, visualize=False)
    print("average model score over", TRIALS, " trials is: ", np.mean(scores.history['episode_reward']))
    plt.plot(np.arange(1, 11), scores.history["episode_reward"])
    plt.title("Episodic Model Reward")
    plt.show()


def humanPlay():
    env = GameEnv()
    # play.play(env)
    scores = env.playGame(TRIALS)
    print("average human score over", TRIALS, " trials is: ", np.mean(scores))
    plt.plot(np.arange(1, 11), scores)
    plt.title("Episodic Human Reward")
    plt.show()


if __name__ == "__main__":
    p1 = multiprocessing.Process(target=modelPlay)
    p2 = multiprocessing.Process(target=humanPlay)
    p1.start()
    p2.start()
