import ray
from ray import tune

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

def build_model(height, width, channels, actions, config):
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


def build_agent(model, actions, config):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(config['eps']), attr='eps', value_max=1., value_min=.1, value_test=0.,
                                  nb_steps=30000)
    memory = SequentialMemory(limit=30000, window_length=2)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                   )
    return dqn


def Trainable(config):
    env = config["env"]

    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(height, width, channels, actions, config)
    dqn = build_agent(model, actions, config)

    dqn.compile(Adam(learning_rate= config['lr']))

    models_dir = "models/"
    read_file = "dqn_winsize2_6.hdf5"
    save_file = "dqn_winsize2_7.hdf5"

    if os.path.exists(models_dir + read_file):
        dqn.load_weights(models_dir + read_file)
        print("Loaded existing model")
    else:
        print("Could not find model at filepath")

    # fitting with params
    h = dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)
    print('done training!')
    dqn.save_weights(models_dir + save_file, overwrite=True)


    #testing for output
    dqn.load_weights('models/' + read_file)
    scores = dqn.test(env, nb_episodes=TRIALS, visualize=False)
    #print("average model score over", TRIALS, " trials is: ", np.mean(scores.history['episode_reward']))
    #plt.plot(np.arange(1, 11), scores.history["episode_reward"])
    #plt.title("Episodic Model Reward")
    #plt.show()

    return np.mean(scores.history['episode_reward'])



if __name__ == "__main__":

    analysis = tune.run(
              Trainable(config),
              config = {"env": GameEnv(), "lr": tune.loguniform(1e-4, 1e-2),
                        "eps": tune.grid_search(([0.5, 0.25, 0.1, 0.01])) },
                        )

    print("best config: ", analysis.get_best_config(metric="score", mode="max"))

