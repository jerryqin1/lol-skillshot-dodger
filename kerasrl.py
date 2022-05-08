import os

gpu = False
if gpu:
    new_dir = os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\")
    new_dir.close()

    # Add dll files to run training on GPU
    # hi = os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\")
    # hi.close()
    # hi = os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\libnvvp\\")
    # hi.close()
    # hi = os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.5\\extras\\CUPTI\\lib64\\")
    # hi.close()
    # hi = os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\include\\")
    # hi.close()
    # hi = os.add_dll_directory("C:\\Program Files\\tools\\cuda1\\bin")
    # hi.close()

from game_env import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


window_size = 4


# Build model to process image/state and output action
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


# Build agent to process environment and output an action
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.,
                                  nb_steps=30000)
    memory = SequentialMemory(limit=30000, window_length=window_size)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                   )
    return dqn


if __name__ == "__main__":
    # Create game environment
    env = GameEnv()

    # Get dimensions
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    # Use dimensions to build model and agent
    model = build_model(height, width, channels, actions)
    dqn = build_agent(model, actions)

    # Compile with adam optimizer
    dqn.compile(Adam(learning_rate=1e-4))

    # Set directories and mode
    mode = "train"
    models_dir = "models/"
    read_file = "none.hdf5"
    save_file = "delete.hdf5"

    if mode == "train": # Training mode
        # Load weights file if it exists
        if os.path.exists(models_dir + read_file):
            dqn.load_weights(models_dir + read_file)
            print("Loaded existing model")
        else: # Start from scratch otherwise
            print("Could not find model at filepath")

        # Train for nb_steps number of timesteps
        h = dqn.fit(env, nb_steps=150000, visualize=False, verbose=2)
        print('done training!')

        print(model.summary())
        # Get episode reward history
        ep_reward = h.history['episode_reward']
        average_ep_reward = []

        # Compute average score over last 100 episodes in sliding window manner
        for i in range(len(ep_reward)):
            if i < 100:
                average_ep_reward.append(np.mean(ep_reward[:i + 1]))
            else:
                average_ep_reward.append(np.mean(ep_reward[i - 99:i + 1]))

        # Show plots
        plt.plot(ep_reward)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.savefig("ep_reward.png")
        plt.show()

        plt.plot(average_ep_reward)
        plt.xlabel("episode")
        plt.ylabel("average reward (last 100)")
        plt.savefig("average_ep_reward.png")
        plt.show()

        # Save weights
        dqn.save_weights('models/' + save_file)
    elif mode == "test": # Test model
        # Load and test model
        dqn.load_weights('models/' + read_file)
        scores = dqn.test(env, nb_episodes=10, visualize=False)
        print('Mean', np.mean(scores.history['episode_reward']))
        print('Std', np.std(scores.history['episode_reward']))