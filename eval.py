from game_env import GameEnv
from kerasrl import build_agent, build_model
import numpy as np
from tensorflow.keras.optimizers import Adam

env = GameEnv()
height, width, channels = env.observation_space.shape
actions = env.action_space.n

model_path = "big_models/"
model_name = "dqn_large_4.hdf5"
#
# model_path = "models/"
# model_name = "dqn_windsize4_6.hdf5"

num_episodes = 50

# Build model
model = build_model(height, width, channels, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-4))

# Load weights then test
dqn.load_weights(model_path + model_name)
scores = dqn.test(env, nb_episodes=num_episodes, visualize=False)
print(np.mean(scores.history['episode_reward']))
print(np.std(scores.history['episode_reward']))