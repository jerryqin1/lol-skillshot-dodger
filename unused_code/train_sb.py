import gym
from game_env_new_backup import GameEnv
from stable_baselines3 import DQN

env = GameEnv()

model = DQN('CnnPolicy', env, learning_starts=25000, verbose=1)
model.learn(total_timesteps=1000)

model.save('sb3/test_model')

# model = DQN.load('sb3/test_model')
#
# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()