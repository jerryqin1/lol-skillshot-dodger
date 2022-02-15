# from game import Game

import random
import tensorflow as tf
import csv
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time


class DQNAgent:
    """ Deep Q Network """

    def __init__(self, env, params):
        self.env = env
        self.action_space = self.env.action_space
        self.state_space = self.env.state_space
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.layer_sizes = params['layer_sizes']
        self.weightFile = "weights_{}".format(self.env.env_info['state_space'])
        self.dataFile = "rewards_{}".format(self.env.env_info['state_space'])
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()

    # create isntance of DQN
    def build_model(self):
        model = Sequential()
        for i in range(len(self.layer_sizes)):
            if i == 0:
                model.add(Dense(self.layer_sizes[i], input_shape=(self.state_space,), activation='relu'))
            else:
                model.add(Dense(self.layer_sizes[i], activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # adds to training batch
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # returns agent action
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    # experience replay
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # loads existing weights
    def loadWeights(self):
        self.model.load_weights('../weights/{}.h5'.format(self.weightFile))
        print("weights loaded successfully!")

    # saves weights
    def saveWeights(self):
        self.model.save_weights("weights/{}.h5".format(self.weightFile))
        print("weights saved successfully!")

    # saves trainng dta into csv file
    def saveTrainingData(self, dataFile, rewards):
        columns = ["reward", "average reward", "10 episode average reward", 'score', 'average score', '10 episode average score']
        data = zip(rewards[0], rewards[1], rewards[2], rewards[3], rewards[4], rewards[5])

        with open('rewards/{}.csv'.format(dataFile), 'w', newline='') as csvfile:
            write = csv.writer(csvfile)
            write.writerow(columns)
            write.writerows(data)

        print('data saved successfully!')

    # trains agent on environment
    def train(self, episodes=100):
        # list to keep track of the episodic rewards
        # rewards[0] = reward
        # rewards[1] = running average reward
        # rewards[2] = running average reward over past 10 episodes
        # rewards[3] = game score
        # rewards[4] = running average game score
        # rewards[5] = running average game score over 10 episodes
        rewards = [[], [], [], [], [], []]
        max_avg_reward = float('-inf')
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, (1, self.env.state_space))
            score = 0
            max_steps = 10000
            for i in range(max_steps):
                action = self.act(state)
                prev_state = state
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                next_state = np.reshape(next_state, (1, self.env.state_space))
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if self.batch_size > 1:
                    self.replay()
                if done:
                    print(f'final state before dying: {str(prev_state)}')
                    print(f'episode: {e + 1}/{episodes}, reward: {score}, score: {self.env.prev_total}')
                    break
            rewards[0].append(score)
            rewards[1].append(np.mean(rewards[0]))
            rewards[2].append(np.mean(rewards[0][-10:]))

            rewards[3].append(self.env.prev_total)
            rewards[4].append(np.mean(rewards[3]))
            rewards[5].append(np.mean(rewards[3][-10:]))

            if(rewards[2][-1] > max_avg_reward):
                self.saveWeights()
                max_avg_reward = rewards[2][-1]

        self.env.reset()
        self.saveWeights()
        self.saveTrainingData(self.dataFile, rewards)
        return rewards

    # test agent
    def test(self, episodes=10):
        self.loadWeights()
        # results[0] = reward
        # results[1] = game score
        # results[2] = maximum game score
        results = [[], [], []]
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, (1, self.env.state_space))
            score = 0
            max_steps = 10000
            for i in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                next_state = np.reshape(next_state, (1, self.env.state_space))
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print(f'episode: {e + 1}/{episodes}, reward: {score}, score: {self.env.prev_total}')
                    # print(f'episode: {e + 1}/{episodes}, score: {self.env.prev_total}')
                    break
            results[0].append(score)
            results[1].append(self.env.prev_total)
            results[2].append(self.env.maximum)

        self.env.reset()
        return results


