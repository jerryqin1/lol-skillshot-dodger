### Todo: 1. figure out why we're doing actions after dying before printing out episodic results
###       2. look into frame processing and whether its deleting too much info
###       3. reward function
###       4. difficulty scaling
###       5. stable baselines
###       6. update game.py code so we can have human play
###       7. training on normal background with ezreal sprite

from pipeline import GameState

# !/usr/bin/env python
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import csv
import random
import numpy as np
from collections import deque

# uncomment for non-render mode
# os.environ["SDL_VIDEODRIVER"] = "dummy"

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# if you are running this on Google Colab (e.g., using Google Drive), enable to True.
drive = False

OBSERVE = 1000  # time steps to init the replay memory.
EXPLORE = 1000000  # frames over which to decay epsilon

FINAL_EPSILON = 0.0001  # final value
INITIAL_EPSILON = 0.45  # starting value

REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

GAME = 'skillshotdodger'  # the name of the game being played for log files
ACTIONS = 9  # number of valid actions
GAMMA = 0.99  # decay rate of past observations

FRAME_LIMIT = 200000


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def create_network():
    # network weights
    weight_conv_1 = weight_variable([8, 8, 4, 32])
    bias_conv_1 = bias_variable([32])

    weight_conv_2 = weight_variable([4, 4, 32, 64])
    bias_conv_2 = bias_variable([64])

    weight_conv_3 = weight_variable([3, 3, 64, 64])
    bias_conv_3 = bias_variable([64])

    weight_fully_1 = weight_variable([1600, 512])
    bias_fully_1 = bias_variable([512])

    weight_fully_2 = weight_variable([512, ACTIONS])
    bias_fully_2 = bias_variable([ACTIONS])

    # input layer
    input_layer = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    hidden_conv_1 = tf.nn.relu(conv2d(input_layer, weight_conv_1, 4) + bias_conv_1)
    pool_1 = max_pool_2x2(hidden_conv_1)

    hidden_conv_2 = tf.nn.relu(conv2d(pool_1, weight_conv_2, 2) + bias_conv_2)
    hidden_conv_3 = tf.nn.relu(conv2d(hidden_conv_2, weight_conv_3, 1) + bias_conv_3)

    hidden_conv_3_flat = tf.reshape(hidden_conv_3, [-1, 1600])

    hidden_fully_connected_1 = tf.nn.relu(tf.matmul(hidden_conv_3_flat, weight_fully_1) + bias_fully_1)

    readout = tf.matmul(hidden_fully_connected_1, weight_fully_2) + bias_fully_2

    return input_layer, readout, hidden_fully_connected_1


def train_test(s, readout, _, sess, testing=False, episodes=20000):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])

    # readout stores the output of the network.
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))

    # the optimizer is declared.
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = GameState()

    # store the previous observations in replay memory
    d = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    # preprocess the image to 80x80x4 and get the image state.
    x_t, _, terminal, _ = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks_v1")

    # check for a saved model
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded weights:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # data structures meant for logging
    epsilon = INITIAL_EPSILON
    episode = 0
    t = 0
    score = []

    # rewards[0] = reward
    # rewards[1] = running average reward
    # rewards[2] = running average reward over past 100 episodes
    rewards = [[], [], []]
    readouts = []

    action_history = np.zeros(ACTIONS)

    # we continue to execute forever, until the game ends.
    print("STARTING EPISODE", episode)
    while episode < episodes:

        # get all the actions from the network
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        # print(readout_t)
        readouts.append(tuple(readout_t))

        a_t = np.zeros([ACTIONS])

        # choose an action
        if testing:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

            # if t % FRAME_PER_ACTION == 0:
            #     action_index = np.argmax(readout_t)
            #     a_t[action_index] = 1
            # else:
            #     a_t[0] = 1
        else:
            if random.random() <= epsilon:
                print("Time Step {}: Random Action Selected Via Epsilon Greedy".format(t))
                action_index = random.randint(0, ACTIONS - 1)
                a_t[action_index] = 1
            else:
                print("Time Step {}: Action Selected Via Max Q-Value".format(t))
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1

        action_history[np.argmax(a_t)] += 1

        # downscale the value of the epsilon.
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        r_t = 0
        x_t1_colored = 0
        for i in range(4):
            x_t1_colored, r_t_acc, terminal, cur_score = game_state.frame_step(a_t)
            r_t += r_t_acc
            if terminal:
                break

        # process the image to 80x80x4 to preparer to feed into the network.
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        # _, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        score.append(r_t)

        # we store memory via the replay memory.
        if not testing:
            # store the transition in d
            d.append((s_t, a_t, r_t, s_t1, terminal))

            # popping when above the memory.
            if len(d) > REPLAY_MEMORY:
                d.popleft()

            # only train if done observing (We've sufficiently filled the replay memory)
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(d, BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []

                readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})

                for i in range(0, len(minibatch)):
                    ended = minibatch[i][4]
                    # if terminal, only equals reward
                    if ended:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                # perform gradient step
                train_step.run(feed_dict={
                    y: y_batch,
                    a: a_batch,
                    s: s_j_batch}
                )

        # save network when training
        if not testing and t % 5000 == 0:
            saver.save(sess, 'saved_networks_v1/' + GAME + '-dqn', global_step=t)
            print("SAVED SUCCESSFULLY")

        if terminal:
            rewards[0].append(sum(score))
            rewards[1].append(np.mean(rewards[0]))
            rewards[2].append(np.mean(rewards[0][-100:]))

            if testing:
                print("TIMESTEP,", t, "Reward,", rewards[0][-1], "Average Reward,", rewards[1][-1])
            else:
                mode = "observe" if t <= OBSERVE else "train"
                log = "TIMESTEP: " + str(t) + ", STATE: " + str(mode) + ", EPSILON: " + str(
                    epsilon) + ", Episode Reward: " + str(rewards[0][-1]) + ", Average Reward: " + str(
                    rewards[1][-1]) + ", 100 Episode Average Reward: " + str(rewards[2][-1])
                print(log)

            print("Game Over")
            episode += 1

            if episode < episodes:
                print("STARTING EPISODE", episode)
            score = []

        s_t = s_t1
        t += 1

        if t > FRAME_LIMIT:
            print("FRAME LIMIT REACHED!")
            break

    if testing:
        # save_training_data('testing_reward_val', rewards)
        print('TESTING COMPLETE')
        print('ACTION DISTRIBUTION: ', action_history)
    else:
        saver.save(sess, 'saved_networks_v1/' + GAME + '-dqn', global_step=t)
        save_training_data('training_reward_val', rewards)
        save_q_values('training_q_values', readouts)
        print('TRAINING COMPLETE')
        print("SAVED SUCCESSFULLY")


def save_training_data(file_name, rewards):
    columns = ["reward", "average reward", "100 episode average reward"]
    data = zip(rewards[0], rewards[1], rewards[2])

    with open('rewards/{}.csv'.format(file_name), 'w', newline='') as csvfile:
        write = csv.writer(csvfile)
        write.writerow(columns)
        write.writerows(data)

    print('REWARD DATA SAVED SUCCESSFULLY!')


def save_q_values(file_name, q_values):
    columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    with open('q_values/{}.csv'.format(file_name), 'w', newline='') as csvfile:
        write = csv.writer(csvfile)
        write.writerow(columns)
        write.writerows(q_values)

    print('Q-VALUE DATA SAVED SUCCESSFULLY!')


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input_layer, readout, hidden_fully_connected_1 = create_network()
    train_test(input_layer, readout, hidden_fully_connected_1, sess, False, 1250)
