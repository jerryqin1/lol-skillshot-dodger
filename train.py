from time import sleep, time

from pipeline import GameState

#!/usr/bin/env python
#from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import random
import numpy as np
from collections import deque


# seed 1001 was the maximal performance seed.
testing = False
seed = 1001
np.random.seed(seed)
random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# if you are running this on Google Colab (e.g., using Google Drive), enable to True.
drive = False
google_drive_colab_path = '/content/drive/My Drive/flappy/' if drive == True else ''

OBSERVE = 1000 # timestpes to init the replay memory.
EXPLORE = 1000000 # frames over which to decay epsilon

FINAL_EPSILON = 0.0001 # final value
INITIAL_EPSILON = 0.4 # starting value

REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

GAME = 'skillshotdodger' # the name of the game being played for log files
ACTIONS = 9 # number of valid actions
GAMMA = 0.99 # decay rate of past observations


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
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

def trainNetwork(s, readout, _, sess):
    counter = 0
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])

    # readout stores the output of the network.
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))

    # the optimizer is declared.
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state
    game_state = GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    # preprocess the image to 80x80x4 and get the image state.
    x_t, _, terminal, _ = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")

    # are we testing or training? the decision is made here.
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training

    # data structures meant for logging
    epsilon = INITIAL_EPSILON
    t = 0
    score = []
    net_score = []
    net_flaps = []
    flaps = []

    # we continue to execute forever, until the game ends.
    while True:

        # get all the actions from the netwokr
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]

        a_t = np.zeros([ACTIONS])
        action_index = 0

        # if we're testing we dont need to follow an epsilon greedy policy.
        # just get the highest action value.
        if testing:
            if counter > 10:
                print("Testing Done")
                return
            if t % FRAME_PER_ACTION == 0:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
            else:
                a_t[0] = 1
        else:
            # otherwise, we should select randomly at times. (Defined by epsilon)
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("Random Action Selected Via Epsilon Greedy")
                    action_index = random.randrange(ACTIONS)
                    a_t[random.randrange(ACTIONS)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1
            else:
                a_t[0] = 1 # do nothing

        # downscale the value of the epsilon.
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal, cur_score = game_state.frame_step(a_t)
        flaps.append(cur_score)

        # process the image to 80x80x4 to preparer to feed into the network.
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        score.append(r_t)

        # we store memory via the replay memory.
        if testing == False:

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))

            # popping when above the memory.
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # only train if done observing (We've sufficiently filled the replay memory)
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []

                readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})

                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch}
                )

        # update the old values
        s_t = s_t1
        t += 1

        if testing == False:
            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, google_drive_colab_path + 'saved_networks_v1/' + GAME + '-dqn', global_step = t)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            else:
                state = "train"

        if terminal:
            net_score.append(sum(score))
            net_flaps.append(max(flaps))

        if terminal and testing:
            counter = counter + 1
            print("TIMESTEP,", t, "Reward,", sum(score), "Average Reward,", np.mean(net_score), "Flaps,", max(flaps))
            score = []
            flaps = []

        if terminal and testing == False:
            string = "GameOver TIMESTEP: " + str(t) + ", STATE: " + str(state) + ", EPSILON: " + str(epsilon) + ", ACTION: " + str(action_index) + ", REWARD: " + str(r_t) + ", Q_MAX: %e" % np.max(readout_t) + ", Episode Reward: " + str(sum(score)) +  ", Average Reward: " + str(np.mean(net_score)) + ", Standard Deviation Of Score: " + str(np.std(net_score))
            print(string)
            print("Game Over")
            with open(google_drive_colab_path + "net_score_cache_v1_game_over.txt", 'a') as f:
                f.write(string + "\n")
                f.close()
            score = []
            flaps = []
            sleep(0.01)
            if t > OBSERVE:
                sleep(5)
            game_state = GameState()

        if terminal == False and testing == False:
            string = "TIMESTEP: " + str(t) + ", STATE: " + str(state) + ", EPSILON: " + str(epsilon) + ", ACTION: " + str(action_index) + ", REWARD: " + str(r_t) + ", Q_MAX: %e" % np.max(readout_t) + ", Episode Reward: " + str(sum(score)) +  ", Average Reward: " + str(np.mean(net_score)) + ", Standard Deviation Of Score: " + str(np.std(net_score))
            print(string)


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    input_layer, readout, hidden_fully_connected_1 = createNetwork()
    trainNetwork(input_layer, readout, hidden_fully_connected_1, sess)
