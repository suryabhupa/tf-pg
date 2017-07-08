import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")

tf.reset_default_graph()

"""
Instead of using Bellman equation to update Q-values, we use a neural network
to predict Q values based on state space. We take a one-hot vector of size 16
and produce a distribution of size four over the four actions in the game.

We also don't update Q values directly; we update the network by backprop.
The loss function is an L2 loss over predicted Q-values and the target:

    L = |Q_target - Q|^2

We can compute Q_target because we know the Bellman equation holds true:

    Q(s, a) = R + y * (max_a Q(s', a'))

and we use this for supervision.
"""

# RL Hyperparameters
lr = 0.85
y = 0.99
num_eps = 2000
ep_len = 100

# Placeholders and weights
inputs = tf.placeholder(shape=[1,16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.05))
Q_out = tf.matmul(inputs, W)

next_Q = tf.placeholder(shape=[1,4], dtype=tf.float32)
cost = tf.reduce_sum(tf.square(Q_out - next_Q))

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
predict_op = tf.argmax(Q_out, 1)

j_list = [] #
r_list = [] # total rewards per episode

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(num_eps):
        s = env.reset()
        r_all = 0
        d = False
        j = 0
        while j < 99:
            pass # NOT DONE YET
