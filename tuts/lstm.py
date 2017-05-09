import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.layers as layers

map_fn = tf.map_fn

base = 2

# Toy task: Generate two numbers in base B; numbers are reversed.

# Converts base 10 number to base B (iteratively divides by base and takes modulo)
def as_base(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % base)
        num //= base
    return res

# Generate single example
def generate_example(num_l):
    a = random.randint(0, base**(num_l - 1) - 1)
    b = random.randint(0, base**(num_l - 1) - 1)
    res = a + b
    return (as_base(a, num_l), as_base(b, num_l), as_base(res, num_l))

# Generate full batch
def generate_batch(num_l, batch_size):
    x = np.empty((num_l, batch_size, 2))
    y = np.empty((num_l, batch_size, 1))

    for i in range(batch_size):
        a, b, r = generate_examples(num_bits)
        x[:, i, 0] = a
        x[:, i, 1] = b
        y[:, i, 0] = r

    return x, y

INPUT_SIZE      = 2
RNN_HIDDEN      = 20
OUTPUT_SIZE     = 1
TINY            = 1e-6
LEARNING_RATE   = 0.01
USE_LSTM = True

inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

if USE_LSTM:
    # state_is_tuple is set to True because it requires that the
    # hidden state and cell state are concatenated in a tuple
    # (largely an artifact from a previous impl of rnn_cell)
    cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
else:
    cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN)

batch_size = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from RNN output size to OUTPUT_SIZE
final_projection = lambda x : layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply project at every timestep?
predicted_outputs = map_fn(final_projection, rnn_outputs)

# compute elementwise binary cross entropy (for base 2)
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# use Adam
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

acc_fn = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))


### Training Loop

NUM_BITS = 10
ITERS = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_l=NUM_BITS, batch_size=BATCH_SIZE)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(1000):
        epoch_error = 0
        for _ in range(ITERS):
            x, y = generate_batch(num_l=NUM_BITS, batch_size=BATCH_SIZE)
            epoch_error, _ += sess.run([error, train_fn], feed_dict={inputs: x, outputs: y})
        epoch_error /= ITERS
        valid_acc = sess.run(acc_fn, feed_dict={inputs: valid_x, outputs: valid_y})

        print "Epoch %d, Training error: %.2f, Validation Acc: %.2f" % (epoch, epoch_error, valid_acc * 100.0)

