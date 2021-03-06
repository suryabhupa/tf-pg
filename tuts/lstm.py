import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.layers as layers

map_fn = tf.map_fn

###################################################################
# Toy task: Generate two numbers in base B; numbers are reversed. #
###################################################################

#####################
### GENERATE DATA ###
#####################

base = 10

# Debugging
def as_10(num, base):
  res = 0
  curr_base = base
  for i in num:
    res += curr_base * i
    curr_base *= base
  return res

def one_hot(a):
  if base != 2:
    b = np.zeros((a.size, 10))
    b[np.arange(a.size), a] = 1
    return b
  else:
    return a

# Converts base 10 number to base B (iteratively divides by base and takes modulo)
def as_base_summands(num, final_size):
    res = []
    for _ in range(final_size-1):
        res.append(num % base)
        num //= base
    res.append(0)
    return one_hot(np.asarray(res))

def as_base_sum(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % base)
        num //= base
    return one_hot(np.asarray(res))

# Generate single example for sum (Converge to 100% validation acc in 5 epochs!)
def generate_example_sum(num_l):
    a = random.randint(0, base**(num_l - 1) - 1)
    b = random.randint(0, base**(num_l - 1) - 1)
    res = a + b
    if False:
      print 'a', a
      print 'b', b
      print 'res', res
      print 'a1', as_base_summands(a, num_l)
      print 'b1', as_base_summands(b, num_l)
      print 'res1', as_base_sum(res, num_l)
      quit()
    return (as_base_summands(a, num_l), as_base_summands(b, num_l), as_base_sum(res, num_l))

# Generate single example for subtraction (Converge to 100% validation acc in 5 epochs!)
def generate_example_sub(num_l):
    a = random.randint(0, base**(num_l - 1) - 1)
    b = random.randint(0, base**(num_l - 1) - 1)
    a, b = max(a, b), min(a, b)
    res = a - b
    return (as_base(a, num_l), as_base(b, num_l), as_base(res, num_l))

# Generate single example for multiplication (Doesn't converge)
def generate_example_mul(num_l):
    a = random.randint(0, base**(num_l - 1) - 1)
    b = random.randint(0, base**(num_l - 1) - 1)
    res = a * b
    return (as_base(a, num_l), as_base(b, num_l), as_base(res, num_l))

# Generate single example of randomness (Doesn't converge)
def generate_example_random(num_l):
    a = random.randint(0, base**(num_l - 1) - 1)
    b = random.randint(0, base**(num_l - 1) - 1)
    res = random.randint(0, base**(num_l - 1) - 1)
    return (as_base(a, num_l), as_base(b, num_l), as_base(res, num_l))

# Generate full batch
def generate_batch(num_l, batch_size):
    x = np.empty((num_l, batch_size, 20))
    y = np.empty((num_l, batch_size, 10))

    for i in range(batch_size):
        a, b, r = generate_example_sum(num_l)
        x[:, i, 0:10] = a
        x[:, i, 10:20] = b
        y[:, i, 0:10] = r

    return x, y

#####################
## GRAPH DEFINTION ##
#####################

INPUT_SIZE      = 20
RNN_HIDDEN      = 100
OUTPUT_SIZE     = 10
TINY            = 1e-6
LEARNING_RATE   = 0.01
USE_LSTM = True

inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

if USE_LSTM:
    # state_is_tuple is set to True because it requires that the
    # hidden state and cell state are concatenated in a tuple
    # (largely an artifact from a previous impl of rnn_cell)
    cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
else:
    cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN)

batch_size = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from RNN output size to OUTPUT_SIZE
final_projection = lambda x : layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply project at every timestep?
predicted_outputs = map_fn(final_projection, rnn_outputs)

# compute elementwise binary cross entropy (for base 2)
# error = tf.reduce_mean(-(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY)))

# compute element cross entropy (for general bases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_outputs, labels=outputs))

# use Adam
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# find op for ACC
# acc_fn = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))
predict_op = tf.argmax(predicted_outputs, 2)

#####################
### Training Loop ###
#####################

NUM_BITS = 10
ITERS = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_l=NUM_BITS, batch_size=BATCH_SIZE)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(10):
        epoch_loss = 0
        for _ in range(ITERS):
            x, y = generate_batch(num_l=NUM_BITS, batch_size=BATCH_SIZE)
            e_loss, _, ps, os = sess.run([loss, train_fn, predicted_outputs, outputs], feed_dict={inputs: x, outputs: y})
            epoch_loss += e_loss
            # print ps[0]
            # print os[0]
        epoch_loss /= ITERS
        # valid_acc = sess.run(acc_fn, feed_dict={inputs: valid_x, outputs: valid_y})
        valid_acc = np.mean(np.argmax(valid_y, axis=2) == sess.run(predict_op, feed_dict={inputs: valid_x}))

        # print "Epoch %d, Training error: %.2f" % (epoch, epoch_error, 000) #valid_acc * 100.0)
        print "Epoch %d, Training loss: %.2f, Validation Acc: %.2f" % (epoch, epoch_loss, valid_acc * 100.0)
