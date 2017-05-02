import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data

def init_weights_xavier(shape, n_inputs=None, n_outputs=None):
    if n_inputs and n_outputs:
        stddev = math.sqrt(3.0 / n_inputs + n_outputs)
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    else:
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=1))

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, w_h)), p_keep_hidden)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w_h2)), p_keep_hidden)
    return tf.matmul(h2, w_o)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

w_h = init_weights([784, 500])
w_h2 = init_weights([500, 250])
w_o = init_weights([250, 10])

py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_input: 0.8, p_keep_hidden: 0.5})
        print "Iteration", i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, p_keep_input: 1.0, p_keep_hidden: 1.0}))

