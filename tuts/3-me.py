import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, b):
    return tf.add(tf.matmul(X, W), b)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

W = init_weights([784, 10])
b = init_weights([1, 10])

y_model = model(X, W, b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_model, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
predict_op = tf.argmax(y_model, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(50):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y:trY[start:end]})
        print 'Iteration', i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}))
