
"""
NOTES:

    The samples look pretty decent! Pixel-wise error kinda sucks and doesn't imduce any meaningful smoothness constraints, which is too bad, but nonetheless, the compression is pretty good. Would be interesting to take this to the next level to see how it owuld do for something like more complex images (probably not well) and for better optimizers/slightly more tuned hyperparameters!

    After 50 epochs, mean-squared error loss gets to 528k (!), but starts at 2M (!!!); raising learning rate from 0.01 to 0.05 to 0.1 made learning go faster, possible that even higher learning rate would be totally fine.

"""

# Autoencoder

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def vis(images, save_name):
    dim = images.shape[0]
    n_image_rows = int(np.ceil(np.sqrt(dim)))
    n_image_cols = int(np.ceil(dim * 1.0/n_image_rows))
    gs = gridspec.GridSpec(n_image_rows, n_image_cols, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)

    for g, count in zip(gs, range(int(dim))):
        ax = plt.subplot(g)
        ax.imshow(images[count,:].reshape((28, 28)))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(save_name + "_vis.png")

mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 500
corruption_level = 0.3
batch_size = 128

X = tf.placeholder("float", [None, n_visible], name='X')
mask = tf.placeholder("float", [None, n_visible], name='mask')

# Simple way to do Xavier/Glorot init (nice!)
W_init_max = 4*np.sqrt(6. / n_visible + n_hidden)
W_init = tf.random_uniform(shape=[n_visible, n_hidden], minval=-W_init_max, maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W) # easy way to tie weights together too
b_prime = tf.Variable(tf.zeros([n_visible]), name="b_prime") # why not tie the biases together too? not the right size.

def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X
    Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)
    return Z

Z = model(X, mask, W, b, W_prime, b_prime)

cost = tf.reduce_sum(tf.pow(X - Z, 2))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
predict_op = Z

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(50):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1-corruption_level, input_.shape)
            sess.run(train_op, feed_dict={X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1-corruption_level, teX.shape)
        print i, sess.run(cost, feed_dict={X: teX, mask: mask_np})

    mask_np = np.random.binomial(1, 1 - corruption_level, teX[:100].shape)
    predicted_imgs = sess.run(predict_op, feed_dict={X: teX[:100], mask: mask_np})
    input_imgs = teX[:100]

    vis(predicted_imgs, 'pred')
    vis(input_imgs, 'in')
    print "DONE"
