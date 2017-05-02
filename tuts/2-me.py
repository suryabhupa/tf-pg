import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.5 - 4

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

def model(X, w, b):
    return tf.add(tf.multiply(X, w), b)

y_model = model(X, w, b)
cost = tf.square(Y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(50):
        for (x, y) in zip(train_X, train_Y):
            _, loss = sess.run([train_op, cost], feed_dict={X: x, Y: y})
        print 'LOSS', loss

    final_w = sess.run(w)
    final_b = sess.run(b)
    print(final_w)
    print(final_b)

    plt.plot(train_X, train_Y, 'ro')
    plt.plot(train_X, train_X * final_w + final_b, 'r-')
    plt.show()

