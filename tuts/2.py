import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 2

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X, w, b):
    return tf.add(tf.multiply(X, w), b)

w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

y_model = model(X, w, b)
cost = tf.square(Y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        for (x, y) in zip(train_X, train_Y):
            _, loss_val = sess.run([train_op, cost], feed_dict={X: x, Y: y})
        print(loss_val)

    final_w = sess.run(w)
    final_b = sess.run(b)

    # graph = tf.get_default_graph()
    # for op in graph.get_operations():
        # print(op.name)

    plt.axis([-1, 1, -10, 10])
    plt.plot(train_X, train_Y, 'ro')
    plt.plot(train_X, train_X * final_w + final_b, 'r-')
    plt.show()
