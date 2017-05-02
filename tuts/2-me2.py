import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X_train = np.linspace(-1, 1, 100)
Y_train = 2 * X_train + np.random.randn(*X_train.shape) * 0.33 + 2

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X, a, b, c):
    first = tf.multiply(tf.multiply(X, X), a)
    second = tf.multiply(X, b)
    third = c
    tmp = tf.add(first, second)
    return tf.add(tmp, third)

a = tf.Variable(0.0, name="a")
b = tf.Variable(2.0, name="b")
c = tf.Variable(2.0, name="c")

y_model = model(X, a, b, c)
cost = tf.square(Y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(50):
        for (x, y) in zip(X_train, Y_train):
            _, loss_val, a_tmp, b_tmp, c_tmp = sess.run([train_op, cost, a, b, c], feed_dict={X: x, Y: y})
        print "Iteration %d:" % i, 'Loss:', loss_val, 'A:', a_tmp, 'B:', b_tmp, 'C:', c_tmp

    final_a = sess.run(a)
    final_b = sess.run(b)
    final_c = sess.run(c)

    plt.axis([-1, 1, -10, 10])
    plt.plot(X_train, Y_train, 'ro')
    plt.plot(X_train, final_a * X_train * X_train + final_b * X_train + final_c, 'r-')
    plt.show()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print sess.run(a)
    print sess.run(b)
    print sess.run(c)
