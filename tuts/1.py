import tensorflow as tf

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)

with tf.Session() as sess:
    print("%f should equal 2" % sess.run(y, feed_dict={a:1, b:2}))
    print("%f should equal 10" % sess.run(y, feed_dict={a:5, b:2}))
