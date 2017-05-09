import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.layers as layers

map_fn = tf.map_fn

base = 10

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


