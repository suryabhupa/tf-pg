import gym
import numpy as np
import tensorflow as tf

env = gym.make("FrozenLake-v0")

"""
Implement Bellman equation:

    Q(s, a) = R + y * (max_a Q(s', a'))

with look up table.
"""

# Initialize lookup table of size |O| x |A|
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Parameters
lr = 0.85
y  = 0.99
num_eps = 2000
ep_len = 100
r_list = [] # Running total of rewards over time per episode

for i in range(num_eps):
    s = env.reset()
    r_all = 0
    d = False
    j = 0

    while j < ep_len:
        j += 1
        # Select action with max Q value in table with random noise
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n) * (1./(i+1)))

        # Get next action
        s1, r, d, _ = env.step(a)

        # Update lookup table
        Q[s, a] = Q[s, a] + lr * (r + y*np.max(Q[s1,:]) - Q[s,a])

        # Add reward to running list
        r_all += r
        s = s1

        if d == True:
            break

    print "Episode %d Mean Reward: %f" % (i, r_all/float(ep_len))
    r_list.append(r_all)

print "Score over time: %s" % str(sum(r_list)/num_eps)
print "Final Q-value Table"
print Q
