import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt

class Bandit():
    def __init__(self, st=True, walk_length=0.1):
        self.st = st
        self.walk_length = walk_length
        self.mean = np.random.normal(0, 1)

class NSBandit(Bandit):
    def pull(self):
        val = np.random.normal(self.mean, 1)
        if np.random.uniform(0, 1) < 0.5:
            self.mean += self.walk_length
        else:
            self.mean -= self.walk_length
        return val

class SBandit(Bandit):
    def pull(self):
        return np.random.normal(self.mean, 1)

class BanditTestBed():
    def __init__(self, num, st=True, walk_length=0.1):
        self.num = num
        self.st = st
        if self.st:
            self.bandits = [SBandit(self.st)] * self.num
        else:
            self.bandits = [NSBandit(self.st, walk_length=0.1)] * self.num

    def get_optimal(self):
        pass

def main():
    num_iters = 10000
    num_bandits = 10

    sBTB = BanditTestBed(num_bandits, True)
    nsBTB = BanditTestBed(num_bandits, False)

    R_Ts = []

    for btb in [sBTB, nsBTB]:
        Q_a = np.zeros(btb.num)
        N_a = np.zeros(btb.num)

        print 'First:', Q_a

        for i in range(num_iters):
            if np.random.uniform(0, 1) < 0.1:
                a_t = np.random.randint(10)
            else:
                a_t = np.argmax(Q_a)

            R_t = btb.bandits[a_t].pull()
            R_Ts.append(R_t)
            print "R_T: %f, I: %d; a_t: %d" % (R_t, i, a_t)
            N_a[a_t] += 1.0
            # Q_a[a_t] += Q_a[a_t] + 1.0/(N_a[a_t]) * (R_t - Q_a[a_t])
            Q_a[a_t] += Q_a[a_t] + 0.01 * (R_t - Q_a[a_t])

        print 'Last:', Q_a
        plt.plot(range(num_iters), R_Ts)
        R_Ts = []

    plt.show()

if __name__ == '__main__':
    main()
