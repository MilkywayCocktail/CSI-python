import csi_loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time


def foo():
    for i in range(100000):
        print('\r\r{}\n{}'.format(i, i+10), end='')
        time.sleep(1)


def plot_this():
    img = np.random.randint(0, 10000, 64).reshape((8, 8))
    plt.imshow(img)
    plt.colorbar()
    plt.show()


def timestamp2time(timestamp):
    t = datetime.datetime.fromtimestamp(timestamp)
    print(t)
    print(t.strftime("%Y-%m-%d %H:%M:%S.%f"))


#timestamp2time(1683623879384.4 / 1e3)
#timestamp2time(1683623879351 / 1e3)

aa = np.array([11,41,51,41,11,45,14])
b = np.array([True, True, False, False, True, True, False])

foo = np.array([[0, 1, 2], [0, 1, 2]])

bar = np.array([[[1]], [[2]]])

print(np.hstack((foo, bar.squeeze(axis=1))))
