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

def labels(path):
    label = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i > 0:
                print(eval(line.split(',')[2][2:]))
                #label.append([eval(line.split(',')[0]), eval(line.split(',')[1])])


e = 2.173520803451538
l= [-1.64485363e+00, -1.08031934e+00, -7.38846849e-01, -4.67698799e-01,
 -2.27544977e-01, -1.39145821e-16,  2.27544977e-01,  4.67698799e-01,
  7.38846849e-01,  1.08031934e+00,  1.64485363e+00]

#print(np.searchsorted(l, e))

x = np.linspace(0, 3 * np.pi, 500)
y = np.sin(x)

x1 = x[:int(500/3)]
x2 = x[int(500/3):int(500/3)*2]
x3 = x[int(500/3)*2:]

y1 = np.sin(x1)
y2 = np.sin(x2)
y3 = np.sin(x3)

plt.plot(x1,y1)
plt.plot(x2,y2,linestyle = '-.')
plt.plot(x3,y3,linestyle = ':')
#plt.show()
