import csi_loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def wrap(x):
    y = (x + np.pi) % (2 * np.pi) - np.pi
    return y


x = np.arange(50)
y1 = 0.5 * x
y2 = 0.3 * x
y3 = 1.3 * x

plt.plot(y1)
plt.plot(y2)
#plt.plot(y1 + y2)
#plt.plot(y3)
plt.legend()
plt.show()

