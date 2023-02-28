import csi_loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def wrap(x):
    y = (x + np.pi) % (2 * np.pi) - np.pi
    return y


x = np.arange(40)
y1 = wrap(0.5 * x)
y2 = wrap(0.8 * x)

plt.plot(y1)
plt.plot(y2)
plt.legend()
plt.show()
