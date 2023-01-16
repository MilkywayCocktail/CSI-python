import numpy as np
import matplotlib.pyplot as plt

a = np.load('../dataset/coord.npy')
print(a.shape)
dm = np.max(a, axis=0)[2]
print(dm)

for i in range(len(a)):
    x = a[i][0]
    y = a[i][1]
    d = a[i][2]
    plt.scatter(x, y, color=(d/dm, 0.2, 0.2))

plt.show()