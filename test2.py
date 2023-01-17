import numpy as np
import matplotlib.pyplot as plt

a = np.load('../dataset/coord.npy')
print(a.shape)

#a[a > 3500] = 0
dm = np.max(a, axis=0)[2]
print(dm)

plt.hist(a[:,2], bins=100)
plt.show()

for i in range(len(a)):
    x = a[i][0]
    y = a[i][1]
    if a[i][2] >= 3500:
        plt.scatter(x, y, color=(0, 1, 1))
    else:
        plt.scatter(x, y, color=(a[i][2]/3500, 0.2, 0.2))
plt.show()