import csi_loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

result = np.array([3.1,4.2,5,6,4,3,5,6,6,4,3,3,5,6,2])
result[5:8] = np.NaN
x = np.arange(len(result))

plt.scatter(x, result, c=np.log(result))
plt.show()