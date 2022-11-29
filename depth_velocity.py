import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import pyrealsense2 as rs

path = '../sense/11282'

dmatrix = np.load(path + '_dmatrix.npy')[:300]
time = np.load(path + '_timestamps.npy')[:300]

time = (time - time[0]) / 1.e3

d_diff = dmatrix[1:] - dmatrix[:-1]

v_map = d_diff / time[1:, np.newaxis, np.newaxis].repeat(480, axis=1).repeat(848, axis=2)

print(np.max(v_map))
print(np.min(v_map))

for i in range(299):
    sns.heatmap(v_map[i])
    plt.show()

#    https: // www.cxyzjd.com / article / qq_25105061 / 111312298