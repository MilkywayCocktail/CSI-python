import numpy as np
import cv2

path = '../sense/1124/take6/take6'

dmatrix = np.load(path + '_dmatrix.npy')[:300]
time = np.load(path + '_timestamps.npy')[:300]

time = time - time[0]

d_diff = dmatrix[1:] - dmatrix[:-1]

v_map = d_diff / time[1:, np.newaxis, np.newaxis].repeat(480, axis=1).repeat(848, axis=2)

print(np.max(v_map))
print(np.min(v_map))

for i in range(299):
    cv2.imshow("vmap", v_map[i] * 100)
    cv2.waitKey(500)