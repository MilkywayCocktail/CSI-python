import numpy as np
import pycsi
import csitest
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import csi_loader
import random
import rosbag
import tqdm

# f = "1213env.npy"
# d = np.load(f)
# print(d.shape)
f = "../dataset/compressed/121301.npy"
d = np.load(f)
print(d.shape)
time_window = 10
# mean = np.mean(d, axis=0)
# std = np.std(d, axis=0)
# threshold = mean - 2 * std
# median = np.median(d, axis=0)
# mad = np.mean(np.abs(d - median[np.newaxis]), axis=0)
# threshold = median - 2 * mad
median = np.median(d, axis=0)
threshold = median * 0.5
fig, axs = plt.subplots(2, 2, figsize=(16, 9))
axs = axs.flatten()
im0 = axs[0].imshow(d[0], "gray", vmin=0, vmax=2 ** 16)
im1 = axs[1].imshow(d[0], "gray", vmin=0, vmax=4000)
im2 = axs[2].imshow(d[0], "gray", vmin=0, vmax=2 ** 16)
im3 = axs[3].imshow(d[0], "gray", vmin=0, vmax=4000)
axs[0].set_title("raw")
axs[1].set_title("threshold <4000") # ignore out of sensing area
axs[2].set_title("median filter") # compute median in time window
axs[3].set_title("threshold <median/2") # ignore background assumed as median
for i in tqdm.tqdm(range(len(d))):
    im0.set_data(d[i])
    mask = d[i] < 4000
    im1.set_data(d[i] * mask)
    im2.set_data(np.median(d[i:i + time_window], axis=0))
    mask = d[i] < threshold
    im3.set_data(d[i] * mask)
    fig.savefig("../dataset/images/{:04}.jpg".format(i))
    # plt.pause(0.1)