import numpy as np
import pycsi
import csitest
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import csi_loader

import rosbag

ts = np.load('../dataset/env/t.npy')
print(len(ts))