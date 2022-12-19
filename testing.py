import numpy as np
import pycsi
import csitest
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import csi_loader

import rosbag

bag = rosbag.Bag('../sense/1202/T01.bag', 'r')
info = bag.g
print(info)