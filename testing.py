import numpy as np
import pycsi
import csitest
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

dimage = Image.open('../sense/1122/take7/dimg00008.jpg')
image = Image.open('../sense/1122/take7/timg00008.jpg')

dmat = np.array(dimage)
mat = np.array(image)

diff = dmat - mat

cv2.imshow('diff', diff)
cv2.waitKey(0)