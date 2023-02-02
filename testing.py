import numpy as np
import cv2

chunk = np.array([[1, 1, 1, 1,],[2, 2, 2, 2],[3, 3, 3, 3]])
conv = np.convolve(chunk, np.ones(3))
print(conv)