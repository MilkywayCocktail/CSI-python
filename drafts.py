import pycsi
import csitest
import numpy as np
import random
import matplotlib.pyplot as plt

l = np.arange(10).reshape(2,5)
print(l)
print(l.reshape(-1, 1))
print(l.reshape(1, -1))
