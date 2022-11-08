import pycsi
import csitest
import numpy as np
import random
import matplotlib.pyplot as plt

a = 0 + 1j
b = 0 + 1j
print("a+b=", a+b)
print("a*b=", a*b)

c = np.exp(1j)
d = np.exp(1j)
print("c,d", np.angle(c), np.angle(d))
print("c+d=", np.angle(c+d))
print("c*d=", np.angle(c*d))