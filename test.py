import csi_loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time

t = time.time()
t1 = t + 1
print(t1)

print(datetime.datetime.fromtimestamp(t1) - datetime.datetime.fromtimestamp(t))