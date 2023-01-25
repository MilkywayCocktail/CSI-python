import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
np.set_printoptions(suppress=True)

file = '../sense/0124/03_timestamps.txt'
result = '../sense/0124/03_timediff.txt'

f = open(file, mode='r', encoding='utf-8')
r = open(result, mode='w+', encoding='utf-8')

out = np.array(f.readlines())
start = None
for i in range(len(out)):
    if start is None:
        start = datetime.strptime(out[i].strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp()
    out[i] = datetime.strptime(out[i].strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp() - start

print(out)
np.savetxt(result, out, fmt='%s')
