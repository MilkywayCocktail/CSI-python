import csi_loader
import numpy as np

file = '../npsave/0124/0124A00-csio.npy'
a, b, c, d = csi_loader.load_npy(file)
b = np.array(b) / 1e3
b = b - b[0]

print(b[:10])