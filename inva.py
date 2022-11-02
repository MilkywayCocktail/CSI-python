import numpy as np
import pycsi
import csitest
import os
import csi_loader

npzpath = '../npsave/1030/csi/'

n = '1030A02'

#csi_loader.dat2npy('../data/1030/csi1030A02.dat', npzpath)

file = npzpath + n + '-csio.npy'

a, b, c, d = csi_loader.load_npy(file)


csi = pycsi.MyCsi(n, file)
csi.load_lists(np.abs(a).swapaxes(1,3), np.angle(a).swapaxes(1,3), b)
csi.data.show_shape()