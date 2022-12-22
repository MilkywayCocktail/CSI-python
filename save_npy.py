import numpy as np
import pycsi
import csitest
import os
import csi_loader

npzpath = '../npsave/1222/'
datapath = "../data/1222/"
name = 'csi1222A60.dat'

csi_loader.dat2npy(datapath + name, npzpath)

#filenames = os.listdir(datapath)

'''
for file in filenames:
    if file[-3:] == 'txt':
        continue
    csi_loader.dat2npy(datapath + file, npzpath)



file = npzpath + name + '-csio.npy'

a, b, c, d = csi_loader.load_npy(file)
b = np.array(b) / 1e6

csi = pycsi.MyCsi(name, file)
csi.load_lists(np.abs(a).swapaxes(1,3), np.angle(a).swapaxes(1,3), b)
'''