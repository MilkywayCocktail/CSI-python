import numpy as np
import pycsi
import csitest
import os
import csi_loader

sourcepath = '../data/1222'
savepath = '../dataset/1222/'

name = 'csi1222A60.dat'


def preprocess(npyfile):
    pass

filenames = os.listdir(sourcepath)

for file in filenames:
    if file[-3:] == 'txt':
        continue
    csi_loader.dat2npy(sourcepath + file, savepath)
