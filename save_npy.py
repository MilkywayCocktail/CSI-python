import numpy as np
import pycsi
import csitest
import os
import csi_loader


def remove_sm_loop(inpath, rate):
    csi, i, j, k = csi_loader.load_npy(inpath)
    print(csi.shape)
    for packet in range(len(csi)):
        csi[packet] = csi_loader.remove_sm(csi[packet], rate)
    print(csi.shape)
    return csi


def save_npy(inpath, outpath):

    filenames = os.listdir(inpath)

    for file in filenames:
        if file[-3:] == 'txt':
            continue
        csi_loader.dat2npy(inpath + file, outpath)


if __name__ == '__main__':

    '''
    name = 'csi1222A60.dat'
    file = npzpath + name + '-csio.npy'p
    
    a, b, c, d = csi_loader.load_npy(file)
    b = np.array(b) / 1e6
    
    csi = pycsi.MyCsi(name, file)
    csi.load_lists(np.abs(a).swapaxes(1,3), np.angle(a).swapaxes(1,3), b)
    '''

    npypath = '../npsave/0307/'
    datapath = "../data/0307/"
    #save_npy(datapath, npypath)
    remove_sm_loop('../npsave/0307/0307A00-csio.npy', 0x1c113)
