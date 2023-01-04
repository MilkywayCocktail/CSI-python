import numpy as np
import pycsi
import csitest
import os
import csi_loader


def windowed_dynamic(in_csi):
    phase_diff = in_csi[:, :, :, :] * in_csi[:, :, 0, :].conj().repeat(3, axis=2)
    static = np.mean(phase_diff, axis=0)
    dynamic = in_csi - static
    return dynamic


if __name__ == '__main__':
    sourcepath = '../data/1222'
    savepath = '../dataset/1213/dyn/'

    name = 'csi1222A60.dat'

    filenames = os.listdir(sourcepath)

    for file in filenames:
        if file[-3:] == 'txt':
            continue
        csi, timestamps = csi_loader.dat2npy(sourcepath + file, savepath, autosave=False)
