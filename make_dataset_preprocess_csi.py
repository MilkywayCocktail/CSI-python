import numpy as np
import pycsi
import csitest
import os
import csi_loader


def windowed_dynamic(in_csi):
    # in_csi = np.squeeze(in_csi)
    phase_diff = in_csi * in_csi[..., 0][..., np.newaxis].conj().repeat(3, axis=2)
    static = np.mean(phase_diff, axis=0)
    dynamic = in_csi - static
    return dynamic


def load_csis(in_path, out_path):
    filenames = os.listdir(in_path)

    for file in filenames:
        if file[-3:] == 'txt':
            continue
        csi, timestamps = csi_loader.dat2npy(in_path + file, out_path, autosave=False)
        yield [csi.swapaxes(1, 3), timestamps]


if __name__ == '__main__':
    sourcepath = '../data/1213/'
    savepath = '../dataset/1213/dyn/'

    csi = load_csis(sourcepath, savepath)
    csit = windowed_dynamic(next(csi)[0])
    print("complete!")



