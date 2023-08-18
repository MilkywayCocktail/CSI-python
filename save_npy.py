import numpy as np
import pycsi
import csitest
import os
import csi_loader
import matplotlib.pyplot as plt


def remove_sm_loop(inpath, rate, autosave=True):
    """
    After removing sm, save as -csis.npy
    :param inpath: input path
    :param rate: monitor_tx_rate
    :return:
    """
    csi, tim, j, k = csi_loader.load_npy(inpath)
    print(csi.shape)
    pkt = np.random.randint(csi.shape[0])
    plt.subplot(2, 1, 1)
    _pkt = csi[pkt].reshape(30, 9)
    _phs = np.unwrap(np.angle(_pkt), axis=0)
    for l in range(9):
        plt.plot(_phs[:, l], label=l)
    plt.legend()
    plt.grid()
    plt.title("before")
    plt.xlabel("sub")
    plt.ylabel("phase")

    for packet in range(len(csi)):
        csi[packet] = csi_loader.remove_sm(csi[packet], rate)

    plt.subplot(2, 1, 2)
    _pkt = csi[pkt].reshape(30, 9)
    _phs = np.unwrap(np.angle(_pkt), axis=0)
    for l in range(9):
        plt.plot(_phs[:, l], label=l)
    plt.legend()
    plt.grid()
    plt.title("after")
    plt.xlabel("sub")
    plt.ylabel("phase")

    out_path = inpath.replace('csio', 'csis')
    save = {'csi': csi,
            'time': np.array(tim) / 1e6}
    if autosave is True:
        np.save(out_path, save)

    plt.tight_layout()
    plt.show()
    return csi


def save_npy(inpath, outpath):

    filenames = os.listdir(inpath)

    for file in filenames:
        if file[-3:] == 'txt':
            continue
        csi = csi_loader.dat2npy(inpath + file, outpath, autosave=True)


if __name__ == '__main__':

    npypath = '../npsave/0726/'
    datapath = "../data/0726/"
    save_npy(datapath, npypath)
    #remove_sm_loop('../npsave/0307/0307A04-csio.npy', 0x1c113, autosave=False)
