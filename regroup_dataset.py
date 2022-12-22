import os
import numpy as np


def run(datapath):
    x = np.zeros((1, 2, 90, 33))
    y = np.zeros((1, 120, 200))
    t = np.zeros(1)

    filenames = os.listdir(datapath)
    for file in filenames:

        if file[:2] in ('00', '01', '04'):
            tmp = np.load(datapath + file)

            print(file)

            if file[-5] == 'x':
                x = np.concatenate((x, tmp), axis=0)

            elif file[-5] == 'y':
                y = np.concatenate((y, tmp), axis=0)

            elif file[-5] == 't':
                t = np.concatenate((t, tmp), axis=0)

    x = np.delete(x, 0, axis=0)
    y = np.delete(y, 0, axis=0)
    t = np.delete(t, 0, axis=0)

    print(x.shape, y.shape, t.shape)

    savepath = '../dataset/concat/1213/depth_3m/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    np.save(savepath + 'x.npy', x)
    np.save(savepath + 'y.npy', y)
    np.save(savepath + 't.npy', t)

    print("All saved!")


run('../dataset/1213/depth/')
