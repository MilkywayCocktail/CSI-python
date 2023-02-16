import os
import numpy as np


def regroup(in_path, out_path, scope: tuple):
    # Initial cell shapes
    result = {'csi': np.zeros((1, 2, 100, 30, 3)),
            'img': np.zeros((1, 128, 128)),
            'tim': np.zeros(1),
            'cod': np.zeros((1, 3)),
            'ind': np.zeros(1),
            'sid': np.zeros((1, 3))
            }

    filenames = os.listdir(in_path)
    for file in filenames:

        if file[:2] in scope:
            tmp = np.load(in_path + file)

            print(file)
            kind = file[-7:-4]

            if kind in result.keys():
                result[kind] = np.concatenate((result[kind], tmp), axis=0)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for key in result.keys():
        result[key] = np.delete(result[key], 0, axis=0)
        if len(result[key]) != 0:
            print(key, len(result[key]))
            np.save(out_path + key + '.npy', result[key])

    print("All saved!")


def onehotscale(in_path, out_path):
    labels = np.load(in_path)
    labels = labels * 10
    np.save(out_path, labels )
    print("All saved!")


if __name__ == '__main__':
    #regroup('../dataset/0208/make01/', '../dataset/0208/make01_finished/', scope=('02', '03'))
    onehotscale('../dataset/0208/make00_finished/sid2.npy', '../dataset/0208/make00_finished/sid_10.npy')
