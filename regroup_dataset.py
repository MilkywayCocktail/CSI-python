import os
import numpy as np


def regroup(in_path, out_path, scope: set):
    # Initial cell shapes
    x = np.zeros((1, 2, 90, 33))
    y = np.zeros((1, 120, 200))
    t = np.zeros(1)

    filenames = os.listdir(in_path)
    for file in filenames:

        if file[:2] in scope:
            tmp = np.load(in_path + file)

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

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    np.save(out_path + 'x.npy', x)
    np.save(out_path + 'y.npy', y)
    np.save(out_path + 't.npy', t)

    print("All saved!")


if __name__ == '__main__':
    # regroup()
    pass
