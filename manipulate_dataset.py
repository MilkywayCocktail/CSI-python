import numpy as np
import cv2
import os
import random


def regroup(in_path, out_path, scope: tuple):
    # Initial cell shapes
    result = {'csi': np.zeros((1, 2, 90, 100)),
            'img': np.zeros((1, 1, 128, 128)),
            'tim': np.zeros(1),
            'cod': np.zeros((1, 3)),
            'ind': np.zeros(1),
            'sid': np.zeros(1)
            }

    filenames = os.listdir(in_path)
    for file in filenames:

        if file[:2] in scope:
            tmp = np.load(in_path + file)

            print(file)
            kind = file[-7:-4]

            if kind in result.keys():
                if kind == 'img':
                    tmp = tmp[:, None, ...]
                    tmp[tmp > 4000] = 4000

                result[kind] = np.concatenate((result[kind], tmp), axis=0)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for key in result.keys():
        result[key] = np.delete(result[key], 0, axis=0)
        if len(result[key]) != 0:
            if key == 'sid':
                result[key] = result[key] - min(result[key])

            print(key, len(result[key]))
            np.save(out_path + key + '.npy', result[key])


    print("All saved!")


def asx(path):
    np.set_printoptions(threshold=np.inf)
    x = np.load(path)
    print(x.shape)
    print(x.dtype)
    print(x[100])


def asy(path):
    vmap = np.load(path)

    for i in range(len(vmap)):
        img = cv2.convertScaleAbs(vmap[i], alpha=0.03)
        cv2.namedWindow('Velocity Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Velocity Image', img)
        #cv2.imwrite('../dataset/view/' + str(i).zfill(4) + '.jpg', img)
        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
    return


def label_convert(in_path, out_path=None, autosave=False):
    labels = []
    with open(in_path) as f:
        for i, line in enumerate(f):
            if i > 0:
                labels.append([eval(line.split(',')[0]), eval(line.split(',')[1])])

    labels = np.array(labels)

    if autosave is True and out_path is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        np.save(out_path, labels)

    else:
        return labels


def to_onehot(path, path2):
    labels = np.load(path)
    out = np.zeros((len(labels), 3))

    for i in range(len(labels)):
        if labels[i] == 0:
            print("-1")
            out[i] = [1, 0, 0]
        elif labels[i] == 1:
            print("0")
            out[i] = [0, 1, 0]
        elif labels[i] == 2:
            print("1")
            out[i] = [0, 0, 1]

    np.save(path2, out)


def from_onehot(path, path2):
    labels = np.load(path)
    out = np.zeros(len(labels))

    for i in range(len(labels)):
        if (labels[i] == [1, 0, 0]).all():
            print("0")
            out[i] = 0
        elif (labels[i] == [0, 1, 0]).all():
            print("1")
            out[i] = 1
        elif (labels[i] == [0, 0, 1]).all():
            print("2")
            out[i] = 2

    np.save(path2, out)


def pseudo_dataset(out_path):
    csi = np.ones((1000, 1, 100, 90), dtype=complex) * (-1)
    csi_1 = np.ones((1000, 1, 100, 90), dtype=complex) * 0.5j
    csi_2 = np.ones((1000, 1, 100, 90), dtype=complex)

    sid = np.zeros(1000)
    sid1 = np.ones(1000)
    sid2 = np.ones(1000) * 2

    csi = np.concatenate((csi, csi_1, csi_2), axis=0)
    sid = np.concatenate((sid, sid1, sid2), axis=0)
    out_sid = np.zeros((3000, 3))
    for i in range(len(sid)):
        if sid[i] == -1:
            print("-1")
            out_sid[i] = [1, 0, 0]
        elif sid[i] == 0:
            print("0")
            out_sid[i] = [0, 1, 0]
        elif sid[i] == 1:
            print("1")
            out_sid[i] = [0, 0, 1]

    csi_abs = np.abs(csi)
    csi_phase = np.angle(csi)
    out_csi = np.concatenate((csi_abs, csi_phase), axis=1)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    np.save(out_path + 'csi.npy', out_csi)
    np.save(out_path + 'sid_.npy', sid)
    print("All saved!")


def pseudo_dataset_frq(out_path):
    x = np.arange(2000)
    y1 = np.sin(x)
    y2 = np.sin(2 * x)
    y3 = np.sin(3 * x)

    csi1 = np.zeros((1000, 1, 100, 90), dtype=complex)
    csi2 = np.zeros((1000, 1, 100, 90), dtype=complex)
    csi3 = np.zeros((1000, 1, 100, 90), dtype=complex)

    ind = np.zeros((3, 1000), dtype=int)
    for i in range(3):
        ind[i] = [random.randint(0, 1900) for _ in range(1000)]

    for i in range(1000):
        csi1[i] = 2 * np.exp(1j * np.arcsin(y1[ind[0,i]:ind[0,i] + 100]))[..., np.newaxis].repeat(90, axis=1).reshape(1, 1, 100, 90)
        csi2[i] = np.exp(1j * np.arcsin(y2[ind[1,i]:ind[1,i] + 100]))[..., np.newaxis].repeat(90, axis=1).reshape(1, 1, 100, 90)
        csi3[i] = 0.5 * np.exp(1j * np.arcsin(y3[ind[2,i]:ind[2,i] + 100]))[..., np.newaxis].repeat(90, axis=1).reshape(1, 1, 100, 90)

    sid = np.zeros(1000)
    sid1 = np.ones(1000)
    sid2 = np.ones(1000) * 2

    csi = np.concatenate((csi1, csi2, csi3), axis=0)
    sid = np.concatenate((sid, sid1, sid2), axis=0)

    csi_abs = np.abs(csi)
    csi_phase = np.angle(csi)
    out_csi = np.concatenate((csi_abs, csi_phase), axis=1)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    np.save(out_path + 'csi.npy', out_csi)
    np.save(out_path + 'sid.npy', sid)
    print("All saved!")


def simu_dataset(paths, out_path):
    out = []
    sid = []
    for path in paths:
        csi = np.load(path, allow_pickle=True)
        s = eval(path[-10])
        for i in range(500):
            amp = csi.item()['amp'][i * 100: (i + 1) * 100].reshape(100, 90).T
            phs = csi.item()['phs'][i * 100: (i + 1) * 100].reshape(100, 90).T
            data = np.concatenate((amp[np.newaxis, ...], phs[np.newaxis, ...]), axis=0)
            out.append(data)
            sid.append(s)
    out = np.array(out)
    sid = np.array(sid)
    print(out.shape)
    print(sid.shape)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #np.save(out_path + 'csi.npy', out)
    np.save(out_path + 'sid.npy', sid)


if __name__ == '__main__':
    #pseudo_dataset('../dataset/0221/make01_finished/')
    #asy('../dataset/0307/make04-finished/img.npy')
    #asx('../dataset/0307/make04-finished/img.npy')
    #to_onehot('../dataset/0208/make00_finished/sid.npy', '../dataset/0208/make00_finished/sid2.npy')
    #from_onehot('../dataset/0208/make00_finished/sid_oh.npy', '../dataset/0208/make00_finished/sid.npy')
    #pseudo_dataset_frq('../dataset/0302/make00_finished/')
    #asx('../dataset/0302/make00_finished/csi.npy')

    regroup('../dataset/0307/make04/', '../dataset/0307/make05-finished/', ('04'))