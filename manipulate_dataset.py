import numpy as np
import cv2
import os


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


def asx(path):
    x = np.load(path)
    print(x.shape)
    print(x[:10])


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
        if labels[i] == -1:
            print("-1")
            out[i] = [1, 0, 0]
        elif labels[i] == 0:
            print("0")
            out[i] = [0, 1, 0]
        elif labels[i] == 1:
            print("1")
            out[i] = [0, 0, 1]

    np.save(path2, out)


def onehotscale(in_path, out_path):
    labels = np.load(in_path)
    labels = labels * 10
    np.save(out_path, labels )
    print("All saved!")


def pseudo_dataset(out_path):
    csi = np.ones((1000, 1, 100, 30, 3), dtype=complex) * (-1)
    csi_1 = np.ones((1000, 1, 100, 30, 3), dtype=complex) * 0.5j
    csi_2 = np.ones((1000, 1, 100, 30, 3), dtype=complex)

    sid = np.ones(1000) * (-1)
    sid1 = np.zeros(1000)
    sid2 = np.ones(1000)

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
    np.save(out_path + 'sid_oh.npy', out_sid)
    print("All saved!")


if __name__ == '__main__':
    pseudo_dataset('../dataset/0221/make00_finished/')
    #asy('../dataset/0124/make02/03_dyn_img.npy')
    #asx('../dataset/1213/masked_depth/01_x.npy')
    #to_onehot('../dataset/0208/make00_finished/sid.npy', '../dataset/0208/make00_finished/sid2.npy')