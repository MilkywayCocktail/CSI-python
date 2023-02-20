import numpy as np
import cv2
import os


def ast(path):
    ts = np.load(path)
    print(ts[:10])


def asx(path):
    csi = np.load(path)
    print(csi.shape)


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

#asy('../dataset/0124/make02/03_dyn_img.npy')
#asx('../dataset/1213/masked_depth/01_x.npy')
ast('../dataset/0208/make01_finished/sid.npy')
#to_onehot('../dataset/0208/make00_finished/sid.npy', '../dataset/0208/make00_finished/sid2.npy')

