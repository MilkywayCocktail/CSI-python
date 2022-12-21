import numpy as np
import cv2


def ast(path):
    ts = np.load(path)
    print(ts)


def asx(path):
    csi = np.load(path)
    print(csi.shape)


def asy(path):
    vmap = np.load(path)
    for i in range(len(vmap)):
        img = cv2.convertScaleAbs(vmap[i], alpha=0.03)
        cv2.namedWindow('Velocity Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Velocity Image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    return


ast('../dataset/1213/depth/01/t.npy')
asx('../dataset/1213/depth/01/x.npy')
asy('../dataset/1213/depth/01/y.npy')

