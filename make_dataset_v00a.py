# Generates (x,y,d) coordinates for masks
# Please run make_dataset first to get masked depth images

import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_coordinates(path):
    dmap = np.load(path)
    out = np.zeros((len(dmap), 3))
    areas = np.zeros(len(dmap))

    for i in range(len(dmap)):
        (T, timg) = cv2.threshold(dmap[i].astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            areas[i] = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            xc, yc = int(x + w / 2), int(y + h / 2)
            img = cv2.rectangle(cv2.cvtColor(np.float32(dmap[i]), cv2.COLOR_GRAY2BGR), (x, y), (x + w, y + h), (0, 255, 0), 1)
            img = cv2.circle(img, (xc, yc), 1, (0, 0, 255), 4)

            if areas[i] > 400:
                out[i] = np.array([xc, yc, dmap[i][yc, xc]])
            else:
                out[i] = np.array([100, 60, 0])
        else:
            img = dmap[i]
            out[i] = np.array([100, 60, 0])

        cv2.namedWindow('Velocity Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Velocity Image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    return out


a = make_coordinates('../dataset/1213/make00_finished/y.npy')
np.save('../dataset/coord.npy', a)
