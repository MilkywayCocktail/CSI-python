import cv2
import pyrealsense2 as rs
import numpy as np
import time
import os


def depth2distance(frame):
    dist_image = np.zeros_like(depth_image)
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            dist_image[y, x] = depth_frame.get_distance(x, y) * 100     # distance in cm
    return dist_image


pipeline = rs.pipeline()
profile = pipeline.start()

index = 0
img = 0
t0 = int(round(time.time() * 1000))

path = '../sense/1124/take1/'

if not os.path.exists(path):
    os.makedirs(path)
timestamp = open(path + 'timestamps.txt', mode='w', encoding='utf-8')
timestamp.write("Timestamps in miliseconds\n")

while index < 10:

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    t = int(round(time.time() * 1000)) - t0
    depth_image = np.asanyarray(depth_frame.get_data())
    transfer_image = depth_image // 10

#    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    index += 1

    fname = str(img).zfill(5) + '.jpg'
    cv2.imwrite(path + 'img' + fname, depth_image)
    cv2.imwrite(path + 'timg' + fname, transfer_image)
    timestamp.write(str(t) + '\n')
    print("save successful")
    img += 1

pipeline.stop()

