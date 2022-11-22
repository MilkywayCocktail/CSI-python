import cv2
import pyrealsense2 as rs
import numpy as np
import time
import os

pipeline = rs.pipeline()
profile = pipeline.start()
sensor = profile.get_device().first_depth_sensor()
scale = sensor.get_depth_scale()
print(scale)

index = 0
img = 0
t0 = int(round(time.time() * 1000))
while index < 10:

    t = int(round(time.time() * 1000)) - t0
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    print(np.max(depth_image), np.min(depth_image))
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    dist_image = np.zeros_like(depth_image)
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            dist_image[y, x] = depth_frame.get_distance(x, y) * 100     # distance in cm
    transfer_image = depth_image // 10
    index += 1

    if index % 2 == 0:

        path = '../sense/1122/take7/'
        if not os.path.exists(path):
            os.makedirs(path)
        timestamp = open(path + 'timestamps.txt', mode='a', encoding='utf-8')
        fname = str(index).zfill(5) + '.jpg'
        cv2.imwrite(path + 'img' + fname, depth_image)
        cv2.imwrite(path + 'dimg' + fname, dist_image)
        cv2.imwrite(path + 'timg' + fname, transfer_image)
        cv2.imwrite(path + 'cimg' + fname, depth_colormap)
        timestamp.write(str(t) + '\n')
        print("save successful")
        img += 1
