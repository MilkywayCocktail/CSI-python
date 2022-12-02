import pyrealsense2 as rs
import numpy as np
import cv2
import os

path = '../sense/'

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, path + '1126.bag', False)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
pipeline.start(config)

save_path = path + 'save/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

i = 0
while True:
    frames = pipeline.wait_for_frames()
    print("Frame claimed")
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path + str(i).zfill(4) + '.jpg', color_image)
    i += 1
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
