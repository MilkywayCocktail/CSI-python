import pyrealsense2 as rs
import numpy as np
import cv2

path = '../sense/1126/'

config = rs.config()
config.enable_device_from_file(path + '1126.bag')
config.enable_all_streams()
pipeline = rs.pipeline()
profile = pipeline.start(config)
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        images = np.hstack((color_image, depth_color_image))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
               break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
