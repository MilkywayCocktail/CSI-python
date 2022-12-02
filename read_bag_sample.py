import pyrealsense2 as rs
import numpy as np
import cv2

path = '../sense/'

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, path + '1126.bag', False)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

pipeline.start(config)
try:
    while True:
        frames = pipeline.wait_for_frames()
        print("Frame claimed")
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.resize(color_image, (848, 480), interpolation=cv2.INTER_CUBIC)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_frame.get_data())
        #depth_color_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_color_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_color_image))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
               break
finally:
    cv2.destroyAllWindows()

