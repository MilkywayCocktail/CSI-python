import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import rosbag

path = '../sense/1202/'
source_name = 'T01.bag'
export_name = 'T01_vmap_nc.avi'

bag = rosbag.Bag(path + source_name, "r")

fps = 30

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, path + source_name, False)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
pipeline.start(config)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videowriter = cv2.VideoWriter(path + export_name, fourcc, fps, (848, 480))

try:
    t_tmp = 0
    t1 = 0
    t_vmap = np.zeros((480, 848))

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth_image = np.asanyarray(depth_frame.get_data())

        timestamp = frames.timestamp
        if t_tmp == 0:
            t_tmp = timestamp
        timestamp = timestamp - t_tmp
        print(timestamp)

        vmap = depth_image - t_vmap
        if t1 != 0:
            vmap = vmap / (timestamp - t1)
        t1 = timestamp

        videowriter.write(cv2.convertScaleAbs(vmap, alpha=0.03))

        t_vmap = depth_image


except RuntimeError:
    print("Read finished!")
finally:
    pipeline.stop()
    videowriter.release()

#    https: // www.cxyzjd.com / article / qq_25105061 / 111312298