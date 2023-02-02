import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

path = '../sense/0124/'

filenames = os.listdir(path)
for file in filenames:
    if file[-3:] == 'bag':
        print(file)

        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, os.path.join(path, file), False)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)

        timefile = open(path + file[:-4] + '_cameratime2.txt', mode='w+', encoding='utf-8')
        difffile = open(path + file[:-4] + '_cameratimediff.txt', mode='w+', encoding='utf-8')
        start = None
        try:
            while True:

                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                timestamp = frames.get_timestamp() / 1000
                timefile.write(str(datetime.utcfromtimestamp(timestamp)) + '\n')
                if start is None:
                    start = timestamp
                timediff = timestamp - start
                difffile.write(str(timediff) + '\n')

                depth_color_frame = rs.colorizer().colorize(depth_frame)
                depth_color_image = np.asanyarray(depth_color_frame.get_data())
                #depth_color_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_color_image, alpha=0.03), cv2.COLORMAP_BONE)
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', depth_color_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except RuntimeError:
            pipeline.stop()

        finally:
            cv2.destroyAllWindows()
            timefile.close()
            difffile.close()

