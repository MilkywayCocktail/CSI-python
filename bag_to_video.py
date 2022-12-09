import pyrealsense2 as rs
import numpy as np
import cv2
import os

path = '../sense/1202/'
source_name = 'T04.bag'
export_name = 'T04.avi'

fps = 30

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, path + source_name, False)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, fps)
pipeline.start(config)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videowriter = cv2.VideoWriter(path + export_name, fourcc, fps, (1280, 720))

hole_filling = rs.hole_filling_filter()

decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 1)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
spatial.set_option(rs.option.filter_smooth_delta, 50)

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)


def filters(frame):
    filter_frame = decimate.process(frame)
    filter_frame = depth_to_disparity.process(filter_frame)
    filter_frame = spatial.process(filter_frame)
    filter_frame = disparity_to_depth.process(filter_frame)
    filter_frame = hole_filling.process(filter_frame)
    result_frame = filter_frame.as_depth_frame()
    return result_frame


try:
    while True:
        frames = pipeline.wait_for_frames()
        print("Frame claimed")
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        videowriter.write(color_image)

        cv2.namedWindow('Bag Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Bag Image', color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    videowriter.release()
    print("Video saved!")