import pyrealsense2 as rs
import numpy as np
import cv2
import os
import mask_by_depth


def my_filter(frame):
    hole_filling = rs.hole_filling_filter()

    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    # spatial.set_option(rs.option.holes_fill, 3)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.5)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    # temporal.set_option(rs.option.holes_fill, 3)

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    filter_frame = decimate.process(frame)
    filter_frame = depth_to_disparity.process(filter_frame)
    filter_frame = temporal.process(filter_frame)
    filter_frame = spatial.process(filter_frame)

    filter_frame = disparity_to_depth.process(filter_frame)
    filter_frame = hole_filling.process(filter_frame)
    result_frame = filter_frame.as_depth_frame()
    return result_frame


def run(in_path, out_path, total_frames):
    # Initial cell shapes
    chunk = np.zeros((total_frames, 120, 200))
    timestamps = np.zeros(total_frames)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(in_path, False)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    try:
        for i in range(total_frames):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            frame_timestamp = np.asanyarray(frames.timestamp)
            depth_frame = my_filter(depth_frame)
            depth_image = np.asanyarray(depth_frame.get_data())
            print('\r',
                  "\033[32mWriting frame " + str(i) + "\033[0m", end='')
            depth_image = cv2.resize(depth_image[24:824, :], (200, 120), interpolation=cv2.INTER_AREA)
            chunk[i] = depth_image.reshape(1, 120, 200)
            timestamps[i] = frame_timestamp

    except RuntimeError:
        print("Read finished!")
        pipeline.stop()

    finally:

        chunk_masked = mask_by_depth.mask_by_depth(chunk)
        np.save(out_path + "_y.npy", chunk_masked.astype(np.uint16))
        np.save(out_path + "_t.npy", timestamps - timestamps[0])
        print(chunk_masked.shape)
        print("Saved!")


if __name__ == '__main__':
    run('../sense/1213/1213env.bag', '../dataset/1213/masked_depth/00', 300)
    #run('../sense/1213/121304.bag', '../dataset/compressed/121304.npy')

