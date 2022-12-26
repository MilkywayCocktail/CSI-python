import pyrealsense2 as rs
import numpy as np
import cv2
from bag_to_matrix import my_filter




def align_channels(frames, show=True):
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    depth_frame_aligned = aligned_frames.get_depth_frame()
    color_frame_aligned = aligned_frames.get_color_frame()
    color_image_aligned = np.asanyarray(color_frame_aligned.get_data())
    color_image_aligned = cv2.cvtColor(color_image_aligned, cv2.COLOR_BGR2RGB)
    depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())
    depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.05), cv2.COLORMAP_BONE)
    if show:
        images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
        cv2.imshow('aligned images', images_aligned)
        key = cv2.waitKey(1) & 0xFF


def mask(frames, show=True):
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    depth_frame_aligned = aligned_frames.get_depth_frame()
    color_frame_aligned = aligned_frames.get_color_frame()
    color_image_aligned = np.asanyarray(color_frame_aligned.get_data())
    color_image_aligned = cv2.cvtColor(color_image_aligned, cv2.COLOR_BGR2RGB)
    depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())
    depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.05), cv2.COLORMAP_BONE)
    if show:
        images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
        cv2.imshow('aligned images', images_aligned)
        key = cv2.waitKey(1) & 0xFF


def run(in_path):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(in_path, False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)


    try:
        while True:
            frames = pipeline.wait_for_frames()
            if not frames:
                continue
            align_channels(frames)

    except RuntimeError:
        print("Read finished!")
        pipeline.stop()

    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run('../sense/1213/121303.bag')