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
    color_image_aligned = cv2.fastNlMeansDenoisingColored(color_image_aligned, None, 10, 10, 7, 15)
    depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())
    depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.05), cv2.COLORMAP_BONE)
    if show:
        images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
        cv2.imshow('aligned images', images_aligned)
        key = cv2.waitKey(1) & 0xFF
    return color_image_aligned


def mask(image, background, show=True):
    image_masked = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) - cv2.cvtColor(background, cv2.COLOR_RGB2GRAY))
    #m = image_masked < 160
    #mage_masked = image_masked * m

    if show:
        cv2.imshow('masked images', image_masked)
        key = cv2.waitKey(1) & 0xFF
    return image_masked


def make_masked(in_path, out_path, background, length):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(in_path, False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    bg = cv2.imread(background)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if not frames:
                continue
            color_image = align_channels(frames, show=False)
            masked_image = mask(color_image, bg)

    except RuntimeError:
        print("Read finished!")
        pipeline.stop()

    finally:
        cv2.destroyAllWindows()


def make_background(in_path, out_path, length):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(in_path, False)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    chunk = np.zeros((length, 720, 1280, 3))
    i = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                # i -= 1
                continue
            color_image = np.asanyarray(color_frame.get_data())
            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            chunk[i] = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
            i += 1

    except RuntimeError:
        print("Read finished!")
        pipeline.stop()

    finally:
        print(chunk.shape)
        env_image = np.squeeze(np.mean(chunk, axis=0))

        cv2.imshow('env', env_image)
        key = cv2.waitKey(1) & 0xFF
        cv2.imwrite(out_path, env_image)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    make_masked('../sense/1213/121303.bag', '../sense/1213/', '../sense/1213/env_new.jpg', length=1800)
    #make_background('../sense/1213/1213env.bag', '../sense/1213/env_new.jpg', 300)