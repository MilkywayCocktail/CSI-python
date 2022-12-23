import pyrealsense2 as rs
import numpy as np
import cv2
import os


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


def circle_mask(border, cutoff):
    mask = np.zeros((border, border))
    for i in range(border):
        for j in range(border):
            arm = np.power(i - border / 2, 2) + np.power(j - border / 2, 2)
            mask[i, j] = 1 if arm <= np.power(cutoff, 2) else 0

    return mask


def low_pass_filter(patch, mask):
    p = np.fft.fft2(patch)
    pshift = np.fft.fftshift(p)
    filtered = np.multiply(pshift, mask)
    repatchshift = np.fft.ifftshift(filtered)
    repatch = np.fft.ifft2(repatchshift)

    return np.abs(repatch)


def spatial_smooth(frame, kernel=(80, 80), step=(10, 5)):
    out_frame = np.zeros_like(frame)
    w_iter = (frame.shape[1] - kernel[0]) // step[0]
    h_iter = (frame.shape[0] - kernel[1]) // step[1]
    rows = (0, 0)
    cols = (0, 0)

    for i in range(h_iter):
        for j in range(w_iter):
            rows = (i * step[1], i * step[1] + kernel[1])
            cols = (j * step[0], j * step[0] + kernel[0])

            if i == h_iter - 1:
                rows = (frame.shape[0] - kernel[1], frame.shape[0])
            if j == w_iter - 1:
                cols = (frame.shape[1] - kernel[0], frame.shape[1])

            patch = frame[rows[0]: rows[1], cols[0]: cols[1]]

            out_frame[rows[0]: rows[1],
                      cols[0]: cols[1]] = low_pass_filter(patch, circle_mask(80, 25))

    return out_frame


def run(in_path, out_path):
    chunk = []

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(in_path, False)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    try:
        t_tmp = 0
        t1 = 0
        t_vmap = np.zeros((480, 848))

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_frame = my_filter(depth_frame)
            depth_image = np.asanyarray(depth_frame.get_data())
            chunk.append(depth_image)

            timestamp = frames.timestamp
            if t_tmp == 0:
                t_tmp = timestamp
            timestamp = timestamp - t_tmp
            print(timestamp)

            vmap = depth_image - t_vmap
            if t1 != 0:
               vmap = vmap / (timestamp - t1)
            t1 = timestamp

            #vmap = spatial_smooth(vmap)
            vmap = cv2.convertScaleAbs(vmap, alpha=0.4)
            depth_image = cv2.convertScaleAbs(depth_image, alpha=0.02)

            #vmap = cv2.bilateralFilter(vmap, 50, 100, 5)

            # vmap = cv2.bilateralFilter(vmap, 0, 100, 5)
            # vmap = cv2.blur(vmap, (15, 15))
            # vmap = cv2.bilateralFilter(vmap, 0, 100, 5)

            t_vmap = depth_image

            cv2.namedWindow('Velocity Image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Velocity Image', depth_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except RuntimeError:
        print("Read finished!")
        pipeline.stop()

    finally:
        cv2.destroyAllWindows()
        chunk = np.array(chunk)
        np.save(out_path, chunk.astype(np.uint16))


if __name__ == '__main__':
    run('../sense/1213/121303.bag', '../dataset/compressed/121303.npy')
    run('../sense/1213/121304.bag', '../dataset/compressed/121304.npy')
