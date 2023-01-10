# Notes
#
# Data contents:
#
# CSI: .npy
# vmap: .bag
# Aligned by timestamps of camera frame


import pyrealsense2 as rs
import cv2
import csi_loader
import numpy as np
import sys
import os
import make_dataset_preprocess_csi


def my_filter(frame):
    hole_filling = rs.hole_filling_filter()

    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    filter_frame = decimate.process(frame)
    filter_frame = depth_to_disparity.process(filter_frame)
    filter_frame = spatial.process(filter_frame)
    filter_frame = disparity_to_depth.process(filter_frame)
    filter_frame = hole_filling.process(filter_frame)
    result_frame = filter_frame.as_depth_frame()
    return result_frame


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MyDataMaker:

    def __init__(self, paths: list, total_frames: int):
        self.paths = paths
        self.total_frames = total_frames
        self.video_stream = self.__setup_video_stream__()
        self.csi_stream = self.__setup_csi_stream__()
        self.result = self.__init_data__()

    def __len__(self):
        return self.total_frames

    def __setup_video_stream__(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.paths[0], False)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)

        return pipeline

    def __setup_csi_stream__(self):
        with HiddenPrints():
            csi, timestamps, i, j = csi_loader.load_npy(self.paths[1])
            return {'csi': csi.swapaxes(1, 3),
                    'time': timestamps-timestamps[0]}

    def __init_data__(self):
        x_csi = np.zeros((self.total_frames, 2, 90, 33))
        y_vmap = np.zeros((self.total_frames, 120, 200))
        t_list = np.zeros(self.total_frames)
        return {'x': x_csi, 'y': y_vmap, 't': t_list}

    def save(self, save_name: str):
        try:
            print('Starting making...')

            t_tmp = 0
            t1 = 0
            t_vmap = np.zeros((120, 200))

            for i in range(self.total_frames):
                frames = self.video_stream.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                depth_frame = my_filter(depth_frame)
                depth_image = np.asanyarray(depth_frame.get_data())

                depth_image = depth_image[24:824, :]
                depth_image = cv2.resize(depth_image, (200, 120), interpolation=cv2.INTER_AREA)

                frame_timestamp = frames.timestamp

                if t_tmp == 0:
                    t_tmp = frame_timestamp
                frame_timestamp = frame_timestamp - t_tmp
                print('\r',
                      "\033[32mWriting frame " + str(i) + " timestamp " + str(frame_timestamp) + "\033[0m", end='')
                self.result['t'][i] = frame_timestamp

                vmap = depth_image - t_vmap
                if t1 != 0:
                    vmap = vmap / (frame_timestamp - t1)
                t1 = frame_timestamp

                self.result['y'][i, :] = depth_image
                t_vmap = depth_image

                csi_index = np.searchsorted(self.csi_stream['time'], frame_timestamp)
                csi_chunk = self.csi_stream['csi'][csi_index: csi_index + 33, :, :, 0]
                csi_dyn_chunk = make_dataset_preprocess_csi.windowed_dynamic(csi_chunk).reshape(33, 90).T

                self.result['x'][i, 0, :, :] = np.abs(csi_dyn_chunk)
                self.result['x'][i, 1, :, :] = np.angle(csi_dyn_chunk)

        except RuntimeError:
            print("Read finished!")

        finally:
            self.result['y'][0] = self.result['y'][1]
            self.video_stream.stop()

            path = '../dataset/' + save_name

            if not os.path.exists(path):
                os.makedirs(path)

            np.save(path + '_x.npy', self.result['x'])
            np.save(path + '_y.npy', self.result['y'])
            np.save(path + '_t.npy', self.result['t'])

            print("\nAll chunks saved!")


if __name__ == '__main__':

    paths = ['../sense/1213/1213env.bag', '../npsave/1213/1213A00-csio.npy']
    mkdata = MyDataMaker(paths, 300)
    mkdata.save('1213/make00/00')
