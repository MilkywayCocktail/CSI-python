
import pyrealsense2 as rs
import cv2
import csi_loader
import numpy as np
import sys
import os
import tqdm


def my_filter(frame):
    """
    Filter used for depth images
    """
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
        config.enable_all_streams()
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)

        return pipeline

    def __setup_csi_stream__(self):
        file = os.path.basename(self.paths[1])
        if file[-3:] == 'npy':
            with HiddenPrints():
                csi, timestamps, i, j = csi_loader.load_npy(self.paths[1])

        elif file[-3:] == 'dat':
            with HiddenPrints():
                csi, timestamps = csi_loader.dat2npy(self.paths[1], '', autosave=False)

        else:
            print("File type not supported")
            return 1

        return {'csi': csi.swapaxes(1, 3),
                'time': timestamps - timestamps[0]}

    def __init_data__(self):
        x_csi = np.zeros((self.total_frames, 2, 90, 33))
        y_vmap = np.zeros((self.total_frames, 120, 200))
        t_list = np.zeros(self.total_frames)
        index_list = np.zeros(self.total_frames)
        return {'x': x_csi, 'y': y_vmap, 't': t_list, 'i': index_list}

    def __get_image__(self, mode):
        frames = self.video_stream.wait_for_frames()
        frame_timestamp = frames.timestamp
        if mode == 'depth':
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                eval('continue')
            depth_frame = my_filter(depth_frame)
            image = np.asanyarray(depth_frame.get_data())

        elif mode == 'color':
            color_frame = frames.get_color_frame()
            if not color_frame:
                eval('continue')
            image = np.asanyarray(color_frame.get_data())

        return image, frame_timestamp

    def playback_raw(self, mode='depth', save_path=None, save_name='new.avi'):
        save_flag = False
        if save_path is not None and save_name is not None:
            save_flag = True
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if mode == 'depth':
                img_size = (848, 480)
            elif mode == 'color':
                img_size = (1280, 720)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videowriter = cv2.VideoWriter(save_path + save_name, fourcc, 30, img_size)

        while True:
            try:
                image, _ = self.__get_image__(mode=mode)
                if mode == 'depth':
                    image = cv2.convertScaleAbs(image, alpha=0.02)
                if save_flag is True:
                    videowriter.write(image)
                cv2.imshow('Image', image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            except RuntimeError:
                print("Read finished!")

            finally:
                self.video_stream.stop()
                if save_flag is True:
                    videowriter.release()

    def match_xy(self, mode='depth', resize=None, dynamic_csi=True):
        try:
            print('Starting making...')

            for i in tqdm.tqdm(range(self.total_frames)):
                image, frame_timestamp = self.__get_image__(mode=mode)

                self.result['t'][i] = frame_timestamp

                if isinstance(resize, set) and len(resize)==2:
                    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
                self.result['y'][i, ...] = image

                csi_index = np.searchsorted(self.csi_stream['time'], frame_timestamp)
                self.result['i'][i] = csi_index
                csi_chunk = self.csi_stream['csi'][csi_index: csi_index + 33, :, :, 0]
                if dynamic_csi is True:
                    csi_chunk = self.windowed_dynamic(csi_chunk).reshape(33, 90).T
                else:
                    csi_chunk = csi_chunk.reshape(33, 90).T

                # Store in two channels
                self.result['x'][i, 0, :, :] = np.abs(csi_chunk)
                self.result['x'][i, 1, :, :] = np.angle(csi_chunk)

        except RuntimeError:
            print("Match finished!")

        finally:
            self.video_stream.stop()

    @staticmethod
    def windowed_dynamic(in_csi):
        # in_csi = np.squeeze(in_csi)
        phase_diff = in_csi * in_csi[..., 0][..., np.newaxis].conj().repeat(3, axis=2)
        static = np.mean(phase_diff, axis=0)
        dynamic = in_csi - static
        return dynamic

    def depth_mask(self):
        median = np.median(self.result['y'], axis=0)
        threshold = median * 0.5

        for i in tqdm.tqdm(range(len(self.result['y']))):
            mask = self.result['y'][i] < threshold
            masked = self.result['y'][i] * mask
            self.result['y'][i] = masked

        print("Mask finished!")

    def compress_y(self):
        self.result['y'] = self.result['y'].astype(np.uint16)
        print("Compress finished!")

    def playback_y(self, save_path=None, save_name='new.avi'):
        save_flag = False
        if save_path is not None and save_name is not None:
            save_flag = True
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            img_size = (self.result['y'].shape[1], self.result['y'].shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videowriter = cv2.VideoWriter(save_path + save_name, fourcc, 30, img_size)

        for i in tqdm.tqdm(range(self.total_frames)):
            if self.result['y'].dypte == 'uint16':
                image = (self.result['y'][i]/256).astype('uint8')
                cv2.imshow('Image', image)
            else:
                image = cv2.convertScaleAbs(self.result['y'][i], alpha=0.02)
                cv2.imshow('Image', image)
            if save_flag is True:
                videowriter.write(image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            print("Read finished!")
            self.video_stream.stop()
            if save_flag is True:
                videowriter.release()

    def save_dataset(self, save_path='../dataset/', save_name=None):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(save_path + save_name + '_x.npy', self.result['x'])
        np.save(save_path + save_name + '_y.npy', self.result['y'])
        np.save(save_path + save_name + '_t.npy', self.result['t'])

        print("All chunks saved!")


if __name__ == '__main__':

    paths = ['../sense/1213/1213env.bag', '../npsave/1213/1213A00-csio.npy']
    mkdata = MyDataMaker(paths, 300)
    mkdata.match_xy()
    mkdata.depth_mask()
    mkdata.save_dataset(save_path='../dataset/1213/make00', save_name='00')
