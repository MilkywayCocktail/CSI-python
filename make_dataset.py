
import pyrealsense2 as rs
import cv2
import csi_loader
import numpy as np
import sys
import os
import tqdm
from datetime import datetime
import datetime as dt


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

    def __init__(self, paths: list, total_frames: int, img_size: tuple):
        # paths = [bag path, local timestamp path, CSI path, CSI timestamp path]
        self.paths = paths
        self.total_frames = total_frames
        self.img_size = img_size
        self.video_stream, self.timestamps = self.__setup_video_stream__()
        self.csi_stream = self.__setup_csi_stream__()
        self.result = self.__init_data__()

    def __len__(self):
        return self.total_frames

    def __setup_video_stream__(self):
        print('Setting camera stream...', end='')
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.paths[0], False)
        config.enable_all_streams()
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)

        local_tf = open(self.paths[1], mode='r', encoding='utf-8')
        local_time = local_tf.readlines()
        local_tf.close()
        print('Done')

        return pipeline, local_time

    def __setup_csi_stream__(self):
        # Please make sure you have calibrated CSI timestamps against local time.
        print('Setting CSI stream...', end='')
        file = os.path.basename(self.paths[2])

        if file[-3:] == 'npy':
            with HiddenPrints():
                csi, _, i, j = csi_loader.load_npy(self.paths[2])

        elif file[-3:] == 'dat':
            with HiddenPrints():
                csi, _ = csi_loader.dat2npy(self.paths[2], '', autosave=False)

        else:
            print("File type not supported")
            return 1

        csi_tf = open(self.paths[3], mode='r', encoding='utf-8')
        csi_time = csi_tf.readlines()
        csi_timestamp = np.zeros(len(csi_time))
        for i in range(len(csi_time)):
            csi_timestamp[i] = datetime.timestamp(datetime.strptime(csi_time[i].strip(), "%Y/%m/%d %H:%M:%S.%f"))
        csi_tf.close()

        print('Done')
        return {'csi': csi,
                'time': csi_timestamp}  # absolute timestamp

    def __init_data__(self):
        # img_size = (width, height)
        x_csi = np.zeros((self.total_frames, 2, 90, 33))
        y_dmap = np.zeros((self.total_frames, self.img_size[1], self.img_size[0]))
        t_list = np.zeros(self.total_frames)
        index_list = np.zeros(self.total_frames)
        return {'x': x_csi, 'y': y_dmap, 't': t_list, 'i': index_list}

    def __get_image__(self, mode):
        frames = self.video_stream.wait_for_frames()

        if mode == 'depth':
            depth_frame = frames.get_depth_frame()
            frame_timestamp = depth_frame.get_timestamp() / 1000  # Camera timestamps in us
            if not depth_frame:
                eval('continue')
            depth_frame = my_filter(depth_frame)
            image = np.asanyarray(depth_frame.get_data())

        elif mode == 'color':
            color_frame = frames.get_color_frame()
            frame_timestamp = color_frame.get_timestamp() / 1000
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
            videowriter = cv2.VideoWriter(save_path + save_name, fourcc, 10, img_size)

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

    def export_y(self, mode='depth', show_img=True):
        #print('Starting exporting y...')
        try:
            for i in tqdm.tqdm(range(self.total_frames)):
                image, frame_timestamp = self.__get_image__(mode=mode)

                self.result['t'][i] = frame_timestamp
                image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
                self.result['y'][i, ...] = image

                if show_img is True:
                    cv2.imshow('Image', image)
                    key = cv2.waitKey(1) & 0xFF

        except RuntimeError:
            pass

        finally:
            print('y exported!')
            self.video_stream.stop()

    def export_x(self, dynamic_csi=True):
        print('Starting exporting x...')
        temp_lag = np.zeros(self.total_frames)

        print('Calibrating camera time against local time file...', end='')
        for i in range(self.total_frames):
            cam_time = datetime.fromtimestamp(self.result['t'][i])
            local_time = datetime.strptime(self.timestamps[i].strip(), "%Y-%m-%d %H:%M:%S.%f")
            temp_lag[i] = (cam_time - local_time).seconds

        lag = dt.timedelta(seconds=np.mean(temp_lag))

        for i in range(self.total_frames):
            self.result['t'][i] = datetime.timestamp(datetime.fromtimestamp(self.result['t'][i]) - lag)
        print('Done')

        print('Matching CSI frames...', end='')
        for i in range(self.total_frames):
            print(self.result['t'][i])

            csi_index = np.searchsorted(self.csi_stream['time'], self.result['t'][i])
            self.result['i'][i] = csi_index
            print(csi_index)
            csi_chunk = self.csi_stream['csi'][csi_index: csi_index + 33, :, :, 0]
            print(csi_chunk.shape)

            if dynamic_csi is True:
                csi_chunk = self.windowed_dynamic(csi_chunk).reshape(33, 90).T
            else:
                csi_chunk = csi_chunk.reshape(33, 90).T

            # Store in two channels
            self.result['x'][i, 0, :, :] = np.abs(csi_chunk)
            self.result['x'][i, 1, :, :] = np.angle(csi_chunk)

        print('Done')
        print('x exported!')

    @staticmethod
    def windowed_dynamic(in_csi):
        # in_csi = np.squeeze(in_csi)
        phase_diff = in_csi * in_csi[..., 0][..., np.newaxis].conj().repeat(3, axis=2)
        static = np.mean(phase_diff, axis=0)
        dynamic = phase_diff - static
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
            if self.result['y'].dtype == 'uint16':
                image = (self.result['y'][i]/256).astype('uint8')
                cv2.imshow('Image', image)

            else:
                image = cv2.convertScaleAbs(self.result['y'][i], alpha=0.02)
                cv2.imshow('Image', image)

            key = cv2.waitKey(1) & 0xFF

            if save_flag is True:
                videowriter.write(image)

        print("Read finished!")
        if save_flag is True:
            videowriter.release()

    def save_dataset(self, save_path='../dataset/', save_name=None):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(os.path.join(save_path, save_name + '_x.npy'), self.result['x'])
        np.save(os.path.join(save_path, save_name + '_y.npy'), self.result['y'])
        np.save(os.path.join(save_path, save_name + '_t.npy'), self.result['t'])

        print("All chunks saved!")


if __name__ == '__main__':

    sub = '01'
    length = 3000

    path = [os.path.join('../sense/0124', sub + '.bag'),
            os.path.join('../sense/0124', sub + '_timestamps.txt'),
            os.path.join('../npsave/0124', '0124A' + sub + '-csio.npy'),
            os.path.join('../data/0124', 'csi0124A' + sub + '_time_mod.txt')]
    mkdata = MyDataMaker(path, length, (128, 128))
    mkdata.export_y(show_img=False)
    mkdata.export_x()
    #print(mkdata.result['t'])
    #mkdata.depth_mask()
    #mkdata.playback_y()
    #mkdata.save_dataset(save_path=os.path.join('../dataset/0124', 'make' + sub), save_name=sub)

