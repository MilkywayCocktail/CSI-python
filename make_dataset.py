
import pyrealsense2 as rs
import cv2
import csi_loader
import numpy as np
import sys
import os
import pycsi
from tqdm import tqdm
from datetime import datetime


def my_filter(frame):
    """
    Filter used for depth images.\n
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
    """
    Hide print lines.\n
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class MyConfigsDM(pycsi.MyConfigs):
    def __init__(self, img_size=(128, 128), sample_length=100, *args, **kwargs):
        pycsi.MyConfigs.__init__(self, *args, **kwargs)
        self.img_size = img_size
        self.sample_length = sample_length


class MyDataMaker:

    # Generates images, CSI

    def __init__(self, configs: MyConfigsDM, paths: list, total_frames: int):
        """
        :param configs: MyConfigsDM
        :param paths: [bag path, local timestamp path, CSI path, (label path)]
        :param total_frames: Full length of bag file
        """

        # local_timestamps is used as standard.

        self.configs = configs
        self.paths = paths
        self.total_frames = total_frames
        self.local_timestamps = self.__load_local_timestamps__()
        self.video_stream = self.__setup_video_stream__()
        self.csi_stream = self.__setup_csi_stream__()
        self.result = self.__init_data__()
        self.cal_cam = False

    def __len__(self):
        return self.total_frames

    def __setup_video_stream__(self):
        # timestamps is the local time; works as reference time
        print('Setting camera stream...', end='')
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.paths[0], False)
        config.enable_all_streams()
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)
        print('Done')

        return pipeline

    def __load_local_timestamps__(self):
        local_tf = open(self.paths[1], mode='r', encoding='utf-8')
        local_time = np.array(local_tf.readlines())
        for i in range(len(local_time)):
            local_time[i] = datetime.timestamp(datetime.strptime(local_time[i].strip(), "%Y-%m-%d %H:%M:%S.%f"))
        local_tf.close()
        return local_time.astype(np.float64)

    def __setup_csi_stream__(self):
        print('Setting CSI stream...')
        _csi = pycsi.MyCsi(self.configs, 'CSI', self.paths[2])
        _csi.load_data(remove_sm=True)

        csi_abs_tf = open(self.paths[3], mode='r', encoding='utf-8')
        _csi_abs_timestamps = np.array(csi_abs_tf.readlines())
        for i in range(len(_csi_abs_timestamps)):
            _csi_abs_timestamps[i] = datetime.timestamp(
                datetime.strptime(_csi_abs_timestamps[i].strip(), "%Y-%m-%d %H:%M:%S.%f"))
        csi_abs_tf.close()

        _csi.abs_timestamps = _csi_abs_timestamps.astype(np.float64)

        return _csi  # Pre-calibrated absolute CSI timestamp

    def __init_data__(self):
        # img_size = (width, height)
        csi = np.zeros((self.total_frames, 2, 90, self.configs.sample_length))
        images = np.zeros((self.total_frames, self.configs.img_size[1], self.configs.img_size[0]))
        timestamps = np.zeros(self.total_frames)
        indices = np.zeros(self.total_frames, dtype=int)
        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices}

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
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    break

            except RuntimeError:
                print("Read finished!")

            finally:
                self.video_stream.stop()
                if save_flag is True:
                    videowriter.release()

    def export_image(self, mode='depth', show_img=False):
        try:
            tqdm.write('Starting exporting image...')
            for i in tqdm(range(self.total_frames)):
                image, frame_timestamp = self.__get_image__(mode=mode)

                self.result['tim'][i] = frame_timestamp
                image = cv2.resize(image, self.configs.img_size, interpolation=cv2.INTER_AREA)
                self.result['img'][i, ...] = image

                if show_img is True:
                    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Image', image)
                    key = cv2.waitKey(33) & 0xFF
                    if key == ord('q'):
                        break

        except RuntimeError:
            pass

        finally:
            self.video_stream.stop()
            self.calibrate_camtime()

    def export_csi(self, dynamic_csi=True, pick_tx=0):
        """
        Finds csi packets according to the timestamps of images.\n
        Requires export_image.\n
        """
        tqdm.write('Starting exporting CSI...')

        for i in tqdm(range(self.total_frames)):

            csi_index = np.searchsorted(self.csi_stream.abs_timestamps, self.result['tim'][i])
            self.result['ind'][i] = csi_index
            csi_chunk = self.csi_stream.csi[csi_index: csi_index + self.configs.sample_length, :, :, pick_tx]

            if dynamic_csi is True:
                csi_chunk = self.windowed_dynamic(csi_chunk).reshape(self.configs.sample_length, 90).T
            else:
                csi_chunk = csi_chunk.reshape(self.configs.sample_length, 90).T

            # Store in two channels
            self.result['csi'][i, 0, :, :] = np.abs(csi_chunk)
            self.result['csi'][i, 1, :, :] = np.angle(csi_chunk)

    def slice_by_label(self):
        """
        Trim non-labeled segments out of dataset.\n
        Executed after exporting x and y.\n
        :return: sliced results
        """
        print('Slicing...', end='')
        labels = []
        with open(self.paths[4]) as f:
            for i, line in enumerate(f):
                if i > 0:
                    labels.append([eval(line.split(',')[0]), eval(line.split(',')[1])])

        labels = np.array(labels)

        rel_timestamps = self.result['tim'] - self.result['tim'][0]
        full = list(range(self.total_frames))
        ids = []
        for (start, end) in labels:
            start_id = np.searchsorted(rel_timestamps, start)
            end_id = np.searchsorted(rel_timestamps, end)
            ids.extend(full[start_id:end_id])

        self.total_frames = len(ids)

        for key in self.result.keys():
            self.result[key] = self.result[key][ids]
        print('Done')

    @staticmethod
    def windowed_dynamic(in_csi):
        # in_csi = np.squeeze(in_csi)
        phase_diff = in_csi * in_csi[..., 0][..., np.newaxis].conj().repeat(3, axis=2)
        static = np.mean(phase_diff, axis=0)
        dynamic = phase_diff - static
        return dynamic

    def calibrate_camtime(self):
        """
        Calibrate camera timestamps against local timestamps. All timestamps are absolute.\n
        :return: result['tim']
        """
        print('Calibrating camera time against local time file...', end='')
        cvt = datetime.fromtimestamp

        if self.cal_cam is False:
            temp_lag = np.zeros(self.total_frames)
            for i in range(self.total_frames):
                temp_lag[i] = self.result['tim'][i] - self.local_timestamps[i]

            lag = np.mean(temp_lag)
            print('lag=', lag)

            for i in range(self.total_frames):
                self.result['tim'][i] = self.result['tim'][i] - lag
            self.cal_cam = True
        print('Done')

    def depth_mask(self):
        tqdm.write("Masking...")
        median = np.median(self.result['img'], axis=0)
        threshold = median * 0.5

        for i in tqdm(range(len(self.result['img']))):
            mask = self.result['img'][i] < threshold
            masked = self.result['img'][i] * mask
            self.result['img'][i] = masked

    def compress_image(self):
        print("Compressing...", end='')
        self.result['img'] = self.result['img'].astype(np.uint16)
        print("Done")

    def playback_image(self, save_path=None, save_name='new.avi'):
        print("Reading...", end='')
        save_flag = False
        if save_path is not None and save_name is not None:
            save_flag = True
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            img_size = (self.result['img'].shape[1], self.result['img'].shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videowriter = cv2.VideoWriter(save_path + save_name, fourcc, 10, img_size)

        for i in range(self.total_frames):
            if self.result['img'].dtype == 'uint16':
                image = (self.result['img'][i]/256).astype('uint8')
                cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Image', image)
                key = cv2.waitKey(33) & 0xFF

            else:
                image = cv2.convertScaleAbs(self.result['img'][i], alpha=0.02)
                cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Image', image)
                key = cv2.waitKey(33) & 0xFF

            if save_flag is True:
                videowriter.write(image)
        print("Done")

        if save_flag is True:
            videowriter.release()

    def save_dataset(self, save_path, save_name, *args):
        print("Saving...", end='')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for key in args:
            if key in self.result.keys():
                np.save(os.path.join(save_path, save_name + '_' + key + '.npy'), self.result[key])
        print("Done")


if __name__ == '__main__':

    date = '0307'
    sub = '04'
    length = 3000

    path = ['../sense/' + date + '/' + sub + '.bag',
            '../sense/' + date + '/' + sub + '_timestamps.txt',
            '../npsave/' + date + '/' + date + 'A' + sub + '-csio.npy',
            '../data/' + date + '/csi' + date + 'A' + sub + '_time_mod.txt',
            '../sense/' + date + '/' + sub + '_labels.csv']

    configs = MyConfigsDM()

    mkdata = MyDataMaker(configs=configs, paths=path, total_frames=length)
    mkdata.csi_stream.extract_dynamic(mode='overall-divide', ref='tx', reference_antenna=1)
    mkdata.csi_stream.extract_dynamic(mode='highpass')
    mkdata.export_image(show_img=False)
    mkdata.depth_mask()
    #print(mkdata.csi_stream.abs_timestamps)
    #print(mkdata.local_timestamps)
    #print(mkdata.result['tim'])
    mkdata.export_csi(dynamic_csi=False, pick_tx=0)
    mkdata.slice_by_label()
    #mkdata.playback_image()
    mkdata.save_dataset('../dataset/0307/make04', sub + '_div', 'csi', 'img')

