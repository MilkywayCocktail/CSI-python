
import pyrealsense2 as rs
import cv2
import csi_loader
import numpy as np
import sys
import os
from tqdm import tqdm
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

    def __init__(self, paths: list, total_frames: int, img_size: tuple, sample_length=33):
        """
        :param paths: [bag path, local timestamp path, CSI path, CSI timestamp path]
        :param total_frames: Preset length of camera record
        :param img_size: resize camera images
        :param sample_length: how many packets in one CSI sample
        """

        self.paths = paths
        self.total_frames = total_frames
        self.img_size = img_size
        self.sample_length = sample_length
        self.video_stream, self.timestamps = self.__setup_video_stream__()
        self.csi_stream = self.__setup_csi_stream__()
        self.result = self.__init_data__()

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

        local_tf = open(self.paths[1], mode='r', encoding='utf-8')
        local_time = local_tf.readlines()
        local_tf.close()
        print('Done')

        return pipeline, local_time

    def __setup_csi_stream__(self):
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
            csi_timestamp[i] = datetime.timestamp(datetime.strptime(csi_time[i].strip(), "%Y-%m-%d %H:%M:%S.%f"))
        csi_tf.close()

        print('Done')
        return {'csi': csi,
                'time': csi_timestamp}  # Calibrated CSI timestamp

    def __init_data__(self):
        # img_size = (width, height)
        csi = np.zeros((self.total_frames, 2, 90, self.sample_length))
        images = np.zeros((self.total_frames, self.img_size[1], self.img_size[0]))
        timestamps = np.zeros(self.total_frames)
        indices = np.zeros(self.total_frames)
        coordinates = np.zeros((self.total_frames, 3))
        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices, 'cod': coordinates}

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
                image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
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
            tqdm.write('Image exported!')
            print('Calibrating camera time against local time file...', end='')
            temp_lag = np.zeros(self.total_frames)
            for i in range(self.total_frames):
                temp_lag[i] = self.calculate_timedelta(self.result['tim'][i], self.timestamps[i])

            lag = np.mean(temp_lag)
            print('lag=', lag)

            for i in range(self.total_frames):
                self.result['tim'][i] = self.result['tim'][i] - lag

            print('Done')
            self.video_stream.stop()

    def export_coordinate(self, show_img=False, min_area=50):
        """
        Requires export_image and depth_mask!\n
        :param show_img: whether to show the coordinate with the image
        :param min_area: a threshold set to filter out wrong bounding boxes
        """
        tqdm.write('Starting exporting coordinate...')
        areas = np.zeros(self.total_frames)
        for i in tqdm(range(self.total_frames)):
            img = None
            (T, timg) = cv2.threshold(self.result['img'][i].astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                contour = max(contours, key=lambda x: cv2.contourArea(x))
                areas[i] = cv2.contourArea(contour)

                if areas[i] < min_area:
                    print(areas[i])
                    self.result['cod'][i] = np.array([self.img_size[1]//2, self.img_size[0]//2, 0])
                    if show_img is True:
                        img = self.result['img'][i]
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    xc, yc = int(x + w / 2), int(y + h / 2)
                    self.result['cod'][i] = np.array([xc, yc, self.result['img'][i][yc, xc]])
                    if show_img is True:
                        img = cv2.rectangle(cv2.cvtColor(np.float32(self.result['img'][i]), cv2.COLOR_GRAY2BGR),
                                            (x, y),
                                            (x + w, y + h),
                                            (0, 255, 0), 1)
                        img = cv2.circle(img, (xc, yc), 1, (0, 0, 255), 4)
            else:
                img = self.result['img'][i]
                self.result['cod'][i] = np.array([self.img_size[1]//2, self.img_size[0]//2, 0])

            if show_img is True:
                cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Image', img)
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    break

    def export_csi(self, dynamic_csi=True):
        """
        Requires export_image
        """
        tqdm.write('Starting exporting CSI...')

        for i in tqdm(range(self.total_frames)):

            csi_index = np.searchsorted(self.csi_stream['time'], self.result['tim'][i])
            self.result['ind'][i] = csi_index
            csi_chunk = self.csi_stream['csi'][csi_index: csi_index + self.sample_length, :, :, 0]

            if dynamic_csi is True:
                csi_chunk = self.windowed_dynamic(csi_chunk).reshape(self.sample_length, 90).T
            else:
                csi_chunk = csi_chunk.reshape(self.sample_length, 90).T

            # Store in two channels
            self.result['csi'][i, 0, :, :] = np.abs(csi_chunk)
            self.result['csi'][i, 1, :, :] = np.angle(csi_chunk)

    def slice_by_label(self, labels: list):
        print('Slicing...', end='')
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

    @staticmethod
    def calculate_timedelta(time1, time2):

        tmp = []
        for t in (time1, time2):
            if isinstance(t, str):
                tmp.append(datetime.timestamp(datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S.%f")))
            elif isinstance(t, float):
                tmp.append(t)

        time_delta = tmp[0] - tmp[1]
        return time_delta

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
            if key in ('img', 'csi', 'tim', 'ind', 'cod'):
                np.save(os.path.join(save_path, save_name + '_' + key + '.npy'), self.result[key])
        print("Done")


if __name__ == '__main__':

    sub = '02'
    length = 1800

    path = [os.path.join('../sense/0124', sub + '.bag'),
            os.path.join('../sense/0124', sub + '_timestamps.txt'),
            os.path.join('../npsave/0124', '0124A' + sub + '-csio.npy'),
            os.path.join('../data/0124', 'csi0124A' + sub + '_time_mod.txt')]

    label02 = [(4.447, 7.315), (8.451, 11.352), (14.587, 18.59), (20.157, 22.16),
               (25.496, 29.397), (30.999, 33.767), (36.904, 40.473), (41.674, 44.108),
               (47.244, 51.046), (53.615, 55.983)]

    label03 = [(8.085, 10.987), (12.422, 14.79), (18.959, 21.862), (21.962, 24.963),
               (28.769, 31.902), (32.669, 36.206), (39.675, 42.611), (43.645, 47.147),
               (50.751, 53.485), (55.187, 57.988)]

    mkdata = MyDataMaker(path, length, (128, 128), sample_length=100)
    mkdata.export_image(show_img=False)
    mkdata.depth_mask()
    #mkdata.export_coordinate(show_img=True, min_area=1000)
    mkdata.export_csi()
    mkdata.slice_by_label(label02)
    mkdata.playback_image()
    mkdata.save_dataset('../dataset/0124/make02', sub + '_dyn', 'ind', 'csi', 'img', 'tim')

