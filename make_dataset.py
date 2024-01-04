
import pyrealsense2 as rs
import cv2
import csi_loader
import numpy as np
import sys
import os
import pycsi
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datetime import datetime
from IPython.display import display, clear_output
from matplotlib.widgets import Slider


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

    def __init__(self, configs: MyConfigsDM, paths: dict, total_frames: int):
        """
        :param configs: MyConfigsDM
        :param paths: {bag, local timestamp, CSI path, csi timestamp, (label)}
        :param total_frames: Full length of bag file
        """

        # local_timestamps is used as standard.

        self.configs = configs
        self.paths = paths
        self.total_frames = total_frames
        self.raw_csi = None
        self.local_timestamps = self.__load_local_timestamps__()
        self.video_stream = self.__setup_video_stream__()
        self.csi_stream = self.__setup_csi_stream__()
        self.result = self.__init_data__()
        self.cal_cam = False
        self.camtime_delta = 0.
        self.jupyter = False

    def __len__(self):
        return self.total_frames

    def __setup_video_stream__(self):
        # timestamps is the local time; works as reference time
        print('Setting camera stream...', end='')
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.paths['bag'], False)
        config.enable_all_streams()
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)
        print('Done')

        return pipeline

    def __load_local_timestamps__(self):
        local_tf = open(self.paths['lt'], mode='r', encoding='utf-8')
        local_time = np.array(local_tf.readlines())
        for i in range(len(local_time)):
            local_time[i] = datetime.timestamp(datetime.strptime(local_time[i].strip(), "%Y-%m-%d %H:%M:%S.%f"))
        local_tf.close()
        return local_time.astype(np.float64)

    def __setup_csi_stream__(self):
        print('Setting CSI stream...')
        _csi = pycsi.MyCsi(self.configs, 'CSI', self.paths['csi'])
        _csi.load_data(remove_sm=True)

        csi_abs_tf = open(self.paths['ct'], mode='r', encoding='utf-8')
        _csi_abs_timestamps = np.array(csi_abs_tf.readlines())
        for i in range(len(_csi_abs_timestamps)):
            _csi_abs_timestamps[i] = datetime.timestamp(
                datetime.strptime(_csi_abs_timestamps[i].strip(), "%Y-%m-%d %H:%M:%S.%f"))
        csi_abs_tf.close()

        _csi.abs_timestamps = _csi_abs_timestamps.astype(np.float64)
        
        self.raw_csi = _csi

        return _csi  # Pre-calibrated absolute CSI timestamp

    def __init_data__(self):
        # img_size = (width, height)
        if self.raw_csi:
            csi = np.zeros((self.total_frames, self.configs.sample_length, 30, 3))
        else:
            csi = np.zeros((self.total_frames, 2, 90, self.configs.sample_length))
        images = np.zeros((self.total_frames, self.configs.img_size[1], self.configs.img_size[0]))
        timestamps = np.zeros(self.total_frames)
        indices = np.zeros(self.total_frames, dtype=int)
        locations = np.zeros((self.total_frames, 4), dtype=float)
        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices, 'loc': locations}

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
            
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1) 
        while True:
            try:
                image, _ = self.__get_image__(mode=mode)
                if mode == 'depth':
                    image = cv2.convertScaleAbs(image, alpha=0.02)
                if save_flag is True:
                    videowriter.write(image)
                # cv2.imshow('Image', image)
                # key = cv2.waitKey(33) & 0xFF
                # if key == ord('q'):
                #     break
                if self.jupyter:
                    clear_output(wait = True)
                    plt.clf()
                    plt.imshow(image)
                    plt.title(f"Image {i} of {len(imgs)}")
                    #display(plt.gcf())'
                    plt.axis('off')
                    plt.show()
                    plt.pause(0.1)
                else:
                    plt.imshow(image)
                    plt.title("Raw Image")
                    plt.pause(0.1)
                    plt.clf()

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

                if show_img:
                    # cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
                    # cv2.imshow('Image', image)
                    # key = cv2.waitKey(33) & 0xFF
                    # if key == ord('q'):
                    #     break
                    # cv2.destroyAllWindows()
                    if self.jupyter:
                        clear_output(wait = True)
                        plt.clf()
                        plt.imshow(image)
                        plt.title(f"Image {i} of {len(imgs)}")
                        #display(plt.gcf())'
                        plt.axis('off')
                        plt.show()
                        plt.pause(0.1)
                    else:
                        plt.imshow(image)
                        plt.title("Raw Image")
                        plt.pause(0.1)
                        plt.clf()

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
            if dynamic_csi:
                csi_chunk = self.windowed_dynamic(csi_chunk).reshape(self.configs.sample_length, 90).T
            else:
                csi_chunk = csi_chunk.reshape(self.configs.sample_length, 90).T

            # Store in two channels
            self.result['csi'][i, 0, :, :] = np.abs(csi_chunk)
            self.result['csi'][i, 1, :, :] = np.angle(csi_chunk)
                
    def reset_csi(self):
        print("Restoring raw csi!")
        self.csi_stream = self.raw_csi

    def slice_by_label(self):
        """
        Trim non-labeled segments out of dataset.\n
        Executed after exporting x and y.\n
        Labels' timestamps are millisecond-level.\n
        :return: sliced results
        """
        print('Slicing...', end='')
        labels = []
        with open(self.paths['label']) as f:
            for i, line in enumerate(f):
                if i > 0:
                    #if eval(line.split(',')[2][2:]) not in (-2, 2):
                    labels.append([eval(line.split(',')[0]) * 1e-3, eval(line.split(',')[1]) * 1e-3,
                                   eval(line.split(',')[2]), eval(line.split(',')[3]), eval(line.split(',')[4]),
                                   eval(line.split(',')[5])])

        labels = np.array(labels)

        # Absolute timestamps or relative timestamps?
        # rel_timestamps = self.result['tim'] - self.result['tim'][0]
        full = list(range(self.total_frames))
        ids = []
        locations = []
        for (start, end, x0, y0, x1, y1) in labels:
            start_id = np.searchsorted(self.result['tim'], start - self.camtime_delta)
            end_id = np.searchsorted(self.result['tim'], end - self.camtime_delta)
            ids.extend(full[start_id:end_id])
            locations.extend([x0, y0, x1, y1] * len(full[start_id:end_id]))

        self.total_frames = len(ids)

        for key in self.result.keys():
            self.result[key] = self.result[key][ids]
        self.result['loc'] = np.array(locations).reshape(-1, 4)
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

            self.camtime_delta = np.mean(temp_lag)
            print('lag=', self.camtime_delta)

            for i in range(self.total_frames):
                self.result['tim'][i] = self.result['tim'][i] - self.camtime_delta
            self.cal_cam = True
        print('Done')

    def depth_mask(self, threshold=0.5):
        tqdm.write("Masking...")
        median = np.median(self.result['img'], axis=0)
        threshold = median * threshold
        for i in tqdm(range(len(self.result['img']))):
            mask = self.result['img'][i] < threshold
            masked = self.result['img'][i] * mask
            self.result['img'][i] = masked

    def compress_image(self):
        print("Compressing...", end='')
        self.result['img'] = self.result['img'].astype(np.uint16)
        print("Done")

    def playback_image(self, save_path=None, save_name='new.avi'):
        print("Reading playback...", end='')
        save_flag = False
        if save_path is not None and save_name is not None:
            save_flag = True
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            img_size = (self.result['img'].shape[1], self.result['img'].shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videowriter = cv2.VideoWriter(save_path + save_name, fourcc, 10, img_size)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1) 
        for i in range(self.total_frames):
            if self.result['img'].dtype == 'uint16':
                image = (self.result['img'][i]/256).astype('uint8')
            else:
                image = cv2.convertScaleAbs(self.result['img'][i], alpha=0.02)

            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
            # cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Image', image)
            # key = cv2.waitKey(33) & 0xFF
            # cv2.destroyAllWindows()
            if self.jupyter:
                clear_output(wait = True)
                plt.clf()
                plt.imshow(image)
                plt.title(f"Image {i} of {len(imgs)}")
                #display(plt.gcf())'
                plt.axis('off')
                plt.show()
                plt.pause(0.1)
            else:
                plt.imshow(image)
                plt.title("Raw Image")
                plt.pause(0.1)
                plt.clf()

            if save_flag:
                videowriter.write(image)
        print("Done")

        if save_flag is True:
            videowriter.release()

    def lookup_image(self):
        print("\033[32mLOOKUP MODE" +
              "\033[0m")
        while True:
            print("Please input a timestamp of the image to show:\n",
                  "- Enter exit to quit")

            accept_string = input()

            if accept_string == 'exit':
                print("\033[32mExiting Lookup Mode...\033[0m")
                break

            else:
                timestamp = eval(accept_string)
                try:
                    t = datetime.fromtimestamp(timestamp)
                    print(t.strftime("%Y-%m-%d %H:%M:%S.%f"))
                    ind = np.searchsorted(self.result['tim'], timestamp)
                    print(ind)
                    plt.imshow(self.result['img'][ind])
                    plt.show()
                except:
                    pass

    def save_dataset(self, save_path, save_name, *args):
        print("Saving...", end='')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for key in args:
            if key in self.result.keys():
                np.save(os.path.join(save_path, save_name + '_' + key + '.npy'), self.result[key])
        print("Done")


if __name__ == '__main__':

    date = '0509'

    subs = [('01', 5400), ('02', 5000), ('03', 5100), ('04', 5100)]

    configs = MyConfigsDM(img_size=(226, 128))
    configs.tx_rate = 0x1c113
    configs.ntx = 3

    for (sub, length) in subs:

        path = {'bag': f"F:/Research/pycsi/sense/{date}/{sub}.bag",
                'lt': f"F:/Research/pycsi/sense/{date}/{sub}_timestamps.txt",
                'csi': f"../npsave/{date}/{date}A{sub}-csio.npy",
                'ct': f"../data/{date}/csi{date}A{sub}_time_mod.txt",
                'label': f"../sense/{date}/{sub}_labels.csv"
               }

        mkdata = MyDataMaker(configs=configs, paths=path, total_frames=length)
        mkdata.csi_stream.extract_dynamic(mode='overall-divide', ref='tx', ref_antenna=1)
        mkdata.csi_stream.extract_dynamic(mode='highpass')
        mkdata.export_image(show_img=False)
        mkdata.depth_mask(0.7)
        mkdata.export_csi(dynamic_csi=False, pick_tx=0)
        #mkdata.lookup_image()
        mkdata.slice_by_label()

        #mkdata.playback_image()
        mkdata.save_dataset('../dataset/0509/make06', sub + '_chk', 'csi', 'img')

