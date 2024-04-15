import numpy as np
import cv2
from scipy import signal
import os
import pycsi
import pyrealsense2 as rs
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import display, clear_output

##############################################################################
# -------------------------------------------------------------------------- #
# ABOUT TIMESTAMPS
# There are 3 kinds of timestamps:
# CSI timestamp -- local timestamp -- camera timestamp
#
# local timestamp works as a standard
# CSI timestamp should be pre-calibrated against local timestamp
# camera timestamp is calibrated when exporting from bag
# -------------------------------------------------------------------------- #
##############################################################################


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


class BagLoader:
    def __init__(self, total_frames, bag_path, local_time_path, img_shape=(128, 128)):
        self.total_frames = total_frames
        self.bag_path = bag_path
        self.local_time_path = local_time_path
        self.img_shape = img_shape
        self.video_stream = self.__setup_video_stream__()
        self.local_time = self.load_local_timestamps()
        self.images = None
        self.camera_time = None
        self.caliberated = False
        self.camtime_delta = 0.

    def __setup_video_stream__(self):
        # timestamps is the local time; works as reference time
        print('Setting camera stream...', end='')
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.bag_path, False)
        config.enable_all_streams()
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)
        print('Done')

        return pipeline

    def load_local_timestamps(self):
        if self.local_time_path:
            local_tf = open(self.local_time_path, mode='r', encoding='utf-8')
            local_time = np.array(local_tf.readlines())
            for i, line in enumerate(local_time):
                local_time[i] = datetime.timestamp(datetime.strptime(line.strip(), "%Y-%m-%d %H:%M:%S.%f"))
            local_tf.close()
            return local_time.astype(np.float64)
        else:
            return None

    def __get_image__(self, mode):
        frame = self.video_stream.wait_for_frames()

        if mode == 'depth':
            depth_frame = frame.get_depth_frame()
            frame_timestamp = depth_frame.get_timestamp() / 1.e3
            if not depth_frame:
                eval('continue')
            depth_frame = my_filter(depth_frame)
            image = np.asanyarray(depth_frame.get_data())

        elif mode == 'color':
            color_frame = frame.get_color_frame()
            frame_timestamp = color_frame.get_timestamp() / 1.e3
            if not color_frame:
                eval('continue')
            image = np.asanyarray(color_frame.get_data())

        else:
            raise Exception('Please specify mode = \'depth\' or \'color\'.')

        return image, frame_timestamp

    def calibrate_camtime(self, camtime, localtime):
        """
        Calibrate camera timestamps against local timestamps.\n
        This is important for fetching CSI packets according to timestamps.\n
        :return: calibrated timestamps
        """
        print('Calibrating camera time against local time file...', end='')
        cvt = datetime.fromtimestamp

        assert len(camtime) == len(localtime)

        if not self.caliberated:
            temp_lag = np.zeros(len(camtime))
            for i in range(len(camtime)):
                temp_lag[i] = camtime[i] - localtime[i]

            camtime_delta = np.mean(temp_lag)

            for i in range(len(camtime)):
                camtime[i] -= camtime_delta

            self.caliberated = True
            self.camtime_delta = camtime_delta
            print('Done')
            print('lag={}'.format(camtime_delta))
            return camtime
        else:
            print("Already calibrated")

    def export_images(self, mode='depth'):
        tqdm.write('Starting exporting image...')

        self.images = np.zeros((self.total_frames, self.img_shape[1], self.img_shape[0]))
        self.camera_time = np.zeros(self.total_frames)

        try:
            self.__setup_video_stream__()
            for i in tqdm(range(self.total_frames)):
                image, self.camera_time[i, ...] = self.__get_image__(mode=mode)
                image = cv2.resize(image, self.img_shape, interpolation=cv2.INTER_AREA)
                self.images[i, ...] = image
        except RuntimeError:
            pass

        finally:
            self.video_stream.stop()
            if self.local_time is not None:
                self.camera_time = self.calibrate_camtime(self.camera_time, self.local_time)

    def depth_mask(self, threshold=0.5):
        tqdm.write("Masking...")
        median = np.median(np.squeeze(self.images), axis=0)
        threshold = median * threshold
        plt.imshow(threshold / np.max(threshold))
        plt.title("Threshold map")
        plt.show()
        for i in tqdm(range(len(self.images))):
            mask = np.squeeze(self.images[i]) < threshold
            self.images[i] *= mask
        tqdm.write("Done")

    def convert_img(self, threshold=3000):
        print('Converting IMG...', end='')
        self.images[self.images > threshold] = threshold
        self.images /= float(threshold)
        print('Done')

    def save_images(self, path=None):
        print("Saving...", end='')
        if path:
            if not os.path.exists(path):
                os.makedirs(path)
            _, filename = os.path.split(self.bag_path)
            file, ext = os.path.splitext(filename)
            np.save(f"{path}{file}_img.npy", self.images)
            np.save(f"{path}{file}_camtime.npy", self.camera_time)
        else:
            np.save(f"../{os.path.splitext(os.path.basename(self.bag_path))[0]}_raw.npy", self.images)
            np.save(f"../{os.path.splitext(os.path.basename(self.bag_path))[0]}_camtime.npy", self.camera_time)
        print("Done")


class LabelParser:
    def __init__(self, label_path, *args, **kwargs):
        self.label_path = label_path
        self.label = None
        self.segment = None
        if self.label_path:
            self.parse()

    def parse(self):
        """
        Label Structure:\n
        --segment (0, 1, 2, ...)\n
        --start (all start timestamps)\n
        --end (all end timestamps)\n
        --other keys\n
        :return: Parsed label
        """
        print('Loading label...', end='')
        label = {'segment': []}
        with open(self.label_path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # ---Headers of label---
                    key_list = line.strip().split(',')
                    for key in key_list:
                        label[key] = []
                else:
                    # ---Values of label---
                    line_list = line.split(',')
                    label['segment'].append(i - 1)
                    for ii, key in enumerate(key_list):
                        if key != 'segment':
                            label[key].append(eval(line_list[ii]))

        for key in label.keys():
            label[key] = np.array(label[key])

        label['start'] *= 1.e-3
        label['end'] *= 1.e-3
        label['indices'] = {}
        self.label = label
        print('Done')

    def add_condition(self):
        self.label['direction'] = []
        for i in range(len(self.label['start'])):
            if self.label['x0'][i] == self.label['x1'][i]:
                self.label['direction'].append('y')
            elif self.label['y0'][i] == self.label['y1'][i]:
                self.label['direction'].append('x')

    def save_raw_labels(self):
        print("Saving labels...", end='')
        np.save(f"../{os.path.splitext(os.path.basename(self.label_path))[0]}_labels.npy", self.label)
        print("Done")


class ImageLoader:
    # Camera time should have been calibrated

    def __init__(self, img_path, camera_time_path, img_shape, *args, **kwargs):
        self.img_path = img_path
        self.camera_time_path = camera_time_path
        self.img, self.camera_time = self.load_img()
        self.img_shape = img_shape

    def load_img(self):
        print('Loading IMG...')
        img = np.load(self.img_path)
        camera_time = np.load(self.camera_time_path)
        print(f" Loaded images of {img.shape} as {img.dtype}")
        return img, camera_time

    def depth_mask(self, threshold=0.5):
        tqdm.write("Masking...")
        median = np.median(np.squeeze(self.img), axis=0)
        threshold = median * threshold
        plt.imshow(threshold / np.max(threshold))
        plt.title("Threshold map")
        plt.show()
        for i in tqdm(range(len(self.img))):
            mask = np.squeeze(self.img[i]) < threshold
            self.img[i] *= mask
        tqdm.write("Done")

    def convert_img(self, threshold=3000):
        print('Converting IMG...', end='')
        self.img[self.img > threshold] = threshold
        self.img /= float(threshold)
        print('Done')


class CSILoader:
    # CSI time should have been calibrated
    # CSI timestamps inside csi

    def __init__(self, csi_path, csi_configs: pycsi.MyConfigs, *args, **kwargs):
        self.csi_path = csi_path
        self.csi_configs = csi_configs
        self.csi = self.load_csi()
        self.filtered_csi = None

    def load_csi(self):
        csi = pycsi.MyCsi(self.csi_configs, 'CSI', self.csi_path)
        csi.load_data(remove_sm=True)
        return csi

    @staticmethod
    def windowed_dynamic(in_csi):
        # in_csi: pkt * sub * rx
        # in_csi = np.squeeze(in_csi)
        phase_diff = in_csi * in_csi[..., 0][..., np.newaxis].conj().repeat(3, axis=2)
        static = np.mean(phase_diff, axis=0)
        dynamic = phase_diff - static
        return dynamic


class MyDataMaker(ImageLoader, CSILoader, LabelParser):
    def __init__(self, total_frames: int,
                 csi_shape: tuple = (2, 100, 30, 3),
                 assemble_number: int = 1,
                 alignment: str = 'tail',
                 jupyter_mode=False,
                 save_path: str = '../saved/',
                 *args, **kwargs
                 ):
        """
        Assemble of ImageLoader, CSILoader and LabelParser.\n
        :param total_frames: number of frames of images
        :param csi_configs: configs for loading CSI
        :param csi_shape: (channel, packet, subcarrier, rx)
        :param img_shape: (width, height)
        :param assemble_number: how many img-csi pairs in one sample
        :param alignment: how to align image and CSI
        :param jupyter_mode: whether to run on jupyter notebook
        :param save_path: save path
        """

        assert alignment in {'head', 'tail'}

        self.save_path = save_path
        self.frames = total_frames
        self.samples = total_frames // assemble_number
        self.csi_shape = csi_shape
        self.assemble_number = assemble_number
        self.alignment = alignment

        self.jupyter_mode = jupyter_mode

        ImageLoader.__init__(self, *args, **kwargs)
        CSILoader.__init__(self, *args, **kwargs)
        LabelParser.__init__(self, *args, **kwargs)
        self.result: dict = {}
        self.init_data()

    def init_data(self):
        csi = np.zeros((self.frames, *self.csi_shape))
        images = np.zeros((self.frames, 1, self.img_shape[1], self.img_shape[0]))
        timestamps = np.zeros((self.frames, 1, 1))
        indices = np.zeros((self.frames, 1, 1), dtype=int)
        self.result = {'csi': csi, 'img': images, 'time': timestamps, 'csi_ind': indices}

    def reshape_csi(self):
        """
        Reshape CSI into expected size. (Excluding assemble).\n
        :return: reshaped CSI
        """
        length, channel, *csi_shape = self.result['csi'].shape
        self.result['csi'] = (self.result['csi'].transpose(
            (0, 1, 4, 3, 2))).reshape(length, channel * 3, self.csi_shape[2], self.csi_shape[1])

    @staticmethod
    def filter_csi(csi):
        csi_real = signal.savgol_filter(np.real(csi), 21, 3, axis=-1)  # denoise for real part
        csi_imag = signal.savgol_filter(np.imag(csi), 21, 3, axis=-1)  # denoise for imag part
        return csi_real + 1.j * csi_imag

    def export_data(self, window_dynamic=False, pick_tx=0, filter=True, alignment=None):
        """
         Find csi packets according to timestamps of images.\n
        :param window_dynamic: whether to subtract static component
        :param pick_tx: select one of the tx
        :return: Preliminarily assorted CSI
        """
        tqdm.write('Starting exporting data...')
        if alignment is not None:
            self.alignment = alignment

        for i in tqdm(range(self.frames)):
            self.result['img'][i, ...] = self.img[i]
            self.result['time'][i, ...] = self.camera_time[i]

            csi_index = np.searchsorted(self.csi.timestamps, self.camera_time[i])
            self.result['csi_ind'][i, ...] = csi_index

            csi_sample = np.zeros((*self.csi_shape[1:], 1))  # (pkt, sub, rx, 1)

            if self.alignment == 'head':
                csi_sample = self.csi.csi[csi_index: csi_index + self.csi_shape[1], :, :, pick_tx]
            elif self.alignment == 'tail':
                csi_sample = self.csi.csi[csi_index - self.csi_shape[1]: csi_index, :, :, pick_tx]

            if window_dynamic:
                csi_sample = self.windowed_dynamic(csi_sample)

            if filter:
                csi_sample = self.filter_csi(csi_sample)

            # Store in two channels and reshape
            self.result['csi'][i, 0, ...] = np.real(csi_sample)
            self.result['csi'][i, 1, ...] = np.imag(csi_sample)

        self.reshape_csi()

    def deoverlap(self):
        """
        Remove overlapping CSI samples.\n
        :return: Deoverlapped CSI and all other metadata
        """
        print("Deopverlapping...", end='')
        de_flag = np.zeros(self.frames, dtype=bool)
        boundary = -1

        for i in range(len(self.result['time'])):
            if self.result['time'][i, 0, 0] > boundary:
                de_flag[i] = True
                boundary = self.result['time'][i, 0, 0] + self.csi_shape[1] * 1.e-3  # CSI sampling rate
            else:
                continue

        for modality, value in self.result.items():
            self.result[modality] = value[de_flag]
        print('Done')

    def slice_by_label(self):
        """
        Segmentation according to image annotation.\n
        :return: sliced metadata
        """
        print(f"Slicing {len(self.label['segment'])} segments...")
        self.segment = {seg: None for seg in range(len(self.label['segment']))}
        self.result['label'] = {}

        # Divide segments
        for seg in range(len(self.label['start'])):

            start_id, end_id = np.searchsorted(self.result['time'][:, 0, 0],
                                               [self.label['start'][seg],
                                                self.label['end'][seg]])
            print(f" Segment {seg} start={start_id}, end={end_id}")
            # Locate images in each segment
            self.segment[seg] = np.arange(start_id, end_id)
            # For saving labels
            self.result['label'][seg] = seg * np.ones((len(self.segment[seg]), 1, 1))

        for mod, modality in self.result.items():
            if mod != 'label':
                self.result[mod] = {seg: None for seg in self.segment.keys()}
                for seg, segment in self.segment.items():
                    self.result[mod][seg] = modality[segment]

        print('Done')

    def assemble(self):
        """
        Assemble every several samples into 1 sample.\n
        :return: Assembled metadata
        """
        assert self.result
        print("Aligning...", end='')
        for mod, modality in self.result.items():
            for seg in self.segment.keys():
                length, channel, *shape = modality[seg].shape
                if length > 0:
                    assemble_length = length // self.assemble_number
                    slice_length = assemble_length * self.assemble_number
                    self.result[mod][seg] = modality[seg][:slice_length].reshape(
                        assemble_length, self.assemble_number * channel, *shape)
        print("Done")

    def save_dataset(self, save_name=None, *args):
        print("Saving...", end='')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for modality in args:
            if modality in self.result.keys():
                # if modality in ('time', 'label'):
                #   np.save(os.path.join(self.paths['save'], f"{save_name}_{modality}.npy"),
                #   self.result[data][modality])
                # else:
                np.save(os.path.join(
                    self.save_path, f"{save_name}_asmb{self.assemble_number}_len{self.csi_shape[1]}_{modality}.npy"),
                    np.concatenate(list(self.result[modality].values())))
        print("Done")
