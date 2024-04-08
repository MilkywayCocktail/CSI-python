import numpy as np
import cv2
import os
import pycsi
import pyrealsense2 as rs
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import display, clear_output


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


class LabelParser:
    def __init__(self, label_path):
        self.__label_path = label_path
        self.label = None
        if self.__label_path:
            self.parse()

    def parse(self):
        print('Loading label...', end='')
        label = {'segment': []}
        with open(self.__label_path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # ---Keys of label---
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
        np.save(f"../{os.path.splitext(os.path.basename(self.__label_path))[0]}_labels.npy", self.label)
        print("Done")


class ImageLoader:
    def __init__(self, img_path, timestamp_path, img_size):
        self.__img_path = img_path
        self.__timestamppath = timestamp_path
        self.__img = np.load(self.__img_path, mmap_mode='r')
        self.__timestamp = np.load(timestamp_path)
        self.img_size = img_size

    def depth_mask(self, threshold=0.5):
        tqdm.write("Masking...")
        median = np.median(np.squeeze(self.__img), axis=0)
        threshold = median * threshold
        plt.imshow(threshold / np.max(threshold))
        plt.title("Threshold map")
        plt.show()
        for i in tqdm(range(len(self.__img))):
            mask = np.squeeze(self.__img[i]) < threshold
            self.__img[i] *= mask
        tqdm.write("Done")


class CSILoader:
    def __init__(self, csi_path, csi_configs):
        self.__csi_path = csi_path
        self.csi_configs = csi_configs
        self.csi = self.__load_csi__()

    def __load_csi__(self):
        print('Loading CSI...')
        csi = pycsi.MyCsi(self.csi_configs, 'CSI', self.__csi_path)
        csi.load_data(remove_sm=True)

        return csi  # Pre-calibrated versus local machine

    @staticmethod
    def windowed_dynamic(in_csi):
        # in_csi = np.squeeze(in_csi)
        phase_diff = in_csi * in_csi[..., 0][..., np.newaxis].conj().repeat(3, axis=2)
        static = np.mean(phase_diff, axis=0)
        dynamic = phase_diff - static
        return dynamic


class MyDataMaker(ImageLoader, CSILoader, LabelParser):
    def __init__(self, total_frames: int,
                 csi_configs: pycsi.MyConfigs,
                 img_size: tuple = (128, 128),
                 csi_length: int = 100,
                 assemble_number: int = 1,
                 alignment: str = 'head',
                 paths: dict = None,
                 jupyter_mode=False
                 ):
        if paths is None:
            paths = {'bag': None,
                     'localtime': None,
                     'csi': None,
                     'label': None,
                     'save': '/saved/'
                     }
        self.paths = paths
        self.frames = total_frames
        self.samples = total_frames // assemble_number
        self.csi_length = csi_length
        self.assemble_number = assemble_number
        self.alignment = alignment

        self.jupyter_mode = jupyter_mode

        ImageLoader.__init__(self, self.paths['img'], self.paths['localtime'], img_size)
        CSILoader.__init__(self, self.paths['csi'], csi_configs)
        LabelParser.__init__(self, self.paths['label'])

        self.result = {'vanilla': self.init_data(),
                       'annotated': {}}

    def init_data(self):
        # img_size = (width, height)

        csi = np.zeros((self.frames, 2, self.csi_length, 30, 3))
        images = np.zeros((self.frames, 1, self.img_size[1], self.img_size[0]))
        timestamps = np.zeros((self.frames, 1, 1))
        indices = np.zeros((self.frames, 1, 1), dtype=int)

        return {'csi': csi, 'img': images, 'time': timestamps, 'ind': indices}

    def reshape_csi(self):
        length, channel, *csi_shape = self.result['vanilla']['csi'].shape
        self.result['vanilla']['csi'] = (self.result['vanilla']['csi'].transpose(
            (0, 1, 4, 3, 2))).reshape(length, channel * 3, 30, 100)

    def export_csi(self, window_dynamic=False, pick_tx=0):
        """
        Finds csi packets according to the timestamps of images.\n
        Requires export_image.\n
        """
        tqdm.write('Starting exporting CSI...')
        boundary = -1

        for i in tqdm(range(self.frames)):
            csi_index = np.searchsorted(self.csi.timestamps, self.result['vanilla']['time'][i, 0, 0])
            self.result['vanilla']['ind'][i, ...] = csi_index

            if self.result['vanilla']['time'][i, 0, 0] > boundary:
                if self.alignment == 'head':
                    csi_sample = self.csi.csi[csi_index: csi_index + self.csi_length, :, :, pick_tx]
                elif self.alignment == 'tail':
                    csi_sample = self.csi.csi[csi_index - self.csi_length: csi_index, :, :, pick_tx]

                boundary = self.csi.timestamps[csi_index] + self.csi_length * 1.e-3

            if window_dynamic:
                csi_sample = self.windowed_dynamic(csi_sample)

            # Store in two channels and reshape
            self.result['vanilla']['csi'][i, 0, ...] = np.abs(csi_sample)
            self.result['vanilla']['csi'][i, 1, ...] = np.angle(csi_sample)

        self.reshape_csi()

    def deoverlap(self):
        print("Deopverlapping...", end='')
        de_flag = np.zeros(self.frames, dtype=bool)
        boundary = -1

        for i in range(len(self.result['vanilla']['time'])):
            if self.result['vanilla']['time'][i, 0, 0] > boundary:
                de_flag[i] = True
                boundary = self.result['vanilla']['time'][i, 0, 0] + self.csi_length * 1.e-3
            else:
                continue

        for modality in self.result['vanilla'].keys():
            self.result['vanilla'][modality] = self.result['vanilla'][modality][de_flag]
        print('Done')

    def slice_by_label(self):
        """
        Segmentation regarding annotation.\n
        :return: sliced results
        """
        print('Slicing...', end='')

        segment = {seg: None for seg in range(len(self.label['segment']))}
        self.result['annotated']['label'] = {}

        for seg in range(len(self.label['start'])):
            start_id, end_id = np.searchsorted(self.result['vanilla']['time'][:, 0, 0],
                                               [self.label['start'][seg] - self.camtime_delta,
                                                self.label['end'][seg] - self.camtime_delta])
            segment[seg] = np.arange(start_id, end_id)
            self.result['annotated']['label'][seg] = seg * np.ones((len(segment[seg]), 1, 1))

        self.label['segment'] = segment

        for modality in self.result['vanilla'].keys():
            self.result['annotated'][modality] = {seg: None for seg in segment.keys()}
            for seg in segment.keys():
                self.result['annotated'][modality][seg] = self.result['vanilla'][modality][segment[seg]]
        print('Done')

    def depth_mask(self, threshold=0.5):
        tqdm.write("Masking...")
        median = np.median(np.squeeze(self.result['vanilla']['img']), axis=0)
        threshold = median * threshold
        plt.imshow(threshold / np.max(threshold))
        plt.title("Threshold map")
        plt.show()
        for i in tqdm(range(len(self.result['vanilla']['img']))):
            mask = np.squeeze(self.result['vanilla']['img'][i]) < threshold
            self.result['vanilla']['img'][i] *= mask
        tqdm.write("Done")

    def assemble(self):
        print("Aligning...", end='')
        if self.result['annotated']:
            for modality in self.result['vanilla'].keys():
                for seg in self.label['segment'].keys():
                    length, channel, *shape = self.result['annotated'][modality][seg].shape
                    assemble_length = length // self.assemble_number
                    slice_length = assemble_length * self.assemble_number
                    self.result['annotated'][modality][seg] = self.result['annotated'][modality][seg][:slice_length].reshape(
                        assemble_length, self.assemble_number * channel, *shape)
        else:
            # Assemble vanilla data (usually not needed)
            pass
        print("Done")

    def save_dataset(self, save_name=None, data='annotated', *args):
        print("Saving...", end='')
        if not os.path.exists(self.paths['save']):
            os.makedirs(self.paths['save'])

        for modality in args:
            if modality in self.result[data].keys():
                # if modality in ('time', 'label'):
                #   np.save(os.path.join(self.paths['save'], f"{save_name}_{modality}.npy"),
                #   self.result[data][modality])
                # else:
                np.save(os.path.join(
                    self.paths['save'], f"{save_name}_asmb{self.assemble_number}_len{self.csi_length}_{modality}.npy"),
                    np.concatenate(list(self.result[data][modality].values())))
        print("Done")
