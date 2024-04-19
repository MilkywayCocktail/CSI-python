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
from functools import wraps


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

def wrapper(func):
    @wraps(func)
    def inner(*args, **kwargs):
        print(f"Starting {func.__name__}...")
        ret = func(*args, **kwargs)
        print('Done')
        return ret

    return inner


class LabelParser:
    def __init__(self, name, label_path, *args, **kwargs):
        self.name = name
        self.label_path: str = label_path
        self.labels: dict = {}
        if self.label_path:
            self.parse()
        self.headers = None

    def parse(self):
        """
        Label Structure:\n
        |-group (0, 1, 2, ...)\n
            |-segment (0, 1, 2, ...)\n
                |-start\n
                |-end\n
                |-other attributes\n
        :return: Parsed label
        """
        print(f'{self.name} loading label...', end='')
        label: dict = {}
        with open(self.label_path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # ---Headers of label---
                    # group, segment, start, end, x0, y0, x1, y1
                    key_list = line.strip().split(',')
                    self.headers = key_list
                    for key in key_list:
                        label[key] = []
                else:
                    sample: dict = {}
                    # ---Values of label---
                    line_list = line.split(',')
                    for header, value in zip(self.headers, line_list):
                        sample[header] = eval(value)

                    sample['start'] *= 1.e-3
                    sample['end'] *= 1.e-3

                    if sample['group'] not in self.labels.keys():
                        self.labels[sample['group']]: dict = {}
                    if sample['segment'] not in self.labels[sample['group']].keys():
                        self.labels[sample['group']][sample['segment']]: dict = {}

                    self.labels[sample['group']][sample['segment']] = {key: value for key, value in sample.items()}

        print('Done')

    def save_labels(self):
        print(f"{self.name} saving labels...", end='')
        np.save(f"{os.path.split(self.label_path)[0]}/{os.path.split(self.label_path)[1]}.npy", self.labels)
        print("Done")


class ImageLoader:
    # Camera time should have been calibrated

    def __init__(self, name, img_path, camera_time_path, img_shape, *args, **kwargs):
        self.name = name
        self.img_path = img_path
        self.camera_time_path = camera_time_path
        self.img, self.camera_time = self.load_img()
        self.img_shape = img_shape
        self.bbx = None
        self.center = None
        self.depth = None
        self.c_img = None

    def load_img(self):
        print(f'{self.name} loading IMG...')
        img = np.load(self.img_path)
        camera_time = np.load(self.camera_time_path)
        print(f" Loaded images of {img.shape} as {img.dtype}")
        return img, camera_time

    def depth_mask(self, threshold=0.5):
        tqdm.write(f"{self.name} masking...")
        median = np.median(np.squeeze(self.img), axis=0)
        threshold = median * threshold
        plt.imshow(threshold / np.max(threshold))
        plt.title("Threshold map")
        plt.show()
        for i in tqdm(range(len(self.img))):
            mask = np.squeeze(self.img[i]) < threshold
            self.img[i] *= mask

    def convert_img(self, threshold=3000):
        print(f'Converting IMG...', end='')
        self.img[self.img > threshold] = threshold
        self.img /= float(threshold)
        print('Done')

    def crop(self, min_area=0, show=False):
        """
        Calculate cropped images, bounding boxes, center coordinates, depth.\n
        :param min_area: 0
        :param show: whether show images with bounding boxes
        :return: bbx, center, depth, c_img
        """
        self.bbx = np.zeros((len(self.img), 4))
        self.center = np.zeros(len(self.img))
        self.depth = np.zeros((len(self.img), 2))
        self.c_img = np.zeros((len(self.img), 1, 128, 128))

        tqdm.write("Calculating center coordinates and depth...")

        for i in tqdm(range(len(self.img))):
            img = np.squeeze(self.img[i]).astype('float32')
            (T, timg) = cv2.threshold((img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                contour = max(contours, key=lambda x: cv2.contourArea(x))
                area = cv2.contourArea(contour)

                if area < min_area:
                    # print(area)
                    pass

                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    patch = img[y:y + h, x:x + w]
                    non_zero = (patch != 0)
                    average_depth = patch.sum() / non_zero.sum()
                    self.bbx[i] = x, y, w, h
                    self.center[i] = int(x + w / 2), int(y + h / 2)
                    self.depth[i] = average_depth

                    subject = np.squeeze(self.img[i])[y:y + h, x:x + w]
                    image = np.zeros((128, 128))
                    image[int(64 - h / 2):int(64 + h / 2), int(64 - w / 2):int(64 + w / 2)] = subject
                    self.c_img[i, 0] = image

                    if show:
                        img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                            (x, y),
                                            (x + w, y + h),
                                            (0, 255, 0), 1)
                        cv2.namedWindow('Raw Image', cv2.WINDOW_AUTOSIZE)
                        cv2.imshow('Raw Image', img)
                        key = cv2.waitKey(33) & 0xFF
                        if key == ord('q'):
                            break

    def convert_bbx(self, w_scale=226, h_scale=128):
        print('Converting bbx...', end='')

        self.bbx[:, 0, 0] /= float(w_scale)
        self.bbx[:, 0, 2] /= float(w_scale)
        self.bbx[:, 0, 1] /= float(h_scale)
        self.bbx[:, 0, 3] /= float(h_scale)

        print('Done')

    def convert_depth(self, threshold=3000):
        print('Converting depth...', end='')
        self.depth[self.depth > threshold] = threshold
        self.depth /= float(threshold)
        print('Done')


class CSILoader:
    # CSI time should have been calibrated
    # CSI timestamps inside csi

    def __init__(self, name, csi_path, csi_configs: pycsi.MyConfigs, *args, **kwargs):
        self.name = name
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
    def __init__(self,
                 name=None,
                 csi_shape: tuple = (2, 100, 30, 3),
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
        :param alignment: how to align image and CSI
        :param jupyter_mode: whether to run on jupyter notebook
        :param save_path: save path
        """

        assert alignment in {'head', 'tail'}
        self.name = name
        self.save_path = save_path
        self.csi_shape = csi_shape
        self.alignment = alignment
        self.jupyter_mode = jupyter_mode

        ImageLoader.__init__(self, *args, **kwargs)
        CSILoader.__init__(self, *args, **kwargs)
        LabelParser.__init__(self, *args, **kwargs)

        self.train_data: dict = {}
        self.test_data: dict = {}
        self.train_pick = None
        self.test_pick = None

    @staticmethod
    def reshape_csi(csi, filter=True, pd=True):
        """
        Reshape CSI into channel * sub * packet.\n
        :return: reshaped (+ filtered) CSI
        """
        # packet * sub * rx * 1
        if filter:
            csi_real = signal.savgol_filter(np.real(csi), 21, 3, axis=0)  # denoise for real part
            csi_imag = signal.savgol_filter(np.imag(csi), 21, 3, axis=0)  # denoise for imag part
        else:
            csi_real = np.real(csi)
            csi_imag = np.imag(csi)

        # packet * sub * rx * 2
        csi = np.concatenate((csi_real, csi_imag), axis=-1)

        # (2 * rx) * sub * packet
        length, sub, rx, ch = csi.shape
        csi = csi.reshape((length, sub, rx * ch)).transpose((2, 1, 0))

        if pd:
            # aoa vector = 3, tof vector = 30,  total len = 33
            # match phasediff shape with CSI
            csi_ = csi_real + 1.j * csi_imag
            length, sub, rx, tx = csi_.shape

            aoatof = np.zeros((length, 33))
            for i, packet in enumerate(csi_):
                u1, s1, v1 = np.linalg.svd(packet.transpose(3, 2, 1, 0).
                                           reshape(-1, rx, sub * length), full_matrices=False)
                aoa = np.angle(u1[:, 0, 0].conj() * u1[:, 1, 0]).squeeze()
                u2, s2, v2 = np.linalg.svd(packet.transpose(3, 1, 2, 0).
                                           reshape(-1, sub, rx * length), full_matrices=False)
                tof = np.angle(u2[:, :-1, 0].conj() * u2[:, 1:, 0]).squeeze()
                aoatof[i] = np.concatenate((aoa, tof), axis=None)
            return csi, aoatof
        else:
            return csi

    def pick_samples(self, alignment=None):
        """
        Pick samples according to labels.\n
        :return: id of picked samples
        """
        print(f"Picking samples...\n")
        if alignment is not None:
            self.alignment = alignment

        for gr in tqdm(self.labels.keys()):
            for seg in tqdm(self.labels[gr].keys(), leave=False):
                start_img_id, end_img_id = np.searchsorted(self.camera_time[:, 0, 0],
                                                           [self.labels[gr][seg]['start'],
                                                            self.labels[gr][seg]['end']])
                start_csi_id, end_csi_id = np.searchsorted(self.csi.timestamps,
                                                           [self.labels[gr][seg]['start'],
                                                            self.labels[gr][seg]['end']])
                # Indices of raw IMG-CSI pair
                img_id = np.arange(start_img_id, end_img_id, dtype=int)
                csi_id = np.arange(start_csi_id, end_csi_id, dtype=int)

                # Remove out-of-segment CSI
                for i in tqdm(range(len(img_id)), leave=False):
                    if self.alignment == 'head':
                        sample_csi_id = np.arange(csi_id[i], csi_id[i] + self.csi_shape[1], dtype=int)
                    elif self.alignment == 'tail':
                        sample_csi_id = np.arange(csi_id[i] - self.csi_shape[1], csi_id[i], dtype=int)

                    if sample_csi_id[0] < start_csi_id or sample_csi_id[-1] > end_csi_id:
                        self.labels[gr][seg][i]: dict = {}
                    else:
                        # Sample-level ids
                        # img: 1 index
                        # CSI: list of indices
                        self.labels[gr][seg][i] = {'img': img_id[i],
                                                   'csi': sample_csi_id}
                # Segment-level ids
                self.labels[gr][seg]['img_id'] = img_id
                self.labels[gr][seg]['csi_id'] = csi_id

                print(f"\r Group {gr} Segment {seg}: IMG = {start_img_id} ~ {end_img_id}, "
                      f"CSI = {start_csi_id} ~ {end_csi_id}."
                      f" In segment: IMG = {img_id[0]} ~ {img_id[-1]},"
                      f"CSI = {csi_id[0]} ~ {csi_id[-1]}", end='')

        print('\nDone')

    def divide_train_test(self, mode='normal'):
        """
        Divide train and test samples at segment level, avoiding leakage.\n
        :param mode: 'normal' or 'few'
        :return: train_pick and test_pick
        """
        print('Dividing train and test...', end='')
        picks = []
        de_picks = []

        for gr in self.labels.keys():
            # Select 2 segments
            pick1 = np.random.choice(self.labels[gr].keys(), 1, replace=False).astype(int)
            pick2 = self.labels[gr].keys()[-1] if pick1 == 0 else pick1 - 1

            for seg in self.labels[gr].keys():
                for sample, sampledata in self.labels[gr][seg].items():
                    if isinstance(sample, int) and sampledata:
                        if seg in (pick1, pick2):
                            picks.append({'group': gr,
                                          'segment': seg,
                                          'sample': sample})
                        else:
                            de_picks.append({'group': gr,
                                             'segment': seg,
                                             'sample': sample})

        # Many train, few test
        if mode == 'normal':
            self.test_pick = picks
            self.train_pick = de_picks

        # Few train, many test
        elif mode == 'few':
            self.train_pick = picks
            self.test_pick = de_picks
        print('Done')

    def export_data(self, filter=True, pd=True):
        """
         Find csi, image and timestamps according to label.\n
        :param filter: whether apply savgol filter on each CSI sample
        :return: Sorted dataset
        """
        modalities = {'rimg': (len(self.train_pick), 1, self.img_shape[1], self.img_shape[0]),
                      'csi': (len(self.train_pick), *self.csi_shape),
                      'time': (len(self.train_pick), 1, 1),
                      'cimg': (len(self.train_pick), 1, 128, 128),
                      'center': (len(self.train_pick), 1, 2),
                      'depth': (len(self.train_pick), 1, 1),
                      'pd': (len(self.train_pick), 100, 33)
                      }

        for mod, shape in modalities.items():
            self.train_data[mod] = np.zeros(shape)
            self.test_data[mod] = np.zeros(shape)

        tqdm.write('Exporting train data...')
        for i in tqdm(range(len(self.train_pick))):
            gr, seg, sample = self.train_pick[i].items()
            self.train_data['rimg'][i] = self.img[self.labels[gr][seg][sample]['img']]
            self.train_data['cimg'][i] = self.c_img[self.labels[gr][seg][sample]['img']]
            self.train_data['center'][i] = self.center[self.labels[gr][seg][sample]['img']]
            self.train_data['depth'][i] = self.depth[self.labels[gr][seg][sample]['img']]
            self.train_data['time'][i] = self.camera_time[self.labels[gr][seg][sample]['img']]

            (self.train_data['csi'][i],
             self.train_data['pd'][i]) = self.reshape_csi(self.csi.csi[self.labels[gr][seg][sample]['csi']],
                                                          filter=filter, pd=pd)
        for mod, value in self.train_data.items():
            tqdm.write(f"Exproted train: {mod} of {value.shape} as {value.dtype}")

        tqdm.write('Exporting test data...')
        for i in tqdm(range(len(self.test_pick))):
            gr, seg, sample = self.test_pick[i].items()
            self.test_data['rimg'][i] = self.img[self.labels[gr][seg][sample]['img']]
            self.test_data['cimg'][i] = self.c_img[self.labels[gr][seg][sample]['img']]
            self.test_data['center'][i] = self.center[self.labels[gr][seg][sample]['img']]
            self.test_data['depth'][i] = self.depth[self.labels[gr][seg][sample]['img']]
            self.test_data['time'][i] = self.camera_time[self.labels[gr][seg][sample]['img']]

            (self.test_data['csi'][i],
             self.test_data['pd'][i]) = self.reshape_csi(self.csi.csi[self.labels[gr][seg][sample]['csi']],
                                                         filter=filter, pd=pd)
        for mod, value in self.test_data.items():
            tqdm.write(f"Exproted test: {mod} of {value.shape} as {value.dtype}")

    def save_data(self):
        print("Saving...", end='')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for modality in self.train_data.keys():
            np.save(os.path.join(self.save_path, f"{self.name}_csilen{self.csi_shape[1]}_{modality}_train.npy"),
                    self.train_data)
            np.save(os.path.join(self.save_path, f"{self.name}_csilen{self.csi_shape[1]}_{modality}_test.npy"),
                    self.test_data)
        np.save(os.path.join(self.save_path, f"{self.name}_csilen{self.csi_shape[1]}_label_train.npy"),
                self.train_pick)
        np.save(os.path.join(self.save_path, f"{self.name}_csilen{self.csi_shape[1]}_label_test.npy"),
                self.test_pick)
        print("Done")


class DatasetMaker:
    version = 'V04'

    def __init__(self, subs=None, jupyter=True, mode='normal', dataset_name=None,
                 img_path='../sense/0509/raw226_128/',
                 camera_time_path='../sense/0509/raw226_128/',
                 csi_path='../npsave/0509/0509A',
                 label_path='../sense/0509/',
                 save_path='../dataset/0509/'
                 ):
        if subs is None:
            subs = ['01', '02', '03', '04']

        self.subs = subs
        configs = pycsi.MyConfigs()
        configs.tx_rate = 0x1c113
        configs.ntx = 3

        self.img_path = img_path
        self.camera_time_path = camera_time_path
        self.csi_path = csi_path
        self.label_path = label_path
        self.save_path = save_path

        self.configs = configs
        self.jupyter = jupyter
        self.mode = mode
        self.dataset_name = dataset_name
        self.train_data = []
        self.test_data = []
        self.train_data_final: dict = {}
        self.valid_data_final: dict = {}
        self.test_data_final: dict = {}

        self.train_length = 0
        self.valid_length = 0
        self.test_length = 0

        self.valid_mask = None
        self.valid_ind = None
        self.tv_ind = None

        self.modalities = ('rimg', 'csi', 'time', 'cimg', 'center', 'depth', 'pd')

    def make_data(self):
        for sub in self.subs:
            mkdata = MyDataMaker(csi_configs=self.configs, img_shape=(226, 128), csi_shape=(2, 900, 30, 3),
                                 img_path=f"{self.img_path}{sub}_img.npy",
                                 camera_time_path=f"{self.camera_time_path}{sub}_camtime.npy",
                                 csi_path=f"{self.csi_path}{sub}-csio.npy",
                                 label_path=f"{self.label_path}{sub}_labels.csv",
                                 save_path=f"{self.save_path}{self.dataset_name}_{self.mode}/")
            mkdata.jupyter = self.jupyter
            mkdata.csi.extract_dynamic(mode='overall-divide', ref='tx', ref_antenna=1)
            mkdata.csi.extract_dynamic(mode='highpass')
            mkdata.convert_img()
            mkdata.crop()
            mkdata.convert_bbx()
            mkdata.convert_depth()
            mkdata.pick_samples(alignment='tail')
            mkdata.divide_train_test(mode=self.mode)
            mkdata.export_data(filter=True)
            self.train_data.append(mkdata.train_data)
            self.test_data.append(mkdata.test_data)
            mkdata.save_data()

    def regroup_data(self):
        print("Regrouping...")
        for modality in self.modalities:
            self.train_data_final[modality] = np.concatenate([train[modality] for train in self.train_data])
            self.test_data_final[modality] = np.concatenate([test[modality] for test in self.test_data])
            print(f"{modality} total train length = {len(self.train_data_final[modality])}, "
                  f"total test length = {len(self.test_data_final[modality])}")
            self.train_length = len(self.train_data_final[modality])
            self.test_length = len(self.test_data_final[modality])

    def divide_valid(self):
        """
        Divide validation set from training set by ratio of 0.2.\n
        :return: Finalized train, valid, test sets
        """
        print("Dividing valid...")
        self.tv_ind = np.arange(self.train_length).astype(int)
        valid_size = int(self.train_length * 0.2)
        self.valid_ind = np.random.choice(self.tv_ind, valid_size, replace=False)
        self.valid_mask = np.ones(self.train_length, np.bool)
        self.valid_mask[self.valid_ind] = 0

        print(f"Divided train length = {self.train_length - valid_size}, "
              f"valid length = {valid_size}, test length = {self.test_length}")

    def save(self):
        tqdm.write("Saving...")
        save_path = f"../dataset/0509/{self.dataset_name}_{self.mode}_split/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for mod in self.modalities:
            print(f" Saving {mod}...")
            np.save(f"{save_path}{mod}_train.npy", self.train_data_final[mod][self.valid_mask])
            np.save(f"{save_path}{mod}_valid.npy", self.train_data_final[mod][self.valid_ind])
            np.save(f"{save_path}{mod}_test.npy", self.test_data_final[mod])
        np.save(f"{save_path}ind_train.npy", self.tv_ind[self.valid_mask])
        np.save(f"{save_path}ind_valid.npy", self.valid_ind)

        print("All saved!")
