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
        print(f'{self.name} converting IMG...', end='')
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
        self.center = np.zeros((len(self.img), 2), dtype=float)
        self.depth = np.zeros((len(self.img), 1))
        self.c_img = np.zeros((len(self.img), 1, 128, 128))

        tqdm.write(f"{self.name} calculating center coordinates and depth...")

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
                    self.center[i][0], self.center[i][1] = int(x + w / 2), int(y + h / 2)
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

    def convert_bbx_ctr(self, w_scale=226, h_scale=128):
        if self.bbx is not None:
            print(f'{self.name} converting bbx...', end='')
            self.bbx[..., 0] /= float(w_scale)
            self.bbx[..., 2] /= float(w_scale)
            self.bbx[..., 1] /= float(h_scale)
            self.bbx[..., 3] /= float(h_scale)
        if self.center is not None:
            print(f'{self.name} converting center...', end='')
            self.center[..., 0] /= float(w_scale)
            self.center[..., 1] /= float(w_scale)
        print('Done')

    def convert_depth(self, threshold=3000):
        print(f'{self.name} converting depth...', end='')
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

        ImageLoader.__init__(self, name, *args, **kwargs)
        CSILoader.__init__(self, name, *args, **kwargs)
        LabelParser.__init__(self, name, *args, **kwargs)

        self.pick_data: dict = {}
        self.de_pick_data: dict = {}
        self.pick = None
        self.de_pick = None
        self.pick_log = []

    @staticmethod
    def reshape_csi(csi, filter=True, pd=True):
        """
        Reshape CSI into channel * sub * packet.\n
        :return: reshaped (+ filtered) CSI
        """
        # packet * sub * rx
        if filter:
            csi_real = signal.savgol_filter(np.real(csi), 21, 3, axis=0)  # denoise for real part
            csi_imag = signal.savgol_filter(np.imag(csi), 21, 3, axis=0)  # denoise for imag part
        else:
            csi_real = np.real(csi)
            csi_imag = np.imag(csi)

        # packet * sub * (rx * 2)
        csi = np.concatenate((csi_real, csi_imag), axis=-1)

        # (rx * 2) * sub * packet
        csi = csi.transpose(2, 1, 0)

        if pd:
            win_len = 10
            # aoa vector = 1, tof vector = 29,  total len = 30
            # match CSI length by padding
            csi_ = csi_real + 1.j * csi_imag
            length, sub, rx = csi_.shape
            aoatof = np.zeros((30, length))
            csi_ = np.pad(csi_, ((4, 5), (0, 0), (0, 0)), 'edge')

            for i in range(length):
                u1, s1, v1 = np.linalg.svd(csi_[i:i + 10].transpose(2, 1, 0).
                                           reshape(-1, rx, sub * win_len), full_matrices=False)
                aoa = np.angle(u1[:, 0, 0].conj() * u1[:, 1, 0]).squeeze()
                u2, s2, v2 = np.linalg.svd(csi_[i:i + 10].transpose(1, 2, 0).
                                           reshape(-1, sub, rx * win_len), full_matrices=False)
                tof = np.angle(u2[:, :-1, 0].conj() * u2[:, 1:, 0]).squeeze()
                aoatof[0, i] = aoa
                aoatof[1:, i] = tof

            aoatof[0, :] = np.unwrap(aoatof[0, :])
            aoatof[1:, :] = np.unwrap(aoatof[1:, :], axis=-1)
            return csi, aoatof
        else:
            return csi

    def pick_samples(self, alignment=None):
        """
        Pick samples according to labels.\n
        :return: id of picked samples
        """
        print(f"{self.name} picking samples...\n")
        if alignment is not None:
            self.alignment = alignment

        for gr in tqdm(self.labels.keys()):
            for seg in tqdm(self.labels[gr].keys(), leave=False):
                start_img_id, end_img_id = np.searchsorted(self.camera_time,
                                                           [self.labels[gr][seg]['start'],
                                                            self.labels[gr][seg]['end']])
                img_id = np.arange(start_img_id, end_img_id, dtype=int)

                start_csi_id, end_csi_id = np.searchsorted(self.csi.timestamps,
                                                           [self.labels[gr][seg]['start'],
                                                            self.labels[gr][seg]['end']])
                # Indices of raw IMG-CSI pair

                csi_id = np.searchsorted(self.csi.timestamps, self.camera_time[img_id])

                # Remove out-of-segment CSI
                for i, (img_id_, csi_id_) in enumerate(zip(img_id, csi_id)):
                    if self.alignment == 'head':
                        sample_csi_id = np.arange(csi_id_, csi_id_ + self.csi_shape[1], dtype=int)
                    elif self.alignment == 'tail':
                        sample_csi_id = np.arange(csi_id_ - self.csi_shape[1], csi_id_, dtype=int)
                    if sample_csi_id[0] < start_csi_id or sample_csi_id[-1] > end_csi_id:
                        self.labels[gr][seg][i]: dict = {}
                    else:

                        # Sample-level ids
                        # img: 1 index
                        # CSI: list of indices
                        self.labels[gr][seg][i] = {'img': img_id_,
                                                   'csi': sample_csi_id}

                print(f" Group {gr} Segment {seg}: IMG = {start_img_id} ~ {end_img_id}, "
                      f"CSI = {start_csi_id} ~ {end_csi_id}.")

        print(' Done picking')

    def divide_train_test(self, pick=None):
        """
        Divide train and test samples at segment level, avoiding leakage.\n
        :param pick: specified segment
        :return: train_pick and test_pick
        """
        print(f'{self.name} dividing train and test...', end='')
        picks = []
        de_picks = []

        for gr in self.labels.keys():
            # Select 2 segments
            if not pick:
                pick1 = np.random.choice(list(self.labels[gr].keys()), 1, replace=False).astype(int)
                pick2 = [list(self.labels[gr].keys())[-1]] if pick1 == 0 else pick1 - 1
            else:
                pick1, pick2 = pick
            self.pick_log.append(f"group {gr}, segment {pick1} + {pick2}")
            for seg in self.labels[gr].keys():
                for sample, sampledata in self.labels[gr][seg].items():
                    if not isinstance(sample, str) and sampledata:
                        if seg in pick1 + pick2:
                            picks.append({'group': gr,
                                          'segment': seg,
                                          'sample': sample})
                        else:
                            de_picks.append({'group': gr,
                                             'segment': seg,
                                             'sample': sample})

        self.pick = picks
        self.de_pick = de_picks
        print('Done')
        return pick1, pick2

    def export_data(self, filter=True, pd=True):
        """
         Find csi, image and timestamps according to label.\n
        :param filter: whether apply savgol filter on each CSI sample
        :return: Sorted dataset
        """
        modalities = {'rimg': (1, 128, 226),
                      'csi': (6, 30, self.csi_shape[1]),
                      'time': (1, 1),
                      'cimg': (1, 128, 128),
                      'center': (1, 2),
                      'depth': (1, 1),
                      'pd': (1, 30, self.csi_shape[1])
                      }

        for mod, shape in modalities.items():
            self.pick_data[mod] = np.zeros((len(self.pick), *shape))
            self.de_pick_data[mod] = np.zeros((len(self.de_pick), *shape))

        tqdm.write(f'{self.name} exporting train data...')
        for i in tqdm(range(len(self.pick))):
            gr, seg, sample = self.pick[i].values()
            self.pick_data['rimg'][i] = self.img[self.labels[gr][seg][sample]['img']]
            self.pick_data['cimg'][i] = self.c_img[self.labels[gr][seg][sample]['img']]
            self.pick_data['center'][i] = self.center[self.labels[gr][seg][sample]['img']]
            self.pick_data['depth'][i] = self.depth[self.labels[gr][seg][sample]['img']]
            self.pick_data['time'][i] = self.camera_time[self.labels[gr][seg][sample]['img']]

            (self.pick_data['csi'][i],
             self.pick_data['pd'][i]) = self.reshape_csi(self.csi.csi[self.labels[gr][seg][sample]['csi'], ..., 0],
                                                         filter=filter, pd=pd)
        for mod, value in self.pick_data.items():
            tqdm.write(f" Exproted picked data: {mod} of {value.shape} as {value.dtype}")

        tqdm.write(f'{self.name} exporting test data...')
        for i in tqdm(range(len(self.de_pick))):
            gr, seg, sample = self.de_pick[i].values()
            self.de_pick_data['rimg'][i] = self.img[self.labels[gr][seg][sample]['img']]
            self.de_pick_data['cimg'][i] = self.c_img[self.labels[gr][seg][sample]['img']]
            self.de_pick_data['center'][i] = self.center[self.labels[gr][seg][sample]['img']]
            self.de_pick_data['depth'][i] = self.depth[self.labels[gr][seg][sample]['img']]
            self.de_pick_data['time'][i] = self.camera_time[self.labels[gr][seg][sample]['img']]

            (self.de_pick_data['csi'][i],
             self.de_pick_data['pd'][i]) = self.reshape_csi(self.csi.csi[self.labels[gr][seg][sample]['csi'], ..., 0],
                                                            filter=filter, pd=pd)
        for mod, value in self.de_pick_data.items():
            tqdm.write(f" Exproted de_picked data: {mod} of {value.shape} as {value.dtype}")

    def save_data(self):
        print(f"{self.name} saving...", end='')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for modality in self.pick_data.keys():
            np.save(os.path.join(self.save_path, f"{self.name}_csilen{self.csi_shape[1]}_{modality}_pick.npy"),
                    self.pick_data[modality])
            np.save(os.path.join(self.save_path, f"{self.name}_csilen{self.csi_shape[1]}_{modality}_depick.npy"),
                    self.de_pick_data[modality])
        np.save(os.path.join(self.save_path, f"{self.name}_csilen{self.csi_shape[1]}_label_pick.npy"),
                self.pick)
        np.save(os.path.join(self.save_path, f"{self.name}_csilen{self.csi_shape[1]}_label_depick.npy"),
                self.de_pick)

        with open(f"{self.save_path}{self.name}_csilen{self.csi_shape[1]}_log.txt", 'w') as logfile:
            logfile.write(f"{self.name}\n"
                          f"Raw paths:\n"
                          f"IMG: {self.img_path}\n"
                          f"CSI: {self.csi_path}\n"
                          f"TIME: {self.camera_time_path}\n"
                          f"LABEL: {self.label_path}\n"
                          f"Picked segments:\n"
                          f"{self.pick_log}\n")

        logfile.close()
        print("Done")


class DatasetMaker:
    version = 'V04'

    def __init__(self, subs=None, jupyter=True, mode='normal', dataset_name=None,
                 csi_shape=(2, 100, 30, 3),
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
        self.csi_shape = csi_shape

        self.picks = {sub: None for sub in self.subs}
        self.many_data = []
        self.few_data = []
        self.many_data_final: dict = {}
        self.few_data_final: dict = {}

        self.many_length = 0
        self.few_length = 0

        self.modalities = ('rimg', 'csi', 'time', 'cimg', 'center', 'depth', 'pd')

    def make_data(self):
        for sub in self.subs:
            mkdata = MyDataMaker(name=sub, csi_configs=self.configs, img_shape=(226, 128), csi_shape=self.csi_shape,
                                 img_path=f"{self.img_path}{sub}_img.npy",
                                 camera_time_path=f"{self.camera_time_path}{sub}_camtime.npy",
                                 csi_path=f"{self.csi_path}{sub}-csio.npy",
                                 label_path=f"{self.label_path}{sub}_labels.csv",
                                 save_path=f"{self.save_path}{self.dataset_name}/")
            mkdata.jupyter = self.jupyter
            mkdata.csi.extract_dynamic(mode='overall-divide', ref='tx', ref_antenna=1)
            mkdata.csi.extract_dynamic(mode='highpass')
            mkdata.convert_img()
            mkdata.crop()
            mkdata.convert_bbx_ctr()
            mkdata.convert_depth()
            mkdata.pick_samples(alignment='tail')
            self.picks[sub] = mkdata.divide_train_test(pick=self.picks[sub])
            mkdata.export_data(filter=True)

            self.many_data.append(mkdata.de_pick_data)
            self.few_data.append(mkdata.pick_data)

            mkdata.save_data()

    def regroup_data(self):
        print("Regrouping...")
        for modality in self.modalities:
            self.many_data_final[modality] = np.concatenate([data[modality] for data in self.many_data])
            self.few_data_final[modality] = np.concatenate([data[modality] for data in self.few_data])
            print(f"{modality} total many length = {len(self.many_data_final[modality])}, "
                  f"total few length = {len(self.few_data_final[modality])}")
            self.many_length = len(self.many_data_final[modality])
            self.few_length = len(self.few_data_final[modality])

    def save_data(self):
        tqdm.write("Saving...")
        save_path = f"../dataset/0509/{self.dataset_name}_split/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for mod in self.modalities:
            print(f" Saving {mod}...")
            np.save(f"{save_path}{mod}_many.npy", self.many_data_final[mod])
            np.save(f"{save_path}{mod}_few.npy", self.few_data_final[mod])

        print("All saved!")

    def divide_valid(self):
        """
        Divide validation set from training set by ratio of 0.2.\n
        :return: Finalized train, valid, test sets
        """
        print("Dividing valid...", end='')
        save_path = f"../dataset/0509/{self.dataset_name}_split/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(f"{save_path}{self.dataset_name}_log.txt", 'w') as logfile:
            logfile.write(f"{self.dataset_name}\n"
                          f"Raw paths:\n"
                          f"IMG: {self.img_path}xx_img.npy\n"
                          f"CSI: {self.csi_path}xx-csio.npy\n"
                          f"TIME: {self.camera_time_path}xx_camtime.npy\n"
                          f"LABEL: {self.label_path}xx_labels.csv\n"
                          f"Sources:\n"
                          f"{self.subs}\n"
                          f"Number of samples:\n"
                          f"Many = {self.many_length}, Few = {self.few_length}\n"
                          f"Train-Valid split:\n")

            for mode, tv_length, test_length in (('many', self.many_length, self.few_length),
                                                 ('few', self.few_length, self.many_length)):
                tv_ind = np.arange(tv_length).astype(int)
                valid_size = int(tv_length * 0.2)
                valid_ind = np.random.choice(tv_ind, valid_size, replace=False)
                valid_mask = np.ones(tv_length, np.bool)
                valid_mask[valid_ind] = 0
                train_ind = tv_ind[valid_mask]

                print(f"Divided train length = {tv_length - valid_size}, "
                      f"valid length = {valid_size}, test length = {test_length}")

                logfile.write(f"{mode}: train = {tv_length - valid_size}, valid = {valid_size}\n")

                np.save(f"{save_path}ind_{mode}_train.npy", train_ind)
                np.save(f"{save_path}ind_{mode}_valid.npy", valid_ind)

        print('Done')