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
modalities = {'rimg': (1, 128, 226),
              'csi': (6, 30, 'csi_len'),
              'time': (1, 1),
              'cimg': (1, 128, 128),
              'center': (1, 2),
              'bbx': (1, 4),
              'depth': (1, 1),
              'pd': (1, 30, 'csi_len'),
              'ind': (1, 1)
              }

ver = 'V05'


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
                    # Space for matching sample indices
                    self.labels[sample['group']][sample['segment']]['samples']: dict = {}

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
        self.id = None

    def load_img(self):
        print(f'{self.name} loading IMG...')
        img = np.load(self.img_path)
        camera_time = np.load(self.camera_time_path)
        self.id = np.arange(len(img))
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
        self.bbx = np.zeros((len(self.img), 4), dtype=float)
        self.center = np.zeros((len(self.img), 2), dtype=float)
        self.depth = np.zeros((len(self.img), 1), dtype=float)
        self.c_img = np.zeros((len(self.img), 1, 128, 128), dtype=float)

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
            self.center[..., 1] /= float(h_scale)
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
    version = ver

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

        # data = group: xx, segment: yy
        # Save data by segment
        self.data: dict = {}

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
            return {'csi': csi, 'pd': aoatof}
        else:
            return {'csi': csi}

    @staticmethod
    def reshape_image(img,
                      min_area=0,
                      cvt_bbx=True, cvt_center=True,
                      w_scale=226, h_scale=128):
        r_img = np.squeeze(img).astype('float32')
        c_img = np.zeros((128, 128))
        average_depth = 0
        x, y, w, h = 0, 0, 0, 0
        (T, timg) = cv2.threshold((r_img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            area = cv2.contourArea(contour)

            if area < min_area:
                # print(area)
                pass
            else:
                x, y, w, h = cv2.boundingRect(contour)
                patch = r_img[y:y + h, x:x + w]
                non_zero = (patch != 0)
                average_depth = patch.sum() / non_zero.sum()
                subject = np.squeeze(r_img)[y:y + h, x:x + w]
                c_img[int(64 - h / 2):int(64 + h / 2), int(64 - w / 2):int(64 + w / 2)] = subject

                if cvt_bbx:
                    bbx = [x, y, x + w, y + h]
                    bbx[0] /= float(w_scale)
                    bbx[2] /= float(w_scale)
                    bbx[1] /= float(h_scale)
                    bbx[3] /= float(h_scale)
                else:
                    bbx = [x, y, w, h]

                center = [int(x + w / 2), int(y + h / 2)]
                if cvt_center:
                    center[0] /= float(w_scale)
                    center[1] /= float(h_scale)

        return {'rimg': r_img, 'cimg': c_img, 'depth': average_depth, 'bbx': bbx, 'center': center}

    def export_data(self, alignment=None, filter=True, pd=True):
        """
        Pick samples according to labels.\n
        :return: id of picked samples
        """
        modalities['csi'] = (6, 30, self.csi_shape[1])
        modalities['pd'] = (1, 30, self.csi_shape[1])

        print(f"{self.name} exporting data...\n")
        if alignment is not None:
            self.alignment = alignment

        self.data: dict = {}
        for gr in tqdm(self.labels.keys()):
            self.data[gr]: dict = {}
            for seg in tqdm(self.labels[gr].keys(), leave=False):
                self.data[gr][seg]: dict = {}
                self.labels[gr][seg]['samples']: dict = {}

                start_img_id, end_img_id = np.searchsorted(self.camera_time,
                                                           [self.labels[gr][seg]['start'],
                                                            self.labels[gr][seg]['end']])
                img_id = np.arange(start_img_id, end_img_id, dtype=int)

                start_csi_id, end_csi_id = np.searchsorted(self.csi.timestamps,
                                                           [self.labels[gr][seg]['start'],
                                                            self.labels[gr][seg]['end']])
                # Indices of single IMG-CSI pair
                csi_id = np.searchsorted(self.csi.timestamps, self.camera_time[img_id])

                # Remove out-of-segment CSI
                selected = np.ones_like(img_id, dtype=bool)
                for i, (img_id_, csi_id_) in enumerate(zip(img_id, csi_id)):
                    if self.alignment == 'head':
                        sample_csi_id = np.arange(csi_id_, csi_id_ + self.csi_shape[1], dtype=int)
                    elif self.alignment == 'tail':
                        sample_csi_id = np.arange(csi_id_ - self.csi_shape[1], csi_id_, dtype=int)
                    if sample_csi_id[0] < start_csi_id or sample_csi_id[-1] > end_csi_id:
                        selected[i] = False
                    else:
                        # Use img_ind as sample ind and key, csi_ind as value
                        # img: 1 index
                        # CSI: list of indices
                        self.labels[gr][seg]['samples'][img_id_] = sample_csi_id

                seglen = len(img_id[selected])
                if seglen > 0:
                    for mod, shape in modalities.items():
                        self.data[gr][seg][mod] = np.zeros((seglen, *shape))

                    for i, ind in enumerate(img_id[selected]):
                        img_ret = self.reshape_image(self.img[ind])
                        csi_ret = self.reshape_csi(self.csi.csi[self.labels[gr][seg]['samples'][ind], ..., 0], filter=filter, pd=pd)
                        for mod, value in (*img_ret.items(), *csi_ret.items()):
                            self.data[gr][seg][mod][i] = value
                        self.data[gr][seg]['ind'][i] = ind

                    print(f" Group {gr} Segment {seg}: IMG = {start_img_id} ~ {end_img_id}, "
                          f"CSI = {start_csi_id} ~ {end_csi_id}.")
                else:
                    print(f" Group {gr} Segment {seg}: No sample selected.")

        tqdm.write(' Done exporting')

    def save_data(self):
        tqdm.write(f"{self.name} saving...")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(f"{self.save_path}{self.name}_csilen{self.csi_shape[1]}_log.txt", 'w') as logfile:
            logfile.write(f"{self.name:0>2}\n"
                          f"Raw paths:\n"
                          f"IMG: {self.img_path}\n"
                          f"CSI: {self.csi_path}\n"
                          f"TIME: {self.camera_time_path}\n"
                          f"LABEL: {self.label_path}\n\n"
                          f"Indices:\n")

            for gr in tqdm(self.labels.keys()):
                for seg in tqdm(self.labels[gr].keys(), leave=False):
                    logfile.write(f"G{gr:0>2}_S{seg:0>2}: "
                                  f"{np.squeeze(self.data[gr][seg]['ind']).astype(str)}\n")
                    for mod in tqdm(modalities.keys(), leave=False):
                        np.save(os.path.join(self.save_path, f"T{self.name:0>2}_G{gr:0>2}_S{seg:0>2}_{mod}.npy"),
                                self.data[gr][seg][mod])

        tqdm.write(" All saved!")


class DatasetMaker:
    version = ver

    def __init__(self, subs=None, jupyter=True, dataset_name=None,
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
        self.csi_shapes = {key: (2, key, 30, 3) for key in (30, 100, 300, 900)}
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
        self.dataset_name = dataset_name
        self.csi_shape = csi_shape

        self.modalities = list(modalities.keys())

    def make_data(self):
        for sub in self.subs:
            mkdata = MyDataMaker(name=sub, csi_configs=self.configs, img_shape=(226, 128), csi_shape=(2, 30, 30, 3),
                                 img_path=f"{self.img_path}{sub}_img.npy",
                                 camera_time_path=f"{self.camera_time_path}{sub}_camtime.npy",
                                 csi_path=f"{self.csi_path}{sub}-csio.npy",
                                 label_path=f"{self.label_path}{sub}_labels.csv",
                                 save_path=f"{self.save_path}{self.dataset_name}/")
            mkdata.jupyter = self.jupyter
            mkdata.csi.extract_dynamic(mode='overall-divide', ref='tx', ref_antenna=1)
            mkdata.csi.extract_dynamic(mode='highpass')
            mkdata.convert_img()
                # mkdata.crop()
                # mkdata.convert_bbx_ctr()
                # mkdata.convert_depth()
            for csi_len, csi_shape in self.csi_shapes.items():
                mkdata.save_path = f"{self.save_path}{self.dataset_name}_{csi_len}/"
                mkdata.csi_shape = csi_shape
                mkdata.export_data(filter=True)
                mkdata.save_data()
