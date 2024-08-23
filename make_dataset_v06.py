import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import wraps
import pandas as pd
import os
from scipy import signal

import cv2
import pycsi
import pyrealsense2 as rs
import yaml
from rosbag.bag import Bag

from tqdm.notebook import tqdm
from IPython.display import display, clear_output


##############################################################################
# -------------------------------------------------------------------------- #
# ABOUT TIMESTAMPS
# There are 3 kinds of timestamps:
# CSI timestamp -- local timestamp -- camera timestamp
#
# Timestamps:
# Label: ms -> s
# CSI: ns -> s
# Local: ms -> s
# Camera: 
#
# Later versions:
# BAG file recorded by realsense SDK will not go with local timestamps!
# -------------------------------------------------------------------------- #
##############################################################################
modalities = {'rimg': (1, 128, 226),
              'csi': (6, 30, 'csi_len'),
              'time': (1, 1),
              'cimg': (1, 128, 128),
              'center': (1, 2),
              'bbx': (1, 4),
              'depth': (1, 1),
              'pd': (1, 2, 'csi_len'),
              'tag': (1, 4)
              }

ver = 'V06'

class RawData:
    
    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, value):
        self._images = value
        
    @property
    def frame_timestamp(self):
        return self._frame_timestamp

    @frame_timestamp.setter
    def frame_timestamp(self, value):
        self._frame_timestamp = value
        
    @property
    def csi(self):
        return self._csi

    @csi.setter
    def csi(self, value):
        self._csi = value
    

class BagLoader:
    def __init__(self, bag_path, local_time_path=None, img_size=(226, 128)):
        self.bag_path = bag_path
        self.local_time_path = local_time_path
        self.img_shape = img_size
        self.jupyter_mode = False
        self.rawdata = RawData()
        
        # Get total frames
        print('Retrieving bag infomation...')
        info_dict = yaml.load(Bag(bag_path, 'r')._get_yaml_info())
        self.duration = info_dict['duration']
        self.num_frames = np.ceil(self.duration * 30).astype(int)
        
        print(f'Loaded bag file of {self.duration} seconds - {self.num_frames} frames (est)')
                
        self.images = np.zeros((self.num_frames, *self.img_shape[::-1]))
        self.camera_time = np.zeros(self.num_frames)
        self.camtime_delta = 0.
                
        if local_time_path:
            with open(local_time_path, mode='r', encoding='utf-8') as f:
                local_time = np.array(f.readlines())
                for i, line in enumerate(local_time):
                    local_time[i] = datetime.timestamp(datetime.strptime(line.strip(), "%Y-%m-%d %H:%M:%S.%f"))
            self.local_time = local_time
            self.calibrated = False
            
        self.config = rs.config()
        self.config.enable_device_from_file(self.bag_path, False)
        self.config.enable_all_streams()
    
    @staticmethod
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

    def reset_data(self):
        self.images = self.rawdata.images
        self.camera_time = self.rawdata.frame_timstamps
        if self.local_time_path:
            self.calibrate_camtime()

    def export_images(self, mode='depth', depth_filter=True, show_img=False):
        tqdm.write('Starting exporting image...')
        e = ''
        
        # Pipeline
        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)
        profile.get_device().as_playback().set_real_time(False)

        try:
            for i in tqdm(range(self.num_frames)):
                frame = pipeline.wait_for_frames()

                if mode == 'depth':
                    _frame = frame.get_depth_frame()
                    if not _frame:
                        continue
                    if depth_filter:
                        _frame = self.my_filter(_frame)

                elif mode == 'color':
                    _frame = frame.get_color_frame()
                    if not color_frame:
                        continue
                    
                frame_timestamp = _frame.get_timestamp() * 1.e-3
                image = np.asanyarray(_frame.get_data())
                image = cv2.resize(image, self.img_shape, interpolation=cv2.INTER_AREA)
                
                self.images[i] = image
                self.camera_time[i] = frame_timestamp
                
                if show_img:
                    self.playback(image=image, i=i, in_iter=True)
                
        except RuntimeError as e:
            tqdm.write(f'Done exporting {i-1} frames, {e}')

        finally:
            pipeline.stop()
            redundancy = len(set(np.where(np.all(self.images==0, axis=-1))[0]))
            print(f'Removing redundant {redundancy} frames')
            self.images = self.images[:-redundancy]
            self.camera_time = self.camera_time[:-redundancy]
            self.num_frames -= redundancy
            self.rawdata.images = self.images.copy()
            self.rawdata.frame_timesatmps = self.camera_time.copy()
            
            if self.local_time_path:
                self.calibrate_camtime()
                
    def calibrate_camtime(self):
        print('Calibrating camera time against local time file...', end='')
        cvt = datetime.fromtimestamp
        assert len(self.camera_time) == len(self.local_time)
        
        camtime_delta = self.camera_time.squeeze() - self.local_time.squeeze()
        self.camera_time -= camtime_delta
        
        self.calibrated = True
        self.camtime_delta = np.mean(camtime_delta)
        self.rawdata.set_frame_time(self.camera_time)
        print(f'Done\nlag={self.camtime_delta}')
                
    def playback(self, image=None, i='', in_iter=False):
        def play(img):
            plt.clf()
            plt.title(f"Image {i} of {self.num_frames}")
            plt.axis('off')
            plt.imshow(img)
            plt.show()
            if self.jupyter_mode:
                clear_output(wait=True)
                # display(plt.gcf())'
            else:
                plt.pause(0.1)
        
        if in_iter:
            play(image)
        else:
            i = 0
            for _image in self.images:
                play(_image)
                i += 1
                
    def depth_mask(self, threshold=0.5):
        """
        Advice: depth_mask can be applied in either BagLoader or ImageLoader.
        """
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
        
    def threshold_depth(self, threshold=3000):
        print(f'Thresholding IMG within {threshold}...', end='')
        self.images[self.images > threshold] = threshold
        print('Done')
        
    def clear_excessive_depth(self, threshold=3000):
        print(f'Clearing depth > {threshold}...', end='')
        self.images[self.images > threshold] = 0
        print('Done')
        
    def normalize_depth(self):
        threshold = np.max(self.images)
        print(f'Normalizing depth within {threshold}...', end='')
        self.images /= float(threshold)
        print('Done')

    def save_images(self, save_path=None):
        print("Saving...", end='')
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            _, filename = os.path.split(self.bag_path)
            file, ext = os.path.splitext(filename)
            np.save(f"{save_path}{file}_img.npy", self.images)
            np.save(f"{save_path}{file}_camtime.npy", self.camera_time)
        else:
            np.save(f"../{os.path.splitext(os.path.basename(self.bag_path))[0]}_raw.npy", self.images)
            np.save(f"../{os.path.splitext(os.path.basename(self.bag_path))[0]}_camtime.npy", self.camera_time)
        print("Done")


class LabelParser:
    def __init__(self, name, label_path=None, *args, **kwargs):
        self.name = name
        self.label_path: str = label_path
        self.labels = None
        self.seg_labels = None
        if self.label_path:
            self.load_label()

    def load_label(self):
        print(f'{self.name} loading label...', end='')
        self.labels = pd.read_csv(self.label_path)
        self.labels.loc[:, ['start', 'end']] *= 1.e-3
        
        sample_labels = pd.DataFrame(columns=['group', 'segment', 'start', 'end', 'img_ind', 'csi_inds'], dtype=object)
        groups = set(self.labels.group.astype(int).values)
        for group in groups:
            segments = set(self.labels[self.labels['group']==group].segment.astype(int).values)
            for segment in segments:
                selected = self.labels[(self.labels['group']==group) & (self.labels['segment']==segment)]
                new_rec = [group, segment, selected['start'].values, selected['end'].values, [], [] ]
                seg_labels.loc[len(seg_labels)] = new_rec
        
        print('Done')


class ImageLoader:
    # Camera time should have been calibrated

    def __init__(self, name, img_path, camera_time_path, *args, **kwargs):
        self.name = name
        self.img_path = img_path
        self.camera_time_path = camera_time_path
        images, camera_time = self.load_images()
        self.images = images
        self.camera_time = camera_time
        
        self.bbx = None
        self.center = None
        self.depth = None
        self.c_img = None
        self.threshold_map = None
        
        self.raw = self.images.copy()

    def load_images(self):
        print(f'{self.name} loading IMG...')
        images = np.load(self.img_path)
        camera_time = np.load(self.camera_time_path)
        self.id = np.arange(len(images))
        print(f" Loaded images of {images.shape} as {images.dtype}")
        return images, camera_time
    
    def reset_data(self):
        self.images = self.raw.copy()
        print('Data reset!')
    
    def playback(self, jupyter_mode=False, compare=False):
        for i, image in enumerate(self.images):
            plt.clf()
            if compare:
                image = np.hstack((image, self.raw[i]))
                plt.title(f"<Masked, Raw> {i} of {len(self.images)}")
            else:
                plt.title(f"Image {i} of {len(self.images)}")
            plt.axis('off')
            plt.imshow(image)
            plt.show()
                
            if jupyter_mode:
                clear_output(wait=True)
                # display(plt.gcf())'
            else:
                plt.pause(0.1)

    def depth_mask(self, threshold=0.5, tmap=None):
        """
        Advice: depth_mask can be applied in either BagLoader or ImageLoader.
        """
        tqdm.write("Masking...")
        if tmap is not None:
            threshold = tmap
        else:   
            median = np.median(np.squeeze(self.images), axis=0)
            threshold = median * threshold
            
        plt.imshow(threshold / np.max(threshold))
        plt.title("Threshold map")
        plt.show()
        for i in tqdm(range(len(self.images))):
            mask = np.squeeze(self.images[i]) < threshold
            self.images[i] *= mask
        tqdm.write("Done")
        return threshold
        
    def threshold_depth(self, threshold=3000):
        print(f'Thresholding IMG within {threshold}...', end='')
        self.images[self.images > threshold] = threshold
        print('Done')
        
    def clear_excessive_depth(self, threshold=3000):
        print(f'Clearing depth > {threshold}...', end='')
        self.images[self.images > threshold] = 0
        print('Done')
        
    def normalize_depth(self):
        """
        To export depth, normalization is required.
        """
        threshold = np.max(self.images)
        print(f'Normalizing depth within {threshold}...', end='')
        self.images /= float(threshold)
        print('Done')

    def crop(self, min_area=0, show=False):
        """
        Calculate cropped images, bounding boxes, center coordinates, depth.\n
        :param min_area: 0
        :param show: whether show images with bounding boxes
        :return: bbx, center, depth, c_img
        """
        self.bbx = np.zeros((len(self.images), 4), dtype=float)
        self.center = np.zeros((len(self.images), 2), dtype=float)
        self.depth = np.zeros((len(self.images), 1), dtype=float)
        self.c_img = np.zeros((len(self.images), 1, 128, 128), dtype=float)

        tqdm.write(f"{self.name} calculating center coordinates and depth...")

        for i in tqdm(range(len(self.images))):
            img = np.squeeze(self.images[i]).astype('float32')
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
            print(f'{self.name} converting bbx...')
            # xywh to xyxy
            self.bbx[..., 2] += self.bbx[..., 0]
            self.bbx[..., 3] += self.bbx[..., 1]
            
            self.bbx[..., 0] /= float(w_scale)
            self.bbx[..., 2] /= float(w_scale)
            self.bbx[..., 1] /= float(h_scale)
            self.bbx[..., 3] /= float(h_scale)
        if self.center is not None:
            print(f'{self.name} converting center...')
            self.center[..., 0] /= float(w_scale)
            self.center[..., 1] /= float(h_scale)
        print('Done!')


class CSILoader:
    # CSI time should have been calibrated
    # CSI timestamps inside csi

    def __init__(self, name, csi_path, csi_configs: pycsi.MyConfigs, *args, **kwargs):
        self.name = name
        self.csi_path = csi_path
        self.csi_configs = csi_configs
        self.csi = self.load_csi()
        self.filtered_csi = None
        self.calibrated = False

    def load_csi(self):
        csi = pycsi.MyCsi(self.csi_configs, 'CSI', self.csi_path)
        csi.load_data(remove_sm=True)
        return csi
    
    def calibrate_csitime(self, lag):
        # lag should be in seconds
        # timestamps should be in milliseconds
        print(f'{self.name} calibrating CSI timestamps with {lag} seconds...', end='')
        self.csi.timestamps -= lag * 1.e3
        self.calibrated = True
        print('Done')
    
class MyDataMaker(ImageLoader, CSILoader, LabelParser):
    version = ver

    def __init__(self,
                 name=None,
                 csi_shape: tuple = (2, 100, 30, 3),
                 alignment: str = 'tail',
                 jupyter_mode=True,
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
        self.save_path = f'{save_path}{name}'
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
    def reshape_csi(csi, filter=True, pd=False):
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
                u1, s1, v1 = np.linalg.svd(csi_[i:i + win_len].transpose(2, 1, 0).
                                           reshape(-1, rx, sub * win_len), full_matrices=False)
                aoa = np.angle(u1[:, 0, 0].conj() * u1[:, 1, 0]).squeeze()
                u2, s2, v2 = np.linalg.svd(csi_[i:i + win_len].transpose(1, 2, 0).
                                           reshape(-1, sub, rx * win_len), full_matrices=False)
                tof = np.angle(u2[:, :-1, 0].conj() * u2[:, 1:, 0]).squeeze()
                aoatof[0, i] = aoa
                aoatof[1:, i] = tof

            aoatof[0, :] = np.unwrap(aoatof[0, :])
            aoatof[1:, :] = np.unwrap(aoatof[1:, :], axis=-1)
            return {'csi': csi, 'pd': aoatof}
        # elif pd == 'filter_aoa':
        #     k = 21
        #     csi_ = csi_real + 1.j * csi_imag
        #     length, sub, rx = csi_.shape
            
        #     u1, s1, v1 = np.linalg.svd(csi_.transpose(0, 2, 1), full_matrices=False)
        #     aoa = np.angle(u1[:, 1:, 0] * u1[:, :-1, 0].conj()).squeeze()
        #     aoa_md = np.zeros_like(aoa)
        #     for i in range(aoa.shape[1]):
        #         aoa_md[:, i] = signal.medfilt(aoa[:, i].real, k) + 1.j * signal.medfilt(tmp[:, i].imag, k)
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
        modalities['pd'] = (1, 2, self.csi_shape[1])

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

    def __init__(self,                  
                 img_path,
                 camera_time_path,
                 csi_path,
                 label_path,
                 save_pathsubs=None, jupyter=True, dataset_name=None,
                 csi_shape=(2, 100, 30, 3)
                 ):

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
            mkdata.crop()
            mkdata.convert_bbx_ctr()
            mkdata.convert_depth()
            for csi_len, csi_shape in self.csi_shapes.items():
                mkdata.save_path = f"{self.save_path}{self.dataset_name}_{csi_len}/"
                mkdata.csi_shape = csi_shape
                mkdata.export_data(filter=True)
                mkdata.save_data()


class FilterPD:
    versio = ver
    
    def __init__(self, datadir, save_path):
        self.datadir = datadir
        self.save_path = save_path
        self.data: dict = {}
        self.pd: dict = {}
        print('Loading...', end='')
        paths = os.walk(datadir)
        for path, _, file_lst in paths:
            for file_name in file_lst:
                file_name_, ext = os.path.splitext(file_name)
                if ext == '.npy':
                    Take, Group, Segment, modality = file_name_.split('_')
                    if modality == 'csi':
                        self.data[file_name.replace('csi', 'pd')] = np.load(os.path.join(path, file_name))
        
        print('Done')
        
    # TODO: ADD TOF VECTOR    
    def filter_pd(self, k=21):
        print('Calculating & filtering PD...', end='')
        for key, csi in self.data.items():
            csi = csi[:, :3] + 1.j * csi[:, 3:]
            u,s,v=np.linalg.svd(csi.reshape(csi.shape[0], 3, -1), full_matrices=False)
            pd=u[:,1:,0]*u[:,:-1,0].conj()
            md = np.zeros_like(pd)
            for i in range(md.shape[1]):
                md[:, i] = signal.medfilt(pd[:, i].real, k) + 1.j * signal.medfilt(pd[:, i].imag, k)
            # num_samples * 4 (real & imag & 2aoa)
            self.pd[key] = np.concatenate((md.real, md.imag), axis=-1).astype(np.float32)
        print('Done')
        
    def save(self):
        # Change filename to pd
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        print('Saving PD...', end='')
        for key, pd in self.pd.items():
            np.save(os.path.join(self.save_path, key), pd)
        print('Done')
        
        
class Segmentation:
    def __init__(self, labels, seg_labels, camtime, csitime, csi_length):
        self.labels = labels
        self.seg_labels = seg_labels
        self.camtime = camtime
        self.csitime = csitime
        self.csi_length = csi_length
                
    def __call__(self, alignment='tail'):
        print(f"{self.name} segmenting...\n")
        
        for _, group, segment, samples, inds in self.seg_labels.itertuples(name=None):
            start = self.labels.loc[(self.labels['group']==group) & (self.labels['segment']==segment)].start.values[0]
            end = self.labels.loc[(self.labels['group']==group) & (self.labels['segment']==segment)].end.values[0]
            
            start_img_id, end_img_id = np.searchsorted(self.camtime, [start, end])
            start_csi_id, end_csi_id = np.searchsorted(self.csitime, [start, end])
            
            # IDs in raw data
            img_id = np.arange(start_img_id, end_img_id, dtype=int)
            csi_id = np.searchsorted(self.csitime, self.camtime[img_id])
            
            # Mask of imgs
            selected = np.ones_like(img_id, dtype=bool)
            
            # Selected CSI
            selected_csi = []
            
            # Exclude out-of-segment imgs
            for i, (_img_id, _csi_id) in enumerate(zip(img_id, csi_id)):
                if self.alignment == 'head':
                    sample_csi_id = np.arange(_csi_id, _csi_id + self.csi_length, dtype=int)
                elif self.alignment == 'tail':
                    sample_csi_id = np.arange(_csi_id - self.csi_length, _csi_id, dtype=int)
                if sample_csi_id[0] < start_csi_id or sample_csi_id[-1] > end_csi_id:
                    selected[i] = False
                else:
                    seleceted_csi.append(sample_csi_id)

            self.seg_labels[_].img_ind = img_id[selected]
            self.seg_labels[_].csi_inds = selected_csi
            
            
class Make:
    def __init__(self, raw_img, raw_csi, csi_len):
        self.raw_img = raw_img
        self.raw_csi = raw_csi
        self.csi_len = csi_len
        self.seg_labels = None
        self.ret_data = {'rimg': None,
                         'cimg': None,
                         'bbx': None,
                         'center': None,
                         'depth': None,
                         'csi': None,
                         'pd': None
                         }
        
    @staticmethod
    def export_image(img,
                     min_area=0, 
                     cvt_bbx=True, cvt_center=True,
                     w_scale=226, h_scale=128):
        r_img = np.squeeze(img).astype('float32')
        c_img = np.zeros((128, 128))
        average_depth = 0
        x, y, w, h = 0, 0, 0, 0
        
        # Bounding box, Center, Cropped image
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
    
    @staticmethod
    def export_csi(csi, filter=True):
        """
        Reshape CSI into channel * sub * packet.\n
        :return: reshaped (+ filtered) CSI
        """
        
        def process(c):
            # packet * sub * rx
            if filter:
                csi_real = signal.savgol_filter(np.real(c), 21, 3, axis=0)  # denoise for real part
                csi_imag = signal.savgol_filter(np.imag(c), 21, 3, axis=0)  # denoise for imag part
            else:
                csi_real = np.real(csi)
                csi_imag = np.imag(csi)

            # packet * sub * (rx * 2)
            c = np.concatenate((csi_real, csi_imag), axis=-1)

            # (rx * 2) * sub * packet
            c = c.transpose(2, 1, 0)
            return c
        
        if csi.ndim == 4:
            ret = np.zeros_like(csi)
            for _csi in csi:
                ret[i] = process(csi)
        else:
            return process(csi)
    
    @staticmethod
    def filter_pd(segment_csi, k=21):
        def cal_pd(u):
            pd = u[:, 1:, 0] * u[:, :-1, 0].conj()
            md = np.zeros_like(pd)
            for i in range(md.shape[1]):
                md[:, i] = signal.medfilt(pd[:, i].real, k) + 1.j * signal.medfilt(pd[:, i].imag, k)
            
            return np.concatenate((md.real, md.imag), axis=-1).astype(np.float32)
        
        csi = segment_csi[:, :3] + 1.j * segment_csi[:, 3:]
        # csi_shape = (n_samples, 3, 30, csi_len)
        
        # AoA = segment_len * 4 (real & imag of 2)
        u, *_ =np.linalg.svd(csi.reshape(csi.shape[0], 3, -1), 
                             full_matrices=False)
        aoa = cal_pd(u)
        
        # ToF = segment_len * 58 (real & imag of 29)
        u, *_ =np.linalg.svd(csi.transpose(0, 2, 1, 3).reshape(csi.shape[0], 30, -1), 
                             full_matrices=False)
        tof = cal_pd(u)
        
        # Return = segment_len * 62
        return {'pd': np.concatenate((aoa, tof), axis=-1)}
    
    def __call__(self, seg_labels):
        for _, group, segment, img_ind, csi_inds in seg_labels.itertuples(name=None):
            seglen = len(img_ind)
            if seglen > 0:
                for _img_ind, _csi_inds in zip(img_ind, csi_inds):
                    img_ret = self.export_image(self.raw_img[_img_ind])
                    csi_ret = self.export_csi(self.raw_csi[_csi_inds])
                    
                    
        
        