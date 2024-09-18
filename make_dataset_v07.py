import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import wraps
import pandas as pd
import os
import time

import cv2
import pycsi
import pyrealsense2 as rs
import yaml
from rosbag.bag import Bag
from scipy import signal
import cupy as cp
from joblib import Parallel, delayed

from tqdm.notebook import tqdm
from IPython.display import display, clear_output
from ipywidgets import interact
import subprocess


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

ver = 'V07'

def timer(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Total time:", end - start, "sec")
        return result

    return wrapper


class MyFilter:
    def __init__(self):
        pass
        
    def __call__(self, frame):
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

class Raw:
    def __init__(self, value):
        self._value = value.copy()
        self._value.setflags(write=False)
        
    # Make sure to use copy() when assigning values!
    @property
    def value(self):
        ret = self._value.copy()
        ret.setflags(write=True)
        return ret
    
class DepthMask:
    def __init__(self):
        self.threshold = None
    
    def __call__(self, images, threshold, tmap=None):
        print("Masking...", end='')
        if tmap is not None:
            threshold = tmap
        else:   
            median = np.median(np.squeeze(images), axis=0)
            threshold = median * threshold
        self.threshold = threshold
        for i in range(len(images)):
            mask = np.squeeze(images[i]) < threshold
            images[i] *= mask
        print("Done")

        if tmap is None:
            plt.imshow(threshold)
            plt.title("Threshold map")
            plt.axis('off')
            plt.show()
        return images
    
class Crop:
    def __init__(self):
        self.bbx = None
        self.center = None
        self.depth = None
        self.cimg = None
    
    def __call__(self, images, min_area=0, cvt_bbx=True, show=False):
        """
        Calculate cropped images, bounding boxes, center coordinates, depth.\n
        :param min_area: 0
        :param show: whether show images with bounding boxes
        :return: bbx, center, depth, c_img
        """
        self.bbx = np.zeros((len(images), 4), dtype=float)
        self.center = np.zeros((len(images), 2), dtype=float)
        self.depth = np.zeros((len(images), 1), dtype=float)
        self.cimg = np.zeros((len(images), 1, 128, 128), dtype=float)

        for i in tqdm(range(len(images))):
            img = np.squeeze(images[i]).astype('float32')
            (T, timg) = cv2.threshold((img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                contour = max(contours, key=lambda x: cv2.contourArea(x))
                area = cv2.contourArea(contour)

                if area < min_area:
                    # print(area)
                    pass

                else:
                    self.depth[i] = np.mean(img[img != 0])
                    x, y, w, h = cv2.boundingRect(contour)
                    w = 128 if w > 128 else w
                    
                    cropped = img[y:y + h, x:x + w]

                    self.bbx[i] = x / 226, y / 128, w / 226, h / 128
                    if cvt_bbx:
                        self.bbx[i, 2] += self.bbx[i, 0]
                        self.bbx[i, 3] += self.bbx[i, 1]
                        
                    self.center[i] = (x + w / 2) / 226, (y + h / 2) / 128
                    
                    image = np.pad(cropped, pad_width=((64 - h // 2,
                                                        64 - h + h //2), 
                                                       (64 - w // 2,
                                                       64 - w + w // 2)),
                                   mode='constant',
                                   constant_values = 0
                                   )

                    self.cimg[i, 0] = image

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
                        
        return self.bbx, self.center, self.depth, self.cimg

class BagLoader:
    def __init__(self, bag_path, local_time_path=None, img_size=(226, 128)):
        self.bag_path = bag_path
        self.local_time_path = local_time_path
        self.img_shape = img_size
        self.jupyter_mode = True
        self.filter = MyFilter()
        self.raw_images = None
        self.depthmask = DepthMask()
        
        # Get total frames
        print(f'{os.path.basename(self.bag_path)} retrieving bag infomation...', end='')
        info_dict = yaml.load(Bag(bag_path, 'r')._get_yaml_info(), Loader=yaml.SafeLoader)
        self.duration = info_dict['duration']
        # self.num_frames = np.ceil(self.duration * 30).astype(int) # Not equal to actual number of frames
        print(f' duration = {self.duration} seconds')
                
        self.num_frames = 40000
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
        
    def reset_images(self):
        self.images = self.raw_images.value
        print('Data reset!')

    def export_images(self, mode='depth', depth_filter=True, show=False):
        
        # Pipeline
        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        
        #self.num_frames = int(30 * playback.get_duration().total_seconds())
        bar_format = '{desc}{percentage:3.0f}%|{bar}|[{elapsed}<{remaining}{postfix}]'
        last_timestamp = 0
        num_frames = 0
        
        with tqdm(total=self.duration, bar_format=bar_format) as _tqdm:
            _tqdm.set_description(f'{os.path.splitext(os.path.split(self.bag_path)[1])[0]} exporting')

            try:
                for i in (range(self.num_frames)):
                    frame = pipeline.wait_for_frames(10000)

                    if mode == 'depth':
                        _frame = frame.get_depth_frame()
                        _frame = self.filter(_frame) if depth_filter else _frame
                    elif mode == 'color':
                        _frame = frame.get_color_frame()
                        
                    if not _frame:
                        continue
                    
                    image = np.asanyarray(_frame.get_data())
                    image = cv2.resize(image, self.img_shape, interpolation=cv2.INTER_AREA)
                    
                    self.images[i] = image
                    self.camera_time[i] = _frame.get_timestamp() * 1.e-3
                    
                    if last_timestamp == 0:
                        last_timestamp = self.camera_time[i]

                    _tqdm.set_postfix({'Got frame': num_frames})
                    _tqdm.update(self.camera_time[i] - last_timestamp)
                    last_timestamp = self.camera_time[i]
                    num_frames += 1
                    
                    if show:
                        self.playback(image=image, i=i, in_iter=True)
                    
            except RuntimeError as e:
                tqdm.write(f'Done exporting {i-1} frames')
                if str(e) != "Frame didn't arrive within 10000":
                    tqdm.write(e)
        
            finally:
                pipeline.stop()
                
        redundancy = len(set(np.where(np.all(self.images==0, axis=-1))[0]))

        if redundancy > 0:
            self.images = self.images[:-redundancy]
            self.camera_time = self.camera_time[:-redundancy]
            self.num_frames -= redundancy
        self.raw_images = Raw(self.images)
        
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
            plt.imshow(img, cmap=plt.get_cmap('Blues'))
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
                
    def browse_images(self):
        n = len(self.images)
        def view_image(i):
            plt.imshow(self.images[i], cmap=plt.get_cmap('Blues'))
            plt.title(f"Image {i} of {self.num_frames}")
            plt.axis('off')
            plt.show()
        interact(view_image, i=(0, n-1))
                
    def depth_mask(self, threshold=0.5, tmap=None):
        self.images = self.depthmask(self.images, threshold, tmap)
        
    def threshold_depth(self, threshold=3000):
        print(f'Thresholding IMG within {threshold}...', end='')
        self.images[self.images > threshold] = threshold
        print('Done')
        
    def clear_excessive_depth(self, threshold=3000):
        print(f'Clearing depth > {threshold}...', end='')
        self.images[self.images > threshold] = 0
        print('Done')
        
    def normalize_depth(self, threshold=3000):
        print(f'Normalizing depth by {threshold}...', end='')
        self.images /= float(threshold)
        print('Done')

    def save_images(self, save_path=None):
        print("Saving...", end='')
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file, _ = os.path.splitext(os.path.split(self.bag_path)[1])
            np.save(f"{save_path}{file}-rimg.npy", self.images)
            np.save(f"{save_path}{file}-camtime.npy", self.camera_time)
        else:
            np.save(f"{self.bag_path[:-4]}-rimg.npy", self.images)
            np.save(f"{self.bag_path[:-4]}-camtime.npy", self.camera_time)
        print("Done")

class ImageLoader:
    # Camera time should have been calibrated

    def __init__(self, img_path, *args, **kwargs):
        _, self.name = os.path.split(img_path)
        self.img_path = img_path
        self.images = self.load_images()
        
        self.raw_images = Raw(self.images)
        
        self.bbx = None
        self.center = None
        self.depth = None
        self.cimg = None
        self.threshold_map = None
        
        self.depthmask = DepthMask()
        self.cropper = Crop()

    def load_images(self):
        print(f'{self.name} loading IMG...')
        images = np.load(self.img_path)
        print(f" Loaded images of {images.shape} as {images.dtype}")
        return images
    
    def reset_data(self):
        self.images = self.raw_images.value
        self.bbx = None
        self.center = None
        self.depth = None
        self.cimg = None
        self.threshold_map = None
        print('Data reset!')
    
    def playback(self, compare=False):
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
                
            if self.jupyter_mode:
                clear_output(wait=True)
                # display(plt.gcf())'
            else:
                plt.pause(0.1)

    def depth_mask(self, threshold=0.5, tmap=None):
        self.images = self.depthmask(self.images, threshold, tmap)
        
    def threshold_depth(self, threshold=3000):
        print(f'Thresholding IMG by {threshold}...', end='')
        self.images[self.images > threshold] = threshold
        print('Done')
        
    def clear_excessive_depth(self, threshold=3000):
        print(f'Clearing depth > {threshold}...', end='')
        self.images[self.images > threshold] = 0
        print('Done')
        
    def normalize_depth(self, threshold=3000):
        print(f'Normalizing depth by {threshold}...', end='')
        self.images /= float(threshold)
        print('Done')

    def crop(self, min_area=0, cvt_bbx=True, show=False):
        print('Cropping...', end='')
        self.bbx, self.center, self.depth, self.cimg = self.cropper(self.images, min_area, cvt_bbx, show)
        print('Done')
        
    def save_cropped(self):
        print('Saving...', end='')
        np.save(f"{self.img_path.replace('rimg', 'cimg')}", self.cimg)
        np.save(f"{self.img_path.replace('rimg', 'bbx')}", self.bbx)
        np.save(f"{self.img_path.replace('rimg', 'ctr')}", self.center)
        np.save(f"{self.img_path.replace('rimg', 'dpt')}", self.depth)
        print('Done')
        
    def save_images(self):
        print('Saving...', end='')
        np.save(self.img_path, self.images)
        print('Done')
        
    def browse_images(self):
        n = len(self.images)
        def view_image(i):
            plt.imshow(self.images[i], cmap=plt.get_cmap('Blues'))
            plt.title(f"Image {i} of {len(self.images)}")
            plt.axis('off')
            plt.show()
        interact(view_image, i=(0, n-1))


class CSILoader:
    # CSI time should have been calibrated
    # CSI timestamps inside csi

    def __init__(self, csi_path,  *args, **kwargs):
        self.csi_path = csi_path
        self.configs = pycsi.MyConfigs()
        self.configs.tx_rate = 0x1c113
        self.configs.ntx = 3
        self.csi = self.load_csi()
        self.filtered_csi = None

    def load_csi(self):
        csi = pycsi.MyCsi(self.configs, os.path.basename(self.csi_path), self.csi_path)
        csi.load_data(remove_sm=True)
        return csi
    
    def preprocess(self):
        self.csi.extract_dynamic(mode='overall-divide', ref='tx', ref_antenna=1)
        self.csi.extract_dynamic(mode='highpass')
        
    def save(self):
        np.save(f"{self.csi_path.replace('csio', 'csi')}", self.csi.csi)
        np.save(f"{self.csi_path.replace('csio', 'csitime')}", self.csi.timestamps)


class LabelLoader:
    def __init__(self, label_path, *args, **kwargs):
        # CSI and IMG timestamps are in the same path
        # Load labels and pair samples by timestamps for each environment
        
        self.label_path = label_path
        self.labels: dict = {}
        
        self.csi_time: dict = {}
        self.img_time: dict = {}
        
        self.columns = ['env', 'subject', 'bag', 'csi', 'group', 'segment', 'timestamp', 'img_inds', 'csi_inds']
        self.segment_labels = pd.DataFrame(columns=self.columns)
        
        self.alignment = 'tail'
        self.csi_len = 300

    def load_label(self):
        """
        Load labels at environment level (load all subjects)
        """
        print(f'Loading label...')
        paths = os.walk(self.label_path)
        for path, _, file_lst in paths:
            for file_name in file_lst:
                file_name_, ext = os.path.splitext(file_name)
                
                # Load Label
                # Label filename is <subject.csv>
                # Store the filename without ext
                if 'checkpoint' in file_name_ or 'matched' in file_name_:
                    continue
                
                if ext == '.csv':
                    sub = file_name_.replace('labels_', '')
                    self.labels[sub] = pd.read_csv(os.path.join(path, file_name), skip_blank_lines=True)
                    self.labels[sub].dropna(axis=0, how='all', inplace=True)
                    print(f' Loaded {file_name} of len {len(self.labels[sub])}')
                    # Adjust scale
                    # self.labels[sub].loc[:, ['start', 'end', 'open', 'shut']] # in ms
                    
                # Load IMG time and CSI time
                if ext == '.npy':
                    if 'camtime' in file_name:
                        self.img_time[file_name_.replace('-camtime', '')] = np.load(os.path.join(path, file_name)) * 1e3 # to ms
                        print(f" Loaded {file_name} of len {len(self.img_time[file_name_.replace('-camtime', '')])}")
                    elif 'csitime' in file_name:
                        self.csi_time[file_name_.replace('-csitime', '')] = np.load(os.path.join(path, file_name)) * 1e3 # to ms
                        print(f" Loaded {file_name} of len {len(self.csi_time[file_name_.replace('-csitime', '')])}")
                    
        
    def save(self):
        for subject in self.labels.keys():
            sub_labels = self.segment_labels.loc[(self.segment_labels['subject']==subject)]
            sub_labels.to_csv(os.path.join(self.label_path, f'matched-{subject}.csv'))
        print('Matched labels saved!')
        
    def match(self, drop_duplicate=True, n_jobs=40):
        """
        Match CSI and IMG by timestamps
        """
        tqdm.write(f"{self.label_path} matching ...\n")
        def process_row(row):
            i, env, sub, bag, csi, group, segment, start, end, open, shut, *_ = row

            start_img_id, end_img_id = np.searchsorted(self.img_time[bag], [start, end])
            img_id = np.arange(start_img_id, end_img_id, dtype=int)

            timestamp = self.img_time[bag][img_id]
            csi_id = np.searchsorted(self.csi_time[csi], timestamp)
            
            return env, sub, bag, csi, group, segment, timestamp, img_id, csi_id
        
        for subject in tqdm(self.labels.keys()):
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_row)(r) for r in tqdm(self.labels[subject].itertuples(), total=len(self.labels[subject]))
            )
            
            for env, sub, bag, csi, group, segment, timestamp, img_id, csi_id in results:
                data = {
                    'env'      : [env] * len(timestamp),
                    'subject'  : [sub] * len(timestamp),
                    'bag'      : [bag] * len(timestamp),
                    'csi'      : [csi] * len(timestamp),
                    'group'    : [group] * len(timestamp),
                    'segment'  : [segment] * len(timestamp),
                    'timestamp': timestamp.astype(int),
                    'img_inds'   : img_id,
                    'csi_inds'   : csi_id
                }
                rec = pd.DataFrame(data, columns=self.columns)
                self.segment_labels = pd.concat([self.segment_labels, rec], ignore_index=True)
    

        if drop_duplicate:
            print(f'Before dropping duplicate timestamps: {len(self.segment_labels)}')
            self.segment_labels = self.segment_labels.drop_duplicates(subset='timestamp')
            print(f'After dropping duplicate timestamps: {len(self.segment_labels)}')
            
        self.segment_labels = self.segment_labels.sort_values(by='img_inds')
        tqdm.write('Environment matching done')
        
        
class ImageSlicer:
    def __init__(self, img_path):
        self.img_path = img_path
    
    def load(self):
        print(f'Loading ...', end='')
        paths = os.walk(self.img_path)
        for path, _, file_lst in paths:
            for file_name in file_lst:
                file_name_, ext = os.path.splitext(file_name)
                
                # Load Label
                # Label filename is <subject.csv>
                # Store the filename without ext
                if 'checkpoint' in file_name_ or 'matched' in file_name_:
                    continue
                
                if ext == '.csv':
                    sub = file_name_.replace('labels_', '')
                    self.labels[sub] = pd.read_csv(os.path.join(path, file_name), skip_blank_lines=True)
                    self.labels[sub].dropna(axis=0, how='all', inplace=True)
                    # Adjust scale
                    self.labels[sub].loc[:, ['start', 'end', 'open', 'shut']] *= 1.e-3
                    
                # Load IMG time and CSI time
                if ext == '.npy':
                    if 'camtime' in file_name:
                        self.img_time[file_name_.replace('-camtime', '')] = np.load(os.path.join(path, file_name))
                        
                    elif 'csitime' in file_name:
                        self.csi_time[file_name_.replace('-csitime', '')] = np.load(os.path.join(path, file_name))
        print('Done')
        
        
class FilterPD_P:
    def __init__(self, k=21):
        self.k = k
    
    def _cal_pd_vectorized(self, u):
        pd = u[:, 1:, 0] * u[:, :-1, 0].conj()
        md_real = cp.zeros_like(pd.real)
        md_imag = cp.zeros_like(pd.imag)
        
        for i in range(pd.shape[1]):
            md_real[:, i] = cp.asarray(signal.medfilt(cp.asnumpy(pd[:, i].real), self.k))
            md_imag[:, i] = cp.asarray(signal.medfilt(cp.asnumpy(pd[:, i].imag), self.k))

        return cp.concatenate((md_real, md_imag), axis=-1).astype(cp.float32)
    
    @timer
    def __call__(self, csi):

        # Convert CSI to CuPy array for GPU acceleration
        csi = cp.asarray(csi)
        
        # CSI shape = seglen * 300 * 30 * 3
        # Reshape into seglen * 3 * (30 * 300)
        u, _, _ = cp.linalg.svd(csi.transpose(0, 3, 2, 1).reshape(csi.shape[0], 3, -1), full_matrices=False)
        aoa = self._cal_pd_vectorized(u)
        
        # Reshape into seglen * 30 * (3 * 300)
        u, _, _ = cp.linalg.svd(csi.transpose(0, 2, 3, 1).reshape(csi.shape[0], 30, -1), full_matrices=False)
        tof = self._cal_pd_vectorized(u)
        
        # AoA = batch * 4 (real & imag of 2)
        # ToF = batch * 58 (real & imag of 29)
        pd = cp.concatenate((aoa, tof), axis=-1)
        pd = cp.asnumpy(pd)  # Convert back to NumPy array if needed

        return pd

class FilterPD:
    def __init__(self, k=21, njobs=20):
        self.k = k
        self.n_jobs = njobs
    
    def __call__(self, csi):
        def cal_pd(u):
            pd = u[:, 1:, 0] * u[:, :-1, 0].conj()
            
            # md = np.zeros_like(pd, dtype=np.cfloat)
            def filter_real_imag(index):

                real_filtered = signal.medfilt(pd[:, index].real, self.k)
                imag_filtered = signal.medfilt(pd[:, index].imag, self.k)

                return real_filtered, imag_filtered
                
            filtered_results = Parallel(n_jobs=self.n_jobs)(delayed(filter_real_imag)(i) for i in range(pd.shape[1]))
            real_filtered = np.stack([res[0] for res in filtered_results], axis=1)
            imag_filtered = np.stack([res[1] for res in filtered_results], axis=1)
            
            return np.concatenate((real_filtered, imag_filtered), axis=-1)
        
        def cal_svd(csi, permute_order):
            u, *_ = np.linalg.svd(csi.transpose(permute_order).reshape(csi.shape[0], csi.shape[permute_order[1]], -1), full_matrices=False)
            return u

        # CSI shape = seglen * 300 * 30 * 3
        # Reshape into seglen * 3 * (30 * 300)
        # csi = cp.asarray(csi)
        # print(f'\rcupy memory used: {cp.get_default_memory_pool().used_bytes() / 1e9} GB', end='')
        # u, _, _ = cp.linalg.svd(csi.transpose(0, 3, 2, 1).reshape(csi.shape[0], 3, -1), full_matrices=False)
        # u = cp.asnumpy(u)
        # aoa = cal_pd(u)
        
        # # Reshape into seglen * 30 * (3 * 300)
        # u, _, _ = cp.linalg.svd(csi.transpose(0, 2, 3, 1).reshape(csi.shape[0], 30, -1), full_matrices=False)
        # u = cp.asnumpy(u)
        # tof = cal_pd(u)
        
        # del csi
        # cp._default_memory_pool.free_all_blocks()
        
        # AoA = batch * 4 (real & imag of 2)
        # ToF = batch * 58 (real & imag of 29)
        # pd = np.concatenate((aoa, tof), axis=-1)

        u = cal_svd(csi, (0, 3, 2, 1))
        aoa = cal_pd(u)
        
        # Reshape into seglen * 30 * (3 * 300)

        u = cal_svd(csi, (0, 2, 3, 1))
        tof = cal_pd(u)
        
        # AoA = batch * 4 (real & imag of 2)
        # ToF = batch * 58 (real & imag of 29)
        pd = np.concatenate((aoa, tof), axis=-1)
        
        return pd
    

        
        
class FilterCSIPD:
    def __init__(self, path):
        self.path = path
        self.labels: dict = {}
        self.csi: dict = {}
        self.csitime: dict = {}
        self.phase: dict = {}
        self.filtered_csi: dict = {}
        self.filterpd = FilterPD()
        
    def load(self):
        paths = os.walk(self.path)
        print(f'Loading {self.path}...\n')
        for path, _, file_lst in paths:
            for file_name in file_lst:
                file_name_, ext = os.path.splitext(file_name)
                
                if 'checkpoint' in file_name_ or 'matched' in file_name_:
                    continue
                # Load Label <subject>=.csv
                if ext == '.csv':
                    sub = file_name_.replace('labels_', '')
                    self.labels[sub] = pd.read_csv(os.path.join(path, file_name), skip_blank_lines=True)
                    self.labels[sub].dropna(axis=0, how='all', inplace=True)
                    # Adjust scale
                    self.labels[sub].loc[:, ['start', 'end', 'open', 'shut']] # in ms
                    print(f'Loaded {file_name} of len {len(self.labels[sub])}')
                       
    @staticmethod 
    def filter_csi(c, n_jobs=30):
        def filter_part(i):

            # Apply Savitzky-Golay filter to real or imaginary part
            filtered_real = signal.savgol_filter(c[:, i].real, 21, 3, axis=0)
            filtered_imag = signal.savgol_filter(c[:, i].imag, 21, 3, axis=0)
            return filtered_real[:, np.newaxis] + 1.j * filtered_imag[:, np.newaxis]
        
        # Parallelize the filtering process
        results = Parallel(n_jobs=n_jobs)(
            delayed(filter_part)(i) for i in range(c.shape[1])
        )
        return np.concatenate(results, axis=1)
                        
    def preprocess(self, n_jobs=20):
        def p_filter(c, start, end):
            start_id, end_id = np.searchsorted(self.csitime[c], [start, end])
            if end_id - start_id < 30:
               start_id = end_id - 30

            try:
                filtered_csi_segment = self.filter_csi(self.csi[c][start_id: end_id])
            except Exception as e:
                print(f"Out-of-segment detected for {c}: start {start} - end {end}")
                filtered_csi_segment = None
                start_id = -1
                end_id = -1
            return c, start_id, end_id, filtered_csi_segment
        
        def p_pd(c, start, end):
            start_id, end_id = np.searchsorted(self.csitime[c], [start, end])
            if end_id - start_id < 300:
                start_id = end_id - 300
            temp_csi = np.array([self.filtered_csi[c][ind-300:ind] for ind in range(start_id, end_id)])         
            try:
                phase_segment = self.filterpd(temp_csi)
            except Exception as e:
                print(f"Out-of-segment detected for {c}: start {start} - end {end}")
                phase_segment = None
                start_id = -1
                end_id = -1

            return c, start_id, end_id, phase_segment

        for subject in tqdm(self.labels, desc='In-subject'):
            # segment-level
            self.csi: dict = {}
            self.csitime: dict = {}
            self.phase: dict = {}
            self.filtered_csi: dict = {}
            for c in set(self.labels[subject].loc[:, 'csi'].values):
                self.csi[c] = np.load(os.path.join(self.path, f'{c}-csi.npy'))[..., 0]
                self.csitime[c] = np.load(os.path.join(self.path, f'{c}-csitime.npy')) * 1e3
                self.phase[c] = np.zeros((self.csi[c].shape[0], 62), dtype=float)
                self.filtered_csi[c] = self.csi[c].copy()
                print(f"CSI for subject {subject}: {c}: {self.csi[c].shape}")

            # Run the processing loop in parallel
            filter_results = Parallel(n_jobs=n_jobs)(
                delayed(p_filter)(
                    c, start, end
                ) for i, env, sub, bag, c, group, segment, start, end, open, shut, *_ 
                in tqdm(self.labels[subject].itertuples(), total=len(self.labels[subject]), desc='In-segment filtering CSI')
            )
            
            for c, start_id, end_id, filtered_csi_segment in filter_results:
                if start_id != -1:
                    self.filtered_csi[c][start_id:end_id] = filtered_csi_segment
                    
            try:
                # Run the processing loop in parallel
                results = Parallel(n_jobs=n_jobs)(
                    delayed(p_pd)(
                        c, start, end
                    ) for i, env, sub, bag, c, group, segment, start, end, open, shut, *_ 
                    in tqdm(self.labels[subject].itertuples(), total=len(self.labels[subject]), desc='In-segment calculating PD')
                )
            except Exception as e:
                print(e)
                print(c)

            for c, start_id, end_id, phase_segment in results:
                if start_id != -1:
                    self.phase[c][start_id:end_id] = phase_segment
            
            for ph in self.phase:
                print(f'DONE {ph} of shape {self.phase[ph].shape}')
                np.save(os.path.join(self.path, f"{ph}-pd.npy"), self.phase[ph])
                np.save(os.path.join(self.path, f"{ph}-csi.npy"), self.filtered_csi[ph])