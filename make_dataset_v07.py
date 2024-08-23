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

ver = 'V07'

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
        return self._value
    
class DepthMask:
    def __init__(self):
        self.threshold = None
    
    def __call__(self, images, threshold, tmap=None):
        tqdm.write("Masking...")
        if tmap is not None:
            threshold = tmap
        else:   
            median = np.median(np.squeeze(images), axis=0)
            threshold = median * threshold
        self.threshold = threshold
            
        plt.imshow(threshold)
        plt.title("Threshold map")
        plt.show()
        for i in tqdm(range(len(images))):
            mask = np.squeeze(images[i]) < threshold
            images[i] *= mask
        tqdm.write("Done")
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

        tqdm.write(f"{self.name} calculating center coordinates and depth...")

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
                    cropped = img[y:y + h, x:x + w]

                    self.bbx[i] = x / 226, y / 128, w / 226, h / 128
                    if cvt_bbx:
                        self.bbx[i, 2] += self.bbx[i, 0]
                        self.bbx[i, 3] += self.bbx[i, 1]
                        
                    self.center[i] = (x + w / 2) / 226, (y + h / 2) / 128
                    
                    image = np.pad(cropped, pad_width=((64 - h // 2, 
                                                        64 - h + h //2),
                                                       (64 - w // 2,
                                                        64 - w + w // 2
                                                       )),
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
        self.jupyter_mode = False
        self.filter = MyFilter()
        self.raw_images = None
        self.depthmask = DepthMask()
        
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
        
    def reset_images(self):
        self.images = self.raw_images
        print('Data reset!')

    def export_images(self, mode='depth', depth_filter=True, show_img=False):
        tqdm.write('Starting exporting image...')
        
        # Pipeline
        pipeline = rs.pipeline()
        profile = pipeline.start(self.config)
        profile.get_device().as_playback().set_real_time(False)

        try:
            for i in tqdm(range(self.num_frames)):
                frame = pipeline.wait_for_frames()

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
                
                if show_img:
                    self.playback(image=image, i=i, in_iter=True)
                
        except RuntimeError as e:
            tqdm.write(f'Done exporting {i-1} frames {e}')

        finally:
            pipeline.stop()
            redundancy = len(set(np.where(np.all(self.images==0, axis=-1))[0]))
            print(f'Removing redundant {redundancy} frames')
            self.images = self.images[:-redundancy]
            self.raw_images = Raw(self.images)
            self.camera_time = self.camera_time[:-redundancy]
            self.num_frames -= redundancy
            
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
            _, filename = os.path.split(self.bag_path)
            file, ext = os.path.splitext(filename)
            np.save(f"{save_path}{file}_image.npy", self.images)
            np.save(f"{save_path}{file}_camtime.npy", self.camera_time)
        else:
            np.save(f"../{os.path.splitext(os.path.basename(self.bag_path))[0]}_image.npy", self.images)
            np.save(f"../{os.path.splitext(os.path.basename(self.bag_path))[0]}_camtime.npy", self.camera_time)
        print("Done")

class ImageLoader:
    # Camera time should have been calibrated

    def __init__(self, img_path, camera_time_path, *args, **kwargs):
        self.img_path = img_path
        _, self.name = os.path.split(img_path)
        
        self.camera_time_path = camera_time_path
        self.images, self.camera_time = self.load_images()
        
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
        camera_time = np.load(self.camera_time_path)
        self.basic_id = np.arange(len(images))
        print(f" Loaded images of {images.shape} as {images.dtype}")
        return images, camera_time
    
    def reset_data(self):
        self.images = self.raw_images.value
        self.bbx = None
        self.center = None
        self.depth = None
        self.cimg = None
        self.threshold_map = None
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

    def crop(self, min_area=0, show=False):
        self.bbx, self.center, self.depth, self.cimg = self.cropper(self.images)

    def convert(self, w_scale=226, h_scale=128):
        if self.bbx is not None:
            print(f'{self.name} converting bbx...')
            # xywh to xyxy & normalize
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

    def __init__(self, csi_path, csi_configs: pycsi.MyConfigs, *args, **kwargs):
        _, self.name = os.path.split(csi_path)
        self.csi_path = csi_path
        self.csi_configs = csi_configs
        self.csi = self.load_csi()
        self.filtered_csi = None

    def load_csi(self):
        csi = pycsi.MyCsi(self.csi_configs, 'CSI', self.csi_path)
        csi.load_data(remove_sm=True)
        return csi


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
                    