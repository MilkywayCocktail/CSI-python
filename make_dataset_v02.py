import numpy
import pyrealsense2 as rs
import cv2
import pandas
import numpy as np
import os
import pycsi
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
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


class BagLoader:
    def __init__(self, bag_path, localtime_path, img_size=(128, 128)):

        self.__bag_path = bag_path
        self.__localtime_path = localtime_path
        self.img_size = img_size
        self.video_stream = self.__setup_video_stream__()
        self.local_time = self.__load_local_timestamps__()

    def __setup_video_stream__(self):
        # timestamps is the local time; works as reference time
        print('Setting camera stream...', end='')
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.__bag_path, False)
        config.enable_all_streams()
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)
        print('Done')

        return pipeline

    def __load_local_timestamps__(self):
        local_tf = open(self.__localtime_path, mode='r', encoding='utf-8')
        local_time = np.array(local_tf.readlines())
        for i in range(len(local_time)):
            local_time[i] = datetime.timestamp(datetime.strptime(local_time[i].strip(), "%Y-%m-%d %H:%M:%S.%f"))
        local_tf.close()
        return local_time.astype(np.float64)

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


class CSILoader:
    def __init__(self, csi_path, csitime_path, csi_configs):
        self.__csi_path = csi_path
        self.__csitime_path = csitime_path
        self.csi_configs = csi_configs
        self.csi = self.__load_csi__()

    def __load_csi__(self):
        print('Loading CSI...')
        csi = pycsi.MyCsi(self.csi_configs, 'CSI', self.__csi_path)
        csi.load_data(remove_sm=True)

        csi_tf = open(self.__csitime_path, mode='r', encoding='utf-8')
        csi_timestamps = np.array(csi_tf.readlines())
        for i in range(len(csi_timestamps)):
            csi_timestamps[i] = datetime.timestamp(
                datetime.strptime(csi_timestamps[i].strip(), "%Y-%m-%d %H:%M:%S.%f"))
        csi_tf.close()

        csi.timestamps = csi_timestamps.astype(np.float64)

        return csi  # Pre-calibrated absolute CSI timestamp

    @staticmethod
    def windowed_dynamic(in_csi):
        # in_csi = np.squeeze(in_csi)
        phase_diff = in_csi * in_csi[..., 0][..., np.newaxis].conj().repeat(3, axis=2)
        static = np.mean(phase_diff, axis=0)
        dynamic = phase_diff - static
        return dynamic


# ------ Data structure of "result" ------
# result
#   |------ 'vanilla'
#               |------ 'img'
#               |------ 'csi
#               |------ 'tim
#               |------ 'ind
#               |------ 'label'
#   |------ 'annotated'
#               |------ 'img'
#                           |------ 0 (#segment)
#                                   |------ 0 (#channel)
#                                               |------ * (data size)
#                                               ...
#                           |------ 1
#                           ...
#               ...
# PS. To save as ndarrays, the {segment: data} structure is
# removed while saving.
# ----------------------------------------

class MyDataMaker(BagLoader, CSILoader, LabelParser):
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
                     'csitime': None,
                     'label': None,
                     'save': '/saved/'
                     }
        self.paths = paths
        self.frames = total_frames
        self.samples = total_frames // assemble_number
        self.csi_length = csi_length
        self.assemble_number = assemble_number
        self.alignment = alignment

        self.caliberated = False
        self.camtime_delta = 0.
        self.jupyter_mode = jupyter_mode

        BagLoader.__init__(self, self.paths['bag'], self.paths['localtime'], img_size)
        CSILoader.__init__(self, self.paths['csi'], self.paths['csitime'], csi_configs)
        LabelParser.__init__(self, self.paths['label'])

        self.result = {'vanilla': self.init_data(),
                       'annotated': {}}

    def init_data(self):
        # img_size = (width, height)

        csi = np.zeros((self.frames, 2, 30, 3))
        images = np.zeros((self.frames, 1, self.img_size[1], self.img_size[0]))
        timestamps = np.zeros((self.frames, 1, 1))
        indices = np.zeros((self.frames, 1, 1), dtype=int)

        return {'csi': csi, 'img': images, 'time': timestamps, 'ind': indices}

    def manual_load(self, types, path):
        print(f"Loading {types}...", end='')
        if types in self.result['vanilla'].keys():
            self.result[types] = np.load(path)
        print('Done')

    def playback(self, source='raw', mode='depth', display_size=(640, 480), save_name=None):
        print(f"Playback {source} {mode}...", end='')
        save_flag = False
        if save_name:
            save_name = f"{save_name}_{mode}_{source}.avi"
            save_flag = True
            if not os.path.exists(self.paths['save']):
                os.makedirs(self.paths['save'])

            if source == 'raw':
                if mode == 'depth':
                    img_size = (848, 480)
                elif mode == 'color':
                    img_size = (1280, 720)
            elif source == 'result':
                display_size = (display_size[0], display_size[1] * self.assemble_number)
                img_size = (self.result['annotated']['img'][0][0].shape[1] * self.assemble_number,
                            self.result['annotated']['img'][0][0].shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videowriter = cv2.VideoWriter(self.paths['save'] + save_name, fourcc, 10, img_size)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        try:
            for i in range(self.frames):
                if source == 'raw':
                    image, _ = self.__get_image__(mode=mode)
                    if mode == 'depth':
                        image = cv2.convertScaleAbs(image, alpha=0.02)
                elif source == 'result':
                    if self.assemble_number == 1:
                        image = np.squeeze(self.result['annotated']['img'][i])
                    else:
                        image = np.concatenate(np.squeeze(self.result['annotated']['img'][i].values()), axis=-1)
                else:
                    raise Exception('Please specify source = \'raw\' or \'result\'.')
                if save_flag is True:
                    videowriter.write(image)

                image = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
                # cv2.imshow('Image', image)
                # key = cv2.waitKey(33) & 0xFF
                # if key == ord('q'):
                #     break
                if self.jupyter_mode:
                    clear_output(wait=True)
                    plt.clf()
                    plt.imshow(image)
                    plt.title(f"Image {i} of {self.samples}")
                    # display(plt.gcf())'
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
            if source == 'raw':
                self.video_stream.stop()
            print("Done!")
            if save_flag:
                videowriter.release()

    def export_image(self, mode='depth', show_img=False):
        try:
            tqdm.write('Starting exporting image...')
            for i in tqdm(range(self.frames)):
                image, frame_timestamp = self.__get_image__(mode=mode)

                self.result['vanilla']['time'][i, ...] = frame_timestamp
                image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
                self.result['vanilla']['img'][i, ...] = image

                if show_img:
                    # cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
                    # cv2.imshow('Image', image)
                    # key = cv2.waitKey(33) & 0xFF
                    # if key == ord('q'):
                    #     break
                    # cv2.destroyAllWindows()
                    if self.jupyter_mode:
                        clear_output(wait=True)
                        plt.clf()
                        plt.imshow(image)
                        plt.title(f"Image {i} of {self.frames}")
                        # display(plt.gcf())'
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

    def reshape_csi(self):
        length, channel, *csi_shape = self.result['vanilla']['csi'].shape
        self.result['vanilla']['csi'] = self.result['vanilla']['csi'].reshape(
            length, channel, 100, 90).transpose((-1, -2))

    def export_csi(self, window_dynamic=False, pick_tx=0):
        """
        Finds csi packets according to the timestamps of images.\n
        Requires export_image.\n
        """
        tqdm.write('Starting exporting CSI...')
        boundary = [-1, -1]
        for i in tqdm(range(self.frames)):
            csi_index = np.searchsorted(self.csi.timestamps, self.result['vanilla']['time'][i, 0, 0])

            self.result['vanilla']['ind'][i, ...] = csi_index
            try:
                if csi_index > boundary[1]:
                    if self.alignment == 'head':
                        csi_sample = self.csi.csi[csi_index: csi_index + self.csi_length, :, :, pick_tx]
                    elif self.alignment == 'tail':
                        csi_sample = self.csi.csi[csi_index - self.csi_length: csi_index, :, :, pick_tx]

                    boundary = [csi_index,
                                csi_index + self.csi_length]

                if window_dynamic:
                    csi_sample = self.windowed_dynamic(csi_sample)

            except Exception:
                print(f"Error at {csi_index}, boundary={boundary}")

            # Store in two channels and reshape
            self.result['vanilla']['csi'][i, 0, ...] = np.abs(csi_sample)
            self.result['vanilla']['csi'][i, 1, ...] = np.angle(csi_sample)
            self.reshape_csi()

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
                    ind = np.searchsorted(self.result['vanilla']['time'][:, 0, 0], timestamp)
                    print(f"Found No.{ind} from results.")
                    plt.imshow(self.result['vanilla']['img'][ind])
                    plt.show()
                except Exception:
                    pass

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

        for types in self.result['vanilla'].keys():
            self.result['annotated'][types] = {seg: None for seg in segment.keys()}
            for seg in segment.keys():
                self.result['annotated'][types][seg] = self.result['vanilla'][types][segment[seg]]

        print('Done')

    def assemble(self):
        print("Aligning...")
        if self.result['annotated']:
            for types in self.result['vanilla'].keys():
                for seg in self.label['segment'].keys():
                    length, channel, *shape = self.result['annotated'][types][seg].shape
                    assemble_length = length // self.assemble_number
                    slice_length = assemble_length * self.assemble_number
                    self.result['annotated'][types][seg] = self.result['annotated'][types][seg][:slice_length].reshape(
                        assemble_length, self.assemble_number * channel, *shape)
        else:
            # Assemble vanilla data (usually not needed)
            pass
        print("Done")

    def calibrate_camtime(self):
        """
        Calibrate camera timestamps against local timestamps. All timestamps are absolute.\n
        :return: result['time']
        """
        print('Calibrating camera time against local time file...', end='')
        cvt = datetime.fromtimestamp

        if not self.caliberated:
            temp_lag = np.zeros(self.frames)
            for i in range(self.frames):
                temp_lag[i] = self.result['vanilla']['time'][i] - self.local_time[i]

            camtime_delta = np.mean(temp_lag)

            for i in range(self.frames):
                self.result['vanilla']['time'][i] = self.result['vanilla']['time'][i] - camtime_delta
            self.caliberated = True
            self.camtime_delta = camtime_delta
            print('Done')
            print('lag={}'.format(camtime_delta))
        else:
            print("Already calibrated")

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

    def compress_image(self):
        print("Compressing...", end='')
        self.result['vanilla']['img'] = self.result['vanilla']['img'].astype(np.uint16)
        print("Done")

    def save_dataset(self, save_name=None, data='vanilla', *args):
        print("Saving...", end='')
        if not os.path.exists(self.paths['save']):
            os.makedirs(self.paths['save'])

        for types in args:
            if types in self.result[data].keys():
                # if types in ('time', 'label'):
                #   np.save(os.path.join(self.paths['save'], f"{save_name}_{types}.npy"),
                #   self.result[data][types])
                # else:
                np.save(os.path.join(
                    self.paths['save'],f"{save_name}_asmb{self.assemble_number}_len{self.csi_length}_{types}.npy"),
                    np.concatenate(list(self.result[data][types].values())))
        print("Done")


class DataMakerV02(MyDataMaker):
    def __init__(self, *args, **kwargs):
        super(DataMakerV02, self).__init__(*args, **kwargs)

    def reshape_csi(self):
        length, channel, *csi_shape = self.result['vanilla']['csi'].shape
        self.result['vanilla']['csi'] = (self.result['vanilla']['csi'].transpose(
            (-1, -3))).reshape(length, channel * 3, 30, 30)
