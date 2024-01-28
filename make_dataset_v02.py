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
        self.labels = None
        if self.__label_path:
            self.parse()

    def parse(self):
        print('Loading labels...', end='')
        labels = {}
        with open(self.__label_path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    key_list = line.strip().split(',')
                    for key in key_list:
                        labels[key] = []
                else:
                    line_list = line.split(',')
                    for ii, key in enumerate(key_list):
                        labels[key].append(eval(line_list[ii]))

        for key in labels.keys():
            labels[key] = np.array(labels[key])

        self.labels = labels
        print('Done')

    def add_condition(self):
        self.labels['direction'] = []
        for i in range(len(self.labels['start'])):
            if self.labels['x0'][i] == self.labels['x1'][i]:
                self.labels['direction'].append('y')
            elif self.labels['y0'][i] == self.labels['y1'][i]:
                self.labels['direction'].append('x')


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
        frames = self.video_stream.wait_for_frames()

        if mode == 'depth':
            depth_frame = frames.get_depth_frame()
            frame_timestamp = depth_frame.get_timestamp()
            if not depth_frame:
                eval('continue')
            depth_frame = my_filter(depth_frame)
            image = np.asanyarray(depth_frame.get_data())

        elif mode == 'color':
            color_frame = frames.get_color_frame()
            frame_timestamp = color_frame.get_timestamp()
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
#                           |------ 1
#                           ...
#               ...
# ----------------------------------------


class MyDataMaker(BagLoader, CSILoader, LabelParser):
    def __init__(self, total_frames: int,
                 csi_configs: pycsi.MyConfigs,
                 img_size: tuple = (128, 128),
                 csi_size: tuple = (90, 100),
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
        self.csi_size = csi_size
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

        csi = np.zeros((self.frames, 2, *self.csi_size))
        images = np.zeros((self.frames, self.img_size[1], self.img_size[0]))
        timestamps = np.zeros(self.frames)
        indices = np.zeros(self.frames, dtype=int)

        labels = [] if not self.labels else self.labels

        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices, 'labels': labels}

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

                self.result['vanilla']['tim'][i] = frame_timestamp
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

    def reshape_csi(self, csi_sample: numpy.ndarray):
        csi_sample = csi_sample.reshape(self.csi_length, 90).T
        return csi_sample

    def export_csi(self, window_dynamic=False, pick_tx=0):
        """
        Finds csi packets according to the timestamps of images.\n
        Requires export_image.\n
        """
        tqdm.write('Starting exporting CSI...')

        for i in tqdm(range(self.frames)):

            csi_index = np.searchsorted(self.csi.timestamps, self.result['vanilla']['tim'][i])
            self.result['vanilla']['ind'][i] = csi_index
            csi_sample = self.csi.csi[csi_index: csi_index + self.csi_length, :, :, pick_tx]
            if window_dynamic:
                csi_sample = self.reshape_csi(self.windowed_dynamic(csi_sample))
            else:
                csi_sample = self.reshape_csi(csi_sample)

            # Store in two channels
            self.result['vanilla']['csi'][i, 0, ...] = np.abs(csi_sample)
            self.result['vanilla']['csi'][i, 1, ...] = np.angle(csi_sample)

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
                    print(f"Found No.{ind} from results.")
                    plt.imshow(self.result['vanilla']['img'][ind])
                    plt.show()
                except Exception:
                    pass

    def slice_by_label(self):
        """
        Trim out labeled segments.\n
        Executed after exporting images and csi.\n
        :return: sliced results
        """
        print('Slicing...', end='')

        segments = {}
        changed_frames = 0
        for seg in range(len(self.labels['start'])):

            start_id = np.searchsorted(self.result['vanilla']['tim'], self.labels['start'][seg] - self.camtime_delta)
            end_id = np.searchsorted(self.result['vanilla']['tim'], self.labels['end'][seg] - self.camtime_delta)
            segments[seg] = np.arange(start_id, end_id)
            changed_frames += 1 + end_id - start_id
        print(segments.items())

        self.frames = changed_frames

        for types in self.result['vanilla'].keys():
            self.result['annotated'][types] = {}
            for seg in segments.keys():
                self.result['annotated'][types][seg] = {}
                if types != 'label':
                    self.result['annotated'][types][seg] = self.result['vanilla'][types][segments[seg]]
                else:
                    self.result['annotated'][types][seg] = [self.labels[seg]
                                                            for _ in range(segments[seg][0], segments[seg][-1])]

        print('Done')

    def assemble(self):
        print(f"Aligning into {self.frames} * {self.assemble_number}...", end='')
        if self.result['annotated']:
            for types in self.result.keys():
                for seg in self.result['annotated'][types].keys():
                    length, *shape = self.result['annotated'][types][seg].shape(0)
                    changed_length = length // self.assemble_number
                    self.result['annotated'][types][seg] = self.result['annotated'][types][seg].reshape(
                        changed_length, self.assemble_number, *shape)
        else:
            # Assemble vanilla data (ordinarily not needed)
            pass
        print("Done")

    def calibrate_camtime(self):
        """
        Calibrate camera timestamps against local timestamps. All timestamps are absolute.\n
        :return: result['tim']
        """
        print('Calibrating camera time against local time file...', end='')
        cvt = datetime.fromtimestamp

        if not self.caliberated:
            temp_lag = np.zeros(self.frames)
            for i in range(self.frames):
                temp_lag[i] = self.result['vanilla']['tim'][i] - self.local_time[i]

            camtime_delta = np.mean(temp_lag)

            for i in range(self.frames):
                self.result['vanilla']['tim'][i] = self.result['vanilla']['tim'][i] - camtime_delta
            self.caliberated = True
            self.camtime_delta = camtime_delta
            print('Done')
            print('lag={}'.format(camtime_delta))
        else:
            print("Already calibrated")

    def depth_mask(self, threshold=0.5):
        tqdm.write("Masking...")
        median = np.median(self.result['vanilla']['img'], axis=0)
        threshold = median * threshold
        plt.imshow(threshold / np.max(threshold))
        plt.title("Threshold map")
        plt.show()
        for i in tqdm(range(len(self.result['vanilla']['img']))):
            mask = self.result['vanilla']['img'][i] < threshold
            masked = self.result['vanilla']['img'][i] * mask
            self.result['vanilla']['img'][i] = masked
        tqdm.write("Done")

    def compress_image(self):
        print("Compressing...", end='')
        self.result['vanilla']['img'] = self.result['vanilla']['img'].astype(np.uint16)
        print("Done")

    def save_dataset(self, save_name, data='vanilla', *args):
        print("Saving...", end='')
        if not os.path.exists(self.paths['save']):
            os.makedirs(self.paths['save'])

        for types in args:
            if types in self.result[data].keys():
                np.save(os.path.join(self.paths['save'], f"{save_name}_{types}.npy"),
                        np.concatenate(self.result[data][types].values(), axis=0))
        print("Done")


class DataMakerV02(MyDataMaker):
    def __init__(self, *args, **kwargs):
        super(DataMakerV02, self).__init__(*args, **kwargs)

    def reshape_csi(self, csi_sample: numpy.ndarray):
        csi_sample = np.transpose(csi_sample, (0, 3, 2, 1))
        return csi_sample
