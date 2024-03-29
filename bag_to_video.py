import pyrealsense2 as rs
import numpy as np
import cv2
from tqdm import tqdm
import os


class BagConverter:

    def __init__(self, path, in_name, out_name, fps=30, f=True):
        self.path = path
        self.in_path = path + in_name
        self.out_path = path + out_name
        self.fps = fps
        self.filter = f
        self.pipeline = None
        self.config = None
        self.videowriter = None
        self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    def setstream(self):
        """
        Set stream params
        """

    def setup(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        rs.config.enable_device_from_file(self.config, self.in_path, False)
        self.setstream()

        self.pipeline.start(self.config)

    def get_frame(self, frames):
        """
        Return frame
        """
        return

    def apply_filter(self, frame):
        """
        Apply filter for depth frame
        """
        return frame

    def colorize(self, image):
        """
        Colorize frames
        """
        return image

    def run(self):
        try:

            while True:
                frames = self.pipeline.wait_for_frames()
                print("Frame claimed")
                frame = self.get_frame(frames)
                if not frame:
                    continue
                frame = self.apply_filter(frame)
                image = np.asanyarray(frame.get_data())
                image = self.colorize(image)
                self.videowriter.write(image)

                cv2.namedWindow('Bag Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Bag Image', image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except RuntimeError:
            print("Read finished!")
            self.pipeline.stop()

        finally:
            cv2.destroyAllWindows()
            self.videowriter.release()
            print("Video saved!")


def my_filter(frame):
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


class Bag2Color(BagConverter):
    def __init__(self, *args, **kwargs):
        BagConverter.__init__(self, *args, **kwargs)

    def setstream(self):
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, self.fps)
        self.videowriter = cv2.VideoWriter(self.out_path, self.fourcc, self.fps, (1280, 720))

    def get_frame(self, frames):
        color_frame = frames.get_color_frame()
        return color_frame

    def colorize(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class Bag2Depth(BagConverter):
    def __init__(self, *args, **kwargs):
        BagConverter.__init__(self, *args, **kwargs)

    def setstream(self):
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, self.fps)
        self.videowriter = cv2.VideoWriter(self.out_path, self.fourcc, self.fps, (848, 480))

    def get_frame(self,  frames):
        depth_frame = frames.get_depth_frame()
        return depth_frame

    def apply_filter(self, frame):
        frame = my_filter(frame) if self.filter else frame
        return frame

    def colorize(self, image):
        image = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.03), cv2.COLORMAP_BONE)
        return image


class Bag2Velocity(BagConverter):
    def __init__(self, *args, **kwargs):
        BagConverter.__init__(self, *args, **kwargs)

    def setstream(self):
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, self.fps)
        self.videowriter = cv2.VideoWriter(self.out_path, self.fourcc, self.fps, (848, 480))

    def get_frame(self,  frames):
        depth_frame = eval("frames.get_depth_frame()")
        return depth_frame

    def run(self):
        try:
            t_tmp = 0
            t1 = 0
            t_vmap = np.zeros((480, 848))

            while True:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_frame = my_filter(depth_frame) if self.filter is True else depth_frame
                depth_image = np.asanyarray(depth_frame.get_data())

                timestamp = frames.timestamp
                if t_tmp == 0:
                    t_tmp = timestamp
                timestamp = timestamp - t_tmp
                print(timestamp)

                vmap = depth_image - t_vmap
                if t1 != 0:
                    vmap = vmap / (timestamp - t1)
                t1 = timestamp


                vmap = cv2.convertScaleAbs(vmap, alpha=0.4)

                # vmap = cv2.bilateralFilter(vmap, 0, 100, 5)
                #vmap = cv2.blur(vmap, (15, 15))
                #vmap = cv2.bilateralFilter(vmap, 0, 100, 5)
                self.videowriter.write(vmap)

                t_vmap = depth_image

                cv2.namedWindow('Velocity Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Velocity Image', vmap)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except RuntimeError:
            print("Read finished!")
            self.pipeline.stop()

        finally:
            cv2.destroyAllWindows()
            self.videowriter.release()
            print("Video saved!")


class Bag2Masked:

    def __init__(self, path, in_name, out_name, length, fps=30, f=True, threshold=0.5):
        self.path = path
        self.in_path = path + in_name
        self.out_path = path + out_name
        self.length = length
        self.fps = fps
        self.filter = f
        self.threshold = threshold
        self.pipeline = None
        self.config = None
        self.result = np.zeros((self.length, 128, 128))

    def setstream(self):
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, self.fps)

    def setup(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        rs.config.enable_device_from_file(self.config, self.in_path, False)
        self.setstream()

        self.pipeline.start(self.config)

    @staticmethod
    def get_frame(frames):
        depth_frame = frames.get_depth_frame()
        return depth_frame

    def apply_filter(self, frame):
        frame = my_filter(frame) if self.filter else frame
        return frame

    def run(self):
        try:
            i = 0
            while True:
                frames = self.pipeline.wait_for_frames()
                print("Frame claimed")
                frame = self.get_frame(frames)
                if not frame:
                    continue
                frame = self.apply_filter(frame)
                image = np.asanyarray(frame.get_data())
                image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
                self.result[i] = image
                i += 1

        except RuntimeError:
            print("Read finished!")
            self.pipeline.stop()

        finally:
            cv2.destroyAllWindows()

            median = np.median(self.result, axis=0)
            threshold = median * self.threshold
            for i in tqdm(range(len(self.result))):
                mask = self.result[i] < threshold
                masked = self.result[i] * mask
                masked[masked > 3000] = 3000
                masked = masked / 3000.

                cv2.namedWindow('Bag Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Bag Image', masked)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break


if __name__ == '__main__':

    path = '../sense/0726/'
    source_name = '00.bag'
    export_name = '01.avi'

    con = Bag2Masked(path, source_name, export_name, f=True, length=6000, threshold=0.75)
    con.setup()
    con.run()
