import pyrealsense2 as rs
import numpy as np
import cv2
import os


class BagConverter:

    def __init__(self, path, in_name, out_name, fps=30, f=True):
        self.path = path
        self.in_path = path + in_name
        self.out_path = path + out_name
        self.fps = fps
        self.filter = f
        self.pipeline = None
        self.videowriter = None

    def setup(self):
        pass

    def run(self):
        pass


class Bag2Color(BagConverter):
    def __init__(self, *args, **kwargs):
        BagConverter.__init__(self, *args, **kwargs)

    def setup(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, self.in_path, False)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, self.fps)
        self.pipeline.start(config)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.videowriter = cv2.VideoWriter(self.out_path, fourcc, self.fps, (1280, 720))

    def run(self):
        try:

            while True:
                frames = self.pipeline.wait_for_frames()
                print("Frame claimed")
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                self.videowriter.write(color_image)

                cv2.namedWindow('Bag Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Bag Image', color_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except RuntimeError:
            self.pipeline.stop()

        finally:
            cv2.destroyAllWindows()
            self.videowriter.release()
            print("Video saved!")


class Bag2Depth(BagConverter):
    def __init__(self, *args, **kwargs):
        BagConverter.__init__(self, *args, **kwargs)

    def setup(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, self.in_path, False)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, self.fps)
        self.pipeline.start(config)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.videowriter = cv2.VideoWriter(self.out_path, fourcc, self.fps, (848, 480))

    def run(self):
        try:
            if self.filter is True:
                hole_filling = rs.hole_filling_filter()

                decimate = rs.decimation_filter()
                decimate.set_option(rs.option.filter_magnitude, 1)

                spatial = rs.spatial_filter()
                spatial.set_option(rs.option.filter_magnitude, 1)
                spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
                spatial.set_option(rs.option.filter_smooth_delta, 50)

                depth_to_disparity = rs.disparity_transform(True)
                disparity_to_depth = rs.disparity_transform(False)

                def filters(frame):
                    filter_frame = decimate.process(frame)
                    filter_frame = depth_to_disparity.process(filter_frame)
                    filter_frame = spatial.process(filter_frame)
                    filter_frame = disparity_to_depth.process(filter_frame)
                    filter_frame = hole_filling.process(filter_frame)
                    result_frame = filter_frame.as_depth_frame()
                    return result_frame

            while True:
                frames = self.pipeline.wait_for_frames()
                print("Frame claimed")
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                depth_frame = filters(depth_frame) if self.filter is True else depth_frame
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                self.videowriter.write(depth_image)

                cv2.namedWindow('Bag Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Bag Image', depth_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except RuntimeError:
            self.pipeline.stop()

        finally:
            cv2.destroyAllWindows()
            self.videowriter.release()
            print("Video saved!")


if __name__ == '__main__':

    path = '../sense/1202/'
    source_name = 'T04.bag'
    export_name = 'T04_filtered.avi'

    con = Bag2Depth(path, source_name, export_name, f=True)
    con.setup()
    con.run()
