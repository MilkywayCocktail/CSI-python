import cv2
import pyrealsense2 as rs
import numpy as np
import time
import os


def depth2distance(depth_image, frame):
    dist_image = np.zeros_like(depth_image)
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            dist_image[y, x] = frame.get_distance(x, y) * 100     # distance in cm
    return dist_image


class MySense:

    def __init__(self, savepath):
        self.length = 100
        self.savepath = savepath
        self.fps = 0
        self.height = 480
        self.width = 848
        self.visual_output = True   # whether to save images as .jpg and timestamps as .txt

        # used in session
        self.pipeline = None
        self.profile = None

    def setup(self, **kwargs):

        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start()

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

    def run(self):

        while True:
            print("Please enter a save name to start recording:\n",
                  "- Enter length=<length> to change length\n",
                  "- Enter exit to quit")

            accept_string = input()

            if accept_string == 'exit':
                print("Exiting...")
                self.pipeline.stop()
                return

            elif accept_string[:7] == 'length=':
                if isinstance(eval(accept_string[7:]), int) and eval(accept_string[7:]) > 0:
                    self.length = eval(accept_string[7:])
                    print("Length changed to", self.length)

            else:
                name = accept_string
                with self._Recorder(self.pipeline, self.savepath, self.visual_output) as r_session:
                    r_session.record(name, self.length, self.height, self.width)

    class _Recorder:

        def __init__(self, pipeline, savepath, vis):
            self.pipeline = pipeline
            self.savepath = savepath
            self.vis = vis

        def __enter__(self):
            print("Recording in process...")
            return self

        def record(self, name, length, height, width):
            dmatrix = np.zeros((length, height, width))
            timestamps = np.zeros(length)
            timefile = None
            img_path = None

            if self.vis is True:
                timefile = open(path + name + '_timestamps.txt', mode='a', encoding='utf-8')
                timefile.write("Timestamps in miliseconds\n")
                img_path = self.savepath + name + '/'
                if not os.path.exists(img_path):
                    os.makedirs(img_path)

            t0 = int(round(time.time() * 1000))

            for i in range(length):

                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                t = int(round(time.time() * 1000)) - t0
                depth_image = np.asanyarray(depth_frame.get_data())

                dmatrix[i] = depth_image
                timestamps[i] = t

                if self.vis is True:
                    #    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                    #    cv2.COLORMAP_JET)

                    img_name = str(i).zfill(5) + '.jpg'
                    transfer_image = depth_image // 10
                    #    cv2.imwrite(path + 'img' + fname, depth_image)
                    cv2.imwrite(img_path + '_timg' + img_name, transfer_image)
                    timefile.write(str(t))

                print('\r', i + 1, "of", length, "recorded", end='')

            np.save(self.savepath + name + "_dmatrix.npy", dmatrix)
            np.save(self.savepath + name + "_timestamps.npy", timestamps)

            if self.vis is True:
                timefile.close()

        def __exit__(self, exc_type, exc_val, exc_tb):

            if exc_type is not None:
                print(exc_type)
            else:
                print("\nRecording complete!")


if __name__ == '__main__':

    path = '../sense/1124/'

    sense = MySense(path)
    sense.setup()
    sense.run()

