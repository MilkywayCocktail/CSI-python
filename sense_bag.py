import cv2
import pyrealsense2 as rs
import numpy as np
import time
import os


class MySense:

    def __init__(self, savepath):
        self.length = 300
        self.savepath = savepath
        self.fps = 0
        self.height = 480
        self.width = 848
        self.visual_output = True   # whether to save images as .jpg and timestamps as .txt

        # used in session
        self.pipeline = None
        self.profile = None
        self.config = None

    def setup(self, **kwargs):

        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_all_streams()

    def run(self):
        print("\033[32mWelcome!" +
              "\nFrame size = " + str(self.length) + ' * ' + str(self.height) + ' * ' + str(self.width) +
              "\nSave path = " + self.savepath +
              "\033[0m")

        while True:
            print("Please enter a save name to start recording:\n",
                  "- Enter length=<length> to change length\n",
                  "- Enter exit to quit")

            accept_string = input()

            if accept_string == 'exit':
                print("\033[32mExiting...\033[0m")
                return

            elif accept_string[:7] == 'length=':
                if isinstance(eval(accept_string[7:]), int) and eval(accept_string[7:]) > 0:
                    self.length = eval(accept_string[7:])
                    print("\033[32mLength changed to " + str(self.length) + "\033[0m")

            else:
                name = accept_string
                with self._Recorder(self.pipeline, self.config, self.savepath, self.visual_output) as r_session:
                    r_session.record(name, self.length, self.height, self.width)

    class _Recorder:

        def __init__(self, pipeline, config, savepath, vis):
            self.pipeline = pipeline
            self.config = config
            self.savepath = savepath
            self.vis = vis

        def __enter__(self):
            print("\033[32mRecording in process...\033[0m")
            return self

        def record(self, name, length, height, width):

            timefile = None
            img_path = None

            self.config.enable_record_to_file(self.savepath + name + '.bag')
            profile = self.pipeline.start(self.config)

            if self.vis is True:
                timefile = open(path + name + '_timestamps.txt', mode='a', encoding='utf-8')
                timefile.write("Timestamps in miliseconds\n")
                img_path = self.savepath + name + '/'
                if not os.path.exists(img_path):
                    os.makedirs(img_path)

            t0 = int(round(time.time() * 1000))

            for i in range(length):

                frames = self.pipeline.wait_for_frames()
                t = int(round(time.time() * 1000)) - t0

                if self.vis is True:
                    #    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                    #    cv2.COLORMAP_JET)

                    img_name = str(i).zfill(5) + '.jpg'
                    depth_frame = frames.get_depth_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())
                    transfer_image = depth_image // 10
                    #    cv2.imwrite(path + 'img' + fname, depth_image)
                    cv2.imwrite(img_path + '_timg' + img_name, transfer_image)
                    timefile.write(str(t) + '\n')

                print('\r', i + 1, "of", length, "recorded", end='')

            if self.vis is True:
                timefile.close()

        def __exit__(self, exc_type, exc_val, exc_tb):

            if exc_type is not None:
                print(exc_type)
            else:
                self.pipeline.stop()
                print("\n\033[32mRecording complete!\033[0m")


if __name__ == '__main__':

    path = '../sense/'

    sense = MySense(path)
    sense.setup()
    sense.run()

