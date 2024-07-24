import cv2
import pyrealsense2 as rs
import numpy as np
import datetime
import os
import keyboard


class MySense:

    def __init__(self, savepath):
        self.length = 72000  # (4min)
        self.savepath = savepath
        self.name = 0
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
        print(f"\033[32mWelcome!"
              f"\nFrame size = {self.width} * {self.height}"
              f"\nSave path = {self.savepath}\033[0m"
              )

        while True:
            print("Press P to start recording\n"
                  "Enter exit to exit")

            accept_string = input()

            if accept_string == 'exit':
                print("\033[32mExiting...\033[0m")
                return
            else:
                with self._Recorder(self.pipeline, self.config, self.savepath, self.visual_output) as r_session:
                    r_session.record(str(self.name), self.length, self.height, self.width)
                self.name += 1

    class _Recorder:

        def __init__(self, pipeline, config, savepath, localtime=True):
            self.pipeline = pipeline
            self.config = config
            self.savepath = savepath
            self.localtime = localtime

        def __enter__(self):
            print("\033[32mRecording in process...\033[0m")
            return self

        def record(self, name, length, height, width):

            timefile = None

            self.config.enable_record_to_file(f"{self.savepath}{name.zfill(2)}.bag")
            profile = self.pipeline.start(self.config)

            if self.localtime:
                timefile = open(path + name.zfill(2) + '_timestamps.txt', mode='w', encoding='utf-8')

            i = 0
            while True:

                frames = self.pipeline.wait_for_frames()
                timestamp = datetime.datetime.now()

                if self.localtime is True:
                    timefile.write(str(timestamp) + '\n')

                print(f"\r{i + 1} frames recorded", end='')
                i += 1

                if keyboard.is_pressed('q'):
                    print("\nStopping recording!")
                    break

            if self.localtime:
                timefile.close()

        def __exit__(self, exc_type, exc_val, exc_tb):

            if exc_type is not None:
                print(exc_type)
            else:
                self.pipeline.stop()
                print("\n\033[32mRecording complete!\033[0m")


if __name__ == '__main__':

    path = '../sense/0724/'

    sense = MySense(path)
    sense.setup()
    sense.run()

