import cv2
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
import os
import keyboard


class MySense:

    def __init__(self, savepath='../sense'):
        self.length = 72000  # (4min)
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

        current_date = datetime.now()
        formatted_date = current_date.strftime('%Y%m%d')

        self.savepath = os.path.join(self.savepath, formatted_date)

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
                  "Press Q to stop recording\n"
                  "Enter exit to exit")

            accept_string = input()

            if accept_string == 'exit':
                print("\033[32mExiting...\033[0m")
                return
            elif accept_string == 'p':
                with self._Recorder(self.pipeline, self.config, self.savepath, self.visual_output) as r_session:
                    r_session.record()

    class _Recorder:

        def __init__(self, pipeline, config, savepath, localtime=True):
            self.pipeline = pipeline
            self.config = config
            self.savepath = savepath
            self.localtime = localtime
            current_time = datetime.now()
            formatted_time = current_time.strftime('%H%M%S')
            self.name = formatted_time

        def __enter__(self):
            print("\033[32mRecording in process...\033[0m")
            return self

        def record(self):

            timefile = None

            self.config.enable_record_to_file(os.path.join(self.savepath, f"{self.name}.bag"))
            profile = self.pipeline.start(self.config)

            if self.localtime:
                timefile = open(os.path.join(self.savepath, f"{self.name}_timestamps.txt"),
                                mode='w', encoding='utf-8')

            i = 0
            while True:

                frames = self.pipeline.wait_for_frames()
                timestamp = datetime.now()

                if self.localtime is True:
                    timefile.write(str(timestamp) + '\n')

                print(f"\r{self.name}: {i + 1} frames recorded", end='')
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

    sense = MySense()
    sense.setup()
    sense.run()

