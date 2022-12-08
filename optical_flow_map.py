import numpy as np
import cv2


class OpticalFlow:

    def __init__(self, file):
        self.file = file

    @staticmethod
    def draw_OF_map(flow, cflowmap, step, color):

        for x in range(0, flow.shape[1], step):
            for y in range(0, flow.shape[0], step):
                cv2.line(cflowmap, (x, y), (int(x+flow[y][x][0]), int(y+flow[y][x][1])), color)
                cv2.circle(cflowmap, (x, y), 2, color, -1)
        return cflowmap

    def run(self):
        print(self.file, "processing...")
        cap = cv2.VideoCapture(self.file)

        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(self.file[:-4] + "_OF.avi", fourcc, int(fps), (int(width), int(height)))

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        i = 0
        while True:
            ret, frame2 = cap.read()
            if ret is False:
                break
            nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            bgr = self.draw_OF_map(flow, frame2, 16, (0, 255, 0))
            # cv2.imwrite("GF_output.jpg",bgr)
            output_video.write(bgr)
            print("\r\033[32m{f} of {t} has been written!\033[0m".format(f=i, t=count), end='')
            i += 1
            prvs = nxt
        cap.release()


if __name__ == '__main__':

    of = OpticalFlow("../sense/1202/T04.avi")
    of.run()
