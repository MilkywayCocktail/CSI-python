import numpy as np
import cv2


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class OpticalFlowEdge:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 30
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0

    def run(self):
        while True:
            _ret, frame = self.cam.read()
            if not _ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            print(self.frame_idx)
            #cv2.imwrite('videoOof-imgs/' + str(self.frame_idx) + '.jpg', cv2.resize(vis, (1280, 720)))
            self.frame_idx += 1

            ch = cv2.waitKey(1)
            if ch == 27:
                break


class OpticalFlowFeature:

    def __init__(self, file):
        self.file = file

    def run(self):
        cap = cv2.VideoCapture(self.file)

        # ShiTomasi角点检测的参数
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Lucas Kanada光流检测的参数
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # 构建随机颜色
        color = np.random.randint(0, 255, (100, 3))

        # 获取第一帧并发现角点
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # 为绘制光流追踪图，构建一个Mask
        mask = np.zeros_like(old_frame)

        num = 0
        while (1):
            ret, frame = cap.read()

            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 使用迭代Lucas Kanade方法计算稀疏特征集的光流
            # - old_gray: 上一帧单通道灰度图
            # - frame_gray: 下一帧单通道灰度图
            # - prePts：p0上一帧坐标pts
            # - nextPts: None
            # - winSize: 每个金字塔级别上搜索窗口的大小
            # - maxLevel: 最大金字塔层数
            # - criteria：指定迭代搜索算法的终止条件，在指定的最大迭代次数criteria.maxCount之后或搜索窗口移动小于criteria.epsilon
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # 选择轨迹点]
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # 绘制轨迹
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow('frame', img)
            # cv2.imwrite('../sense/1202/' + str(num) + '.jpg', cv2.resize(img, (1280, 720)))
            print(str(num))
            num = num + 1
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # 更新之前的帧和点
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cv2.destroyAllWindows()
        cap.release()


class OpticalFlowMap:

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


class OpticalFlowMask:

    def __init__(self, file):
        self.file = file

    def run(self, savepath):
        print(self.file, "processing...")
        cap = cv2.VideoCapture(self.file)

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        videowriter = cv2.VideoWriter(savepath, fourcc, 30, (1280, 720))

        while (1):
            ret, frame2 = cap.read()
            if not ret:
                break
            nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # 使用迭代Gunner Farneback 方法计算密集特征的光流
            # - prvs: previous grayscale frame
            # - next: next grayscale frame
            # - flow: None
            # - pyr_scale: 0.5经典金字塔，构建金字塔缩放scale
            # - level：3 No. of level of pyramid
            # - winsize：3 平均窗口大小，数值越大，算法对图像的鲁棒性越强
            # - iterations：15 迭代次数
            # - poly_n：5 像素邻域的参数多边形大小，用于在每个像素中找到多项式展开式；较大的值意味着图像将使用更平滑的曲面进行近似，从而产生更高的分辨率、鲁棒算法和更模糊的运动场；通常多边形n=5或7。
            # - poly_sigma：1.2 高斯标准差，用于平滑导数
            # - flags: 可以是以下操作标志的组合：OPTFLOW_USE_INITIAL_FLOW：使用输入流作为初始流近似值。OPTFLOW_FARNEBACK_GAUSSIAN: 使用GAUSSIAN过滤器而不是相同尺寸的盒过滤器；

            flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            cv2.imshow('frame2', bgr)
            videowriter.write(bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('../sense/opticalfb.png', frame2)
                cv2.imwrite('../sense/opticalhsv.png', bgr)
            prvs = nxt

        cap.release()
        cv2.destroyAllWindows()
        videowriter.release()


if __name__ == '__main__':

    of = OpticalFlowMask('../sense/1213/121303.avi')
    of.run('../sense/1213/121303_OFM.avi')
    print('Done')
    cv2.destroyAllWindows()