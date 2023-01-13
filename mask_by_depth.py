import numpy as np
import pyrealsense2 as rs
import cv2
import tqdm


def align_channels(frames, show=True):
    """
    Align color and depth
    """

    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    depth_frame_aligned = aligned_frames.get_depth_frame()
    color_frame_aligned = aligned_frames.get_color_frame()
    color_image_aligned = np.asanyarray(color_frame_aligned.get_data())
    color_image_aligned = cv2.cvtColor(color_image_aligned, cv2.COLOR_BGR2RGB)
    color_image_aligned = cv2.fastNlMeansDenoisingColored(color_image_aligned, None, 10, 10, 7, 15)
    depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())
    depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.05), cv2.COLORMAP_BONE)
    if show:
        images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
        cv2.imshow('aligned images', images_aligned)
        key = cv2.waitKey(1) & 0xFF
    return color_image_aligned


class MyMask:
    def __init__(self, algo, data):
        self.algo = self.__set_algo__(algo)
        self.data = self.__set_data__(data)

    def __set_algo__(self, algo):
        return

    def __set_data__(self, data):
        return

    def run(self, data):
        pass


class MyMaskbyMedian(MyMask):
    def __set_algo__(self, algo):
        return self.mask_by_depth

    def run(self, data):
        data = np.load(data)
        print(data.shape)

        median = np.median(data, axis=0)
        threshold = median * 0.5

        out = np.zeros_like(data)

        for i in tqdm.tqdm(range(len(data))):
            mask = data[i] < threshold
            masked = data[i] * mask
            out[i] = masked

        print(out[0])

    @staticmethod
    def mask_by_depth(data):
        median = np.median(data, axis=0)
        threshold = median * 0.5

        out = np.zeros_like(data)

        for i in tqdm.tqdm(range(len(data))):
            mask = data[i] < threshold
            masked = data[i] * mask
            out[i] = masked

        return out


class MyMaskbyCV(MyMask):

    def __set_algo__(self, algo):
        if algo == 'MOG':
            return cv2.bgsegm.createBackgroundSubtractorMOG()
        elif algo == 'MOG2':
            return cv2.createBackgroundSubtractorMOG2()
        elif algo == 'KNN':
            return cv2.createBackgroundSubtractorKNN()
        elif algo == 'GMG':
            return cv2.bgsegm.createBackgroundSubtractorGMG()

    def __set_data__(self, data):
        return

    def run(self, data):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(data, False)
        config.enable_all_streams()

        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)

        try:
            i = 0
            while True:
                frames = pipeline.wait_for_frames()
                if not frames:
                    continue
                depth_frame = frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
                # color_image = align_channels(frames, show=False)
                fgMask = self.algo.apply((depth_image/256).astype('uint8'))
                cv2.imshow('FG Mask', fgMask)
                key = cv2.waitKey(1) & 0xFF
                i += 1

        except RuntimeError:
            print("Read finished!")
            pipeline.stop()

        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    #in_path = '../dataset/compressed/121304.npy'
    #out_path = '../dataset/compressed/121304_masked.npy'
    masker = MyMaskbyCV('GMG', None)
    masker.run('../sense/1213/121304.bag')

