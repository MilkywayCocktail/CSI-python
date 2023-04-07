import cv2
import numpy as np
import os
from tqdm import tqdm

from make_dataset import MyDataMaker


class MyDataMaker_v01a(MyDataMaker):
    # Generates images, CSI, center coordinates

    def __init__(self, *args, **kwargs):
        MyDataMaker.__init__(*args, **kwargs)

    def __init_data__(self):
        # img_size = (width, height)
        csi = np.zeros((self.total_frames, 2, 90, self.configs.sample_length))
        images = np.zeros((self.total_frames, self.configs.img_size[1], self.configs.img_size[0]))
        timestamps = np.zeros(self.total_frames)
        indices = np.zeros(self.total_frames)
        coordinates = np.zeros((self.total_frames, 3))
        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices, 'cod': coordinates}

    def export_coordinate(self, show_img=False, min_area=50):
        """
        Requires export_image and depth_mask!\n
        :param show_img: whether to show the coordinate with the image
        :param min_area: a threshold set to filter out wrong bounding boxes
        """
        tqdm.write('Starting exporting coordinate...')
        areas = np.zeros(self.total_frames)
        for i in tqdm(range(self.total_frames)):
            img = None
            (T, timg) = cv2.threshold(self.result['img'][i].astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                contour = max(contours, key=lambda x: cv2.contourArea(x))
                areas[i] = cv2.contourArea(contour)

                if areas[i] < min_area:
                    print(areas[i])
                    self.result['cod'][i] = np.array([self.configs.img_size[1]//2, self.configs.img_size[0]//2, 0])
                    if show_img is True:
                        img = self.result['img'][i]
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    xc, yc = int(x + w / 2), int(y + h / 2)
                    self.result['cod'][i] = np.array([xc, yc, self.result['img'][i][yc, xc]])
                    if show_img is True:
                        img = cv2.rectangle(cv2.cvtColor(np.float32(self.result['img'][i]), cv2.COLOR_GRAY2BGR),
                                            (x, y),
                                            (x + w, y + h),
                                            (0, 255, 0), 1)
                        img = cv2.circle(img, (xc, yc), 1, (0, 0, 255), 4)
            else:
                img = self.result['img'][i]
                self.result['cod'][i] = np.array([self.configs.img_size[1]//2, self.configs.img_size[0]//2, 0])

            if show_img is True:
                cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Image', img)
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    break


if __name__ == '__main__':

    sub = '02'
    length = 1800

    path = [os.path.join('../sense/0124', sub + '.bag'),
            os.path.join('../sense/0124', sub + '_timestamps.txt'),
            os.path.join('../npsave/0124', '0124A' + sub + '-csio.npy'),
            os.path.join('../data/0124', 'csi0124A' + sub + '_time_mod.txt')]

    label02 = [(4.447, 7.315), (8.451, 11.352), (14.587, 18.59), (20.157, 22.16),
               (25.496, 29.397), (30.999, 33.767), (36.904, 40.473), (41.674, 44.108),
               (47.244, 51.046), (53.615, 55.983)]

    label03 = [(8.085, 10.987), (12.422, 14.79), (18.959, 21.862), (21.962, 24.963),
               (28.769, 31.902), (32.669, 36.206), (39.675, 42.611), (43.645, 47.147),
               (50.751, 53.485), (55.187, 57.988)]

    mkdata = MyDataMaker_v01a(path, length, (128, 128), sample_length=100)
    mkdata.export_image(show_img=False)
    mkdata.depth_mask()
    #mkdata.export_coordinate(show_img=True, min_area=1000)
    mkdata.export_csi()
    mkdata.slice_by_label(label02)
    mkdata.playback_image()
    mkdata.save_dataset('../dataset/0124/make02', sub + '_dyn', 'ind', 'csi', 'img', 'tim')

