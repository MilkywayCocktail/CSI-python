import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageGen:
    def __init__(self, name, assemble_number=1):
        self.name = name
        self.assemble_number = assemble_number
        self.ind = []
        self.img_size = None
        self.raw_img = None
        self.raw_bbx = None
        self.gen_img = None
        self.gen_bbx = None
        self.depth = None
        self.patches = []
        self.bbx_order = 'xywh'

    def load_images(self, path):
        print(f"{self.name} loading images...")
        self.raw_img = np.load(path)
        self.raw_img = self.raw_img.reshape(-1, 1, 128, 226)
        self.raw_bbx = np.zeros((len(self.raw_img), self.assemble_number, 4))
        self.depth = np.zeros((len(self.raw_img), self.assemble_number, 1))
        length, channels, *self.img_size = self.raw_img.shape
        if channels != self.assemble_number:
            print(f"Attention: channels {channels} doesn't match assemble number {self.assemble_number}")
        print(f"Loaded img of {self.raw_img.shape} as {self.raw_img.dtype}")

    def print_shape(self):
        try:
            print(f"raw images: {self.raw_img.shape}")
            print(f"raw bbx: {self.raw_bbx.shape}")
            print(f"gen images: {self.gen_img.shape}")
            print(f"gen_bbx: {self.gen_bbx.shape}")
            print(f"depth: {self.depth.shape}")
        except Exception:
            pass

    def show_images(self, select_ind=None, select_num=8):
        """
        Currently not applicable for assemble_number >1\n
        :param select_ind: specify some indices to display. Default is None
        :param select_num: specify the number of images to display. Default is 8
        :return: None
        """
        if self.raw_img is not None:
            if select_ind:
                inds = np.array(select_ind)
            else:
                inds = np.random.choice(list(range(len(self.raw_img))), select_num, replace=False)

            fig = plt.figure(constrained_layout=True)
            axes = fig.subplots(1, select_num)
            if select_num == 1:
                axes = [axes]
                inds = [inds]
            else:
                inds = np.sort(inds)
            for j in range(len(axes)):
                img = axes[j].imshow(np.squeeze(self.raw_img[inds[j]]), vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{inds[j]}")
            fig.colorbar(img, ax=axes, shrink=0.8)
            plt.show()
        else:
            print("Please load images")

    def add_index(self, index):
        self.ind.append(index)
        print(f"Added index {index}")

    def bounding_box(self, min_area=100, show=False):
        print("Labeling bounding boxes...", end='')

        for i in range(len(self.raw_img)):
            for j in range(self.assemble_number):
                img = np.squeeze(self.raw_img[i][j]).astype('float32')
                (T, timg) = cv2.threshold((img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) != 0:
                    contour = max(contours, key=lambda x: cv2.contourArea(x))
                    area = cv2.contourArea(contour)

                    if area < min_area:
                        # print(area)
                        pass

                    else:
                        x, y, w, h = cv2.boundingRect(contour)
                        patch = img[y:y + h, x:x + w]
                        non_zero = (patch != 0)
                        average_depth = patch.sum() / non_zero.sum()
                        self.patches.append(patch)
                        self.depth[i][j] = average_depth
                        self.raw_bbx[i][j] = np.array([x, y, w, h])

                        img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                            (x, y),
                                            (x + w, y + h),
                                            (0, 255, 0), 1)
                        if show:
                            cv2.namedWindow('Raw Image', cv2.WINDOW_AUTOSIZE)
                            cv2.imshow('Raw Image', img)
                            key = cv2.waitKey(33) & 0xFF
                            if key == ord('q'):
                                break
        print("Complete!")

    def generate_imgs(self, Bx=None, By=None, HW=None, select_ind=None):
        """
        Currently not applicable for assemble_number > 1\n
        :param Bx: Variations to generate on x axis
        :param By: Variations to generate on y axis
        :param HW: Variations to generate of size
        :param select_ind: specify some images to reproduce. Default is None
        :return: None
        """
        if select_ind:
            ind = [select_ind]
        else:
            ind = self.ind

        if ind:
            for i in ind:
                print(f"Generating {i} of {ind}...")
                x, y, w, h = self.raw_bbx[i]
                x, y, w, h = int(x), int(y), int(w), int(h)
                subject = np.squeeze(self.raw_img[i])[y:y + h, x:x + w]

                plt.imshow(subject)
                plt.show()
                generated_images = np.zeros((1, 128, 128))
                generated_bbx = np.zeros((1, 5))

                if HW:
                    subject = cv2.resize(subject, (w*HW, h*HW), interpolation=cv2.INTER_AREA)
                    (h, w) = subject.shape
                if Bx:
                    Bxs = np.floor(np.arange(0, 128 - w, int(Bx))).astype(int)
                    print(f"{len(Bxs)} images to generate")
                    for Bxi in Bxs:
                        image = np.zeros((128, 128))
                        image[y:y + h, Bxi:Bxi + w] = subject
                        # bbx = np.array([Bxi, y, w, h, d])
                        bbx = np.array([Bxi, y, w, h])
                        generated_images = np.concatenate((generated_images, image.reshape((1, 128, 128))), axis=0)
                        generated_bbx = np.concatenate((generated_bbx, bbx.reshape(1, 5)), axis=0)

                if By:
                    Bys = np.floor(np.arange(0, 128 - h, int(By))).astype(int)
                    print(f"{len(Bys)} images to generate")
                    for Byi in Bys:
                        image = np.zeros((128, 128))
                        image[Byi:Byi + h, x:x + w] = subject
                        # bbx = np.array([x, Byi, w, h, d])
                        bbx = np.array([x, Byi, w, h])
                        generated_images = np.concatenate((generated_images, image.reshape((1, 128, 128))), axis=0)
                        generated_bbx = np.concatenate((generated_bbx, bbx.reshape(1, 5)), axis=0)

                generated_images = np.delete(generated_images, 0, axis=0)
                generated_bbx = np.delete(generated_bbx, 0, axis=0)

                if not self.gen_img:
                    self.gen_img = generated_images
                else:
                    self.gen_img = np.concatenate((self.gen_img, generated_images), axis=0)
                if not self.gen_bbx:
                    self.gen_bbx = generated_bbx
                else:
                    self.gen_bbx = np.concatenate((self.gen_bbx, generated_bbx), axis=0)

            print("Generation complete!")
        else:
            print("Please specify an index!")

    def align_to_center(self, unified_size=False):
        """
        Align the cropped images to center.
        :param unified_size: whether to unify the size of cropped images. Default is False
        :return: None
        """
        generated_images = np.zeros((1, self.assemble_number, 128, 128))
        generated_bbx = np.zeros((1, self.assemble_number, 4))
        tqdm.write('Starting exporting image...')
        for i in tqdm(range(len(self.raw_img))):
            img_anchor = np.zeros((1, 1, 128, 128))
            bbx_anchor = np.zeros((1, 1, 4))
            for j in range(self.assemble_number):
                x, y, w, h = self.raw_bbx[i][j]
                x, y, w, h = int(x), int(y), int(w), int(h)
                subject = np.squeeze(self.raw_img[i][j])[y:y + h, x:x + w]

                image = np.zeros((128, 128))
                if unified_size:
                    f1 = 128 / w
                    f2 = 128 / h
                    f = min(f1, f2)  # resizing factor
                    dim = (int(w * f), int(h * f))
                    subject = cv2.resize(subject, dim)
                    image[int(64 - dim[1] / 2):int(64 + dim[1] / 2), int(64 - dim[0] / 2):int(64 + dim[0] / 2)] = subject
                    bbx = np.array([int(64 - dim[0] / 2), int(64 - dim[1] / 2), dim[0], dim[1]])
                else:
                    image[y:y + h, int(64-w/2):int(64+w/2)] = subject
                    # bbx = np.array([int(64-w/2), y, w, h, d])
                    bbx = np.array([int(64 - w / 2), y, w, h])
                img_anchor = np.concatenate((img_anchor, image.reshape(1, 1, 128, 128)), axis=1)
                bbx_anchor = np.concatenate((bbx_anchor, bbx.reshape(1, 1, 4)), axis=1)

            img_anchor = np.delete(img_anchor, 0, axis=1)
            bbx_anchor = np.delete(bbx_anchor, 0, axis=1)
            generated_images = np.concatenate((generated_images, img_anchor), axis=0)
            generated_bbx = np.concatenate((generated_bbx, bbx_anchor), axis=0)

        generated_images = np.delete(generated_images, 0, axis=0)
        generated_bbx = np.delete(generated_bbx, 0, axis=0)

        if not self.gen_img:
            self.gen_img = generated_images
        else:
            self.gen_img = np.concatenate((self.gen_img, generated_images), axis=0)
        if not self.gen_bbx:
            self.gen_bbx = generated_bbx
        else:
            self.gen_bbx = np.concatenate((self.gen_bbx, generated_bbx), axis=0)

        print("Generation complete!")

    def view_generation(self, save_path=None):
        if self.gen_img is not None:
            print(f"Viewing generated {self.gen_img.shape[0]} images...\n")

            for i in range(len(self.gen_img)):
                print(f"\r{i} of {self.gen_img.shape[0]}", end='')
                x, y, w, h = self.gen_bbx[i]
                x, y, w, h = int(x), int(y), int(w), int(h)
                img = np.squeeze(self.gen_img[i]) * 255

                img = cv2.rectangle(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 255, 0), 1)
                cv2.namedWindow('Generated Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Generated Image', img)
                if save_path is not None:
                    cv2.imwrite(f"{save_path}{str(i).zfill(4)}.jpg", img)

                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
        else:
            print("No generated images!")

    def convert_bbx(self, w_scale=226, h_scale=128, bbx_order='xyxy'):
        print('Converting bbx...', end='')
        if bbx_order == 'xyxy':
            self.raw_bbx[..., -1] = self.raw_bbx[..., -1] + self.raw_bbx[..., -3]
            self.raw_bbx[..., -2] = self.raw_bbx[..., -2] + self.raw_bbx[..., -4]

        self.raw_bbx[..., -1] /= float(w_scale)
        self.raw_bbx[..., -3] /= float(w_scale)
        self.raw_bbx[..., -2] /= float(h_scale)
        self.raw_bbx[..., -4] /= float(h_scale)

        self.bbx_order = bbx_order
        print('Done')

    def convert_depth(self, threshold=3000):
        print('Converting depth...', end='')
        self.depth[self.depth > threshold] = threshold
        self.depth /= float(threshold)
        print('Done')

    def save(self, save_path, save_terms=('raw_bbx', 'gen_img', 'gen_bbx', 'depth')):
        print('Saving...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for save_term in save_terms:
            if eval(f"self.{save_term} is not None"):
                if save_term == 'raw_bbx':
                    np.save(f"{save_path}{self.name}_{self.bbx_order}_{save_term}.npy", eval(f"self.{save_term}"))
                else:
                    np.save(f"{save_path}{self.name}_{save_term}.npy", eval(f"self.{save_term}"))
                print(f"Saved {save_term}")
        print('All saved!')


if __name__ == '__main__':

    names = {'01'}
        #('01', '02', '03', '04')
    for name in names:
        print(name)
        gen = ImageGen(name)
        gen.load_images(f"../dataset/0509/make05/{name}_226_img.npy")
        gen.bounding_box(min_area=0, show=False)
        gen.align_to_center(unified_size=True)
        print(gen.depth[0])
        plt.subplot(1, 2, 1)
        plt.imshow(gen.patches[0])
        plt.subplot(1, 2, 2)
        plt.imshow(gen.gen_img[0][0])
        plt.show()

        #gen.print_len()
        #gen.save('../dataset/0509/make05-resize/', save_terms=('raw_bbx', 'gen_img'))
        #gen.view_generation()
    #gen.show_images(select_ind=[250, 300, 350, 200], select_num=4)

    #gen.generate_imgs(Bx=20, select_ind=250)
    #gen.view_generation("../dataset/0509/make01-finished/gen1")
    #bounding_box('../dataset/0725/make00-finished/img.npy')
