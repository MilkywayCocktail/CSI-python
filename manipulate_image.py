import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageGen:
    def __init__(self, name):
        self.name = name
        self.ind = []
        self.raw_imgs = None
        self.raw_bbx = None
        self.gen_imgs = None
        self.gen_bbx = None

    def load_images(self, path):
        print("Loading images...")
        self.raw_imgs = np.load(path)
        self.raw_bbx = np.zeros((len(self.raw_imgs), 4))
        print(f"Loaded img of {self.raw_imgs.shape} as {self.raw_imgs.dtype}")

    def print_len(self):
        print(f"raw images: {len(self.raw_imgs)}, raw bbx: {len(self.raw_bbx)}, gen images: {len(self.gen_imgs)}, "
              f"gen_bbx: {len(self.gen_bbx)}")

    def show_images(self, select_ind=None, select_num=8):
        if self.raw_imgs is not None:
            if select_ind:
                inds = np.array(select_ind)
            else:
                inds = np.random.choice(list(range(len(self.raw_imgs))), select_num, replace=False)

            fig = plt.figure(constrained_layout=True)
            axes = fig.subplots(1, select_num)
            if select_num == 1:
                axes = [axes]
                inds = [inds]
            else:
                inds = np.sort(inds)
            for j in range(len(axes)):
                img = axes[j].imshow(np.squeeze(self.raw_imgs[inds[j]]), vmin=0, vmax=1)
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
        imgs = np.squeeze(self.raw_imgs)

        for i in range(len(imgs)):
            img = np.squeeze(imgs[i]).astype('float32')
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
                    patch = np.average(img[y:y + h, x:x + w])
                    non_zero = (patch != 0)
                    average_depth = patch.sum() / non_zero.sum()
                    # self.raw_bbx[i] = np.array([x, y, w, h, average_depth])
                    self.raw_bbx[i] = np.array([x, y, w, h])

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
        if select_ind:
            ind = [select_ind]
        else:
            ind = self.ind

        if ind:
            for i in ind:
                print(f"Generating {i} of {ind}...")
                # x, y, w, h, d = self.raw_bbx[i]
                x, y, w, h = self.raw_bbx[i]
                x, y, w, h = int(x), int(y), int(w), int(h)
                subject = np.squeeze(self.raw_imgs[i])[y:y + h, x:x + w]

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

                if not self.gen_imgs:
                    self.gen_imgs = generated_images
                else:
                    self.gen_imgs = np.concatenate((self.gen_imgs, generated_images), axis=0)
                if not self.gen_bbx:
                    self.gen_bbx = generated_bbx
                else:
                    self.gen_bbx = np.concatenate((self.gen_bbx, generated_bbx), axis=0)

            print("Generation complete!")
        else:
            print("Please specify an index!")

    def align_to_center(self, unified_size=False):
        generated_images = np.zeros((1, 128, 128))
        generated_bbx = np.zeros((1, 4))
        tqdm.write('Starting exporting image...')
        for i in tqdm(range(len(self.raw_imgs))):
            # x, y, w, h, d = self.raw_bbx[i]
            x, y, w, h = self.raw_bbx[i]
            x, y, w, h = int(x), int(y), int(w), int(h)
            subject = np.squeeze(self.raw_imgs[i])[y:y + h, x:x + w]

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
            generated_images = np.concatenate((generated_images, image.reshape((1, 128, 128))), axis=0)
            generated_bbx = np.concatenate((generated_bbx, bbx.reshape(1, 4)), axis=0)

        generated_images = np.delete(generated_images, 0, axis=0)
        generated_bbx = np.delete(generated_bbx, 0, axis=0)

        if not self.gen_imgs:
            self.gen_imgs = generated_images
        else:
            self.gen_imgs = np.concatenate((self.gen_imgs, generated_images), axis=0)
        if not self.gen_bbx:
            self.gen_bbx = generated_bbx
        else:
            self.gen_bbx = np.concatenate((self.gen_bbx, generated_bbx), axis=0)

        print("Generation complete!")

    def view_generation(self, save_path=None):
        if self.gen_imgs is not None:
            print(f"Viewing generated {self.gen_imgs.shape[0]} images...\n")

            for i in range(len(self.gen_imgs)):
                print(f"\r{i} of {self.gen_imgs.shape[0]}", end='')
                # x, y, w, h, d = self.gen_bbx[i]
                x, y, w, h = self.gen_bbx[i]
                x, y, w, h = int(x), int(y), int(w), int(h)
                img = np.squeeze(self.gen_imgs[i]) * 255

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

    def save(self, save_path, save_terms=('raw_bbx', 'gen_img', 'gen_bbx')):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if 'raw_bbx' in save_terms and self.raw_bbx is not None:
            np.save(f"{save_path}{self.name}_raw_bbx.npy", self.raw_bbx)
            print("Saved raw_bbx")
        if 'gen_img' in save_terms and self.gen_imgs is not None:
            np.save(f"{save_path}{self.name}_gen_img.npy", self.gen_imgs)
            print("Saved gen_img")
        if 'gen_bbx' in save_terms and self.gen_bbx is not None:
            np.save(f"{save_path}{self.name}_gen_bbx.npy",  self.gen_bbx)
            print("Saved gen_bbx")


if __name__ == '__main__':

    names = ('01', '02', '03', '04')
    for name in names:
        print(name)
        gen = ImageGen(name)
        gen.load_images(f"../dataset/0509/make05/{name}_226_img.npy")
        gen.bounding_box(min_area=0, show=False)
        gen.align_to_center(unified_size=True)
        #gen.print_len()
        gen.save('../dataset/0509/make05-resize/', save_terms=('raw_bbx', 'gen_img'))
        #gen.view_generation()
    #gen.show_images(select_ind=[250, 300, 350, 200], select_num=4)

    #gen.generate_imgs(Bx=20, select_ind=250)
    #gen.view_generation("../dataset/0509/make01-finished/gen1")
    #bounding_box('../dataset/0725/make00-finished/img.npy')
