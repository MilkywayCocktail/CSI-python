import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def bounding_box(in_path, min_area=100):
    imgs = np.load(in_path)
    out = np.zeros((len(imgs), 5))

    for i in range(len(imgs)):
        img = np.squeeze(imgs[i])
        (T, timg) = cv2.threshold((img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            area = cv2.contourArea(contour)

            if area < min_area:
                #print(area)
                pass

            else:
                x, y, w, h = cv2.boundingRect(contour)
                patch = np.average(img[y:y+h, x:x+w])
                non_zero = (patch != 0)
                average_depth = patch.sum() / non_zero.sum()
                print(average_depth * 3000)
                out[i] = np.array([average_depth, x, y, w, h])

                img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 255, 0), 1)

                cv2.namedWindow('IMAGE', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('IMAGE', img)

                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    break

    # np.save(f"{in_path[:-3]}ibv.npy", out)
    print(f"Saved!")

    depths = out[:, 0] * 3
    plt.plot(depths)
    plt.title("Average Depth Trace")
    plt.xlabel("#Frame")
    plt.ylabel("Depth/m")
    plt.show()


class ImageGen:
    def __init__(self, name):
        self.name = name
        self.ind = []
        self.raw_imgs = None
        self.raw_bbx = None
        self.gen_imgs = None
        self.gen_bbx = None

    def load_images(self, path):
        print("Loading images...", end='')
        self.raw_imgs = np.load(path)
        self.raw_bbx = np.zeros((len(self.raw_imgs), 5))
        print("Complete!")

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
                    self.raw_bbx[i] = np.array([x, y, w, h, average_depth])

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

    def generate_imgs(self, Bx=None, By=None, HW=None, overlap_ratio=1., select_ind=None):
        if select_ind:
            ind = [select_ind]
        else:
            ind = self.ind

        if ind:
            for i in ind:
                print(f"Generating {i} of {ind}...")
                x, y, w, h, d = self.raw_bbx[i]
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
                    Bxs = np.floor(np.arange(0, 128 - w, int(Bx * overlap_ratio))).astype(int)
                    print(f"{len(Bxs)} images to generate")
                    for Bxi in Bxs:
                        image = np.zeros((128, 128))
                        image[y:y + h, Bxi:Bxi + w] = subject
                        bbx = np.array([Bxi, y, w, h, d])
                        generated_images = np.concatenate((generated_images, image.reshape((1, 128, 128))), axis=0)
                        generated_bbx = np.concatenate((generated_bbx, bbx.reshape(1, 5)), axis=0)

                if By:
                    Bys = np.floor(np.arange(0, 128 - h, int(By * overlap_ratio))).astype(int)
                    print(f"{len(Bys)} images to generate")
                    for Byi in Bys:
                        image = np.zeros((128, 128))
                        image[Byi:Byi + h, x:x + w] = subject
                        bbx = np.array([x, Byi, w, h, d])
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

    def view_gen(self):
        for i in range(len(self.gen_imgs)):
            img = np.squeeze(self.gen_imgs[i])
            plt.imshow(img)
            plt.show()

    def view_generation(self):
        if self.gen_imgs is not None:
            print(f"Viewing generated {self.gen_imgs.shape[0]} images...")

            for i in range(len(self.gen_imgs)):
                x, y, w, h, d = self.gen_bbx[i]
                x, y, w, h = int(x), int(y), int(w), int(h)
                img = np.squeeze(self.gen_imgs[i]).astype('float32')
                img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 255, 0), 1)
                cv2.namedWindow('Generated Image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Generated Image', img)

                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break

        else:
            print("No generated images!")

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.raw_bbx:
            np.save(self.raw_bbx, f"{save_path}{self.name}_raw_bbx.npy")
        if self.gen_imgs:
            np.save(self.gen_imgs, f"{save_path}{self.name}_gen_imgs.npy")
        if self.gen_bbx:
            np.save(self.gen_bbx, f"{save_path}{self.name}_gen_bbx.npy")
        print("All saved!")


if __name__ == '__main__':

    gen = ImageGen("01")
    gen.load_images("../dataset/0509/make01-finished/img.npy")
    gen.bounding_box(show=False)
    gen.show_images(select_ind=250, select_num=1)
    gen.generate_imgs(Bx=10, overlap_ratio=0.5, select_ind=250)
    gen.view_generation()
    #bounding_box('../dataset/0725/make00-finished/img.npy')
