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


if __name__ == '__main__':
    bounding_box('../dataset/0725/make00-finished/img.npy')
