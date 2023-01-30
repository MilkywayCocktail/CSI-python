import os
import cv2

impath = '../visualization/0124/aoa-doppler/A03/'

videopath = '../visualization/0124/aoa-doppler/'

if not os.path.exists(videopath):
    os.makedirs(videopath)

fps = 30

frames = sorted(os.listdir(impath), key=lambda x: eval(x[-8:-3]))

img = cv2.imread(os.path.join(impath, frames[0]))
imgsize = (img.shape[1], img.shape[0])

videoname = 'A03'
videopath = os.path.join(videopath, videoname + '.avi')

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

videowriter = cv2.VideoWriter(videopath, fourcc, fps, imgsize)

for frame in frames:
    f_path = os.path.join(impath, frame)
    image = cv2.imread(f_path)
    videowriter.write(image)
    print(frame + ' has been written!')

videowriter.release()
