import numpy as np
import cv2

chunk = np.load('../dataset/1213/make00_finished/y.npy')
out = np.zeros((len(chunk), 200, 200))
for i in range(len(chunk)):
    image = chunk[i]
    out[i] = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA )


np.save('../dataset/1213/make02_finished/x.npy', out)

print("finished")