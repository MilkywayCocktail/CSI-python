import csi_loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time


def foo():
    for i in range(100000):
        print('\r\r{}\n{}'.format(i, i+10), end='')
        time.sleep(1)


def plot_this():
    img = np.random.randint(0, 10000, 64).reshape((8, 8))
    plt.imshow(img)
    plt.colorbar()
    plt.show()

print(np.arange(2))

'latent_straight_loss': [],
'latent_distil_loss': [],
'student_image_loss': [],