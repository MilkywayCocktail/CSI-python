import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
import torch.utils.data as Data
import h5py
from Trainer_v03b2 import *


# ------------------------------------- #
# Model v03b3
# VAE version; Adaptive to 3D Shapes dataset.
# Added interpolating decoder
# Adjusted for MNIST

# ImageEncoder: in = 128 * 128, out = 1 * 256
# ImageDecoder: in = 1 * 256, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256 (Unused)

class Shapes3D(Data.Dataset):
    def __init__(self, datadir, transform=None, number=0):
        self.seeds = None
        self.transform = transform
        self.data = self.load_data(datadir, number)

    def __getitem__(self, index):
        if self.transform:
            image = self.transform(self.data['x'][index])
        else:
            image = self.data['x'][index]

        return image, self.data['y'][index]

    def __len__(self):
        return self.data['labels'].shape[0]

    def load_data(self, datadir, number):
        data = h5py.File(datadir, 'r')
        x = data['images']
        y = data['labels']

        if number != 0:
            picked = np.random.choice(list(range(len(x))), size=number, replace=False)
            self.seeds = picked
            x = x[picked]
            y = y[picked]

        return {'x': x, 'y': y}
