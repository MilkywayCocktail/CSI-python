import numpy as np
import pycsi
import csitest
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import csi_loader
import random
import rosbag


class MyDataset:
    def __init__(self, x_path, y_path):
        self.load_data(x_path, y_path)
        print('loaded')

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def load_data(self, x_path, y_path):
        x = np.load(x_path)
        y = np.load(y_path)

        if x.shape[0] == y.shape[0]:
            total_count = x.shape[0]
        else:
            print(x.shape, y.shape, "not equal!")

        print(total_count)
        train_choices = random.sample(list(range(total_count)), int(total_count * 0.8))
        test_choices = set(range(total_count)) - set(train_choices)
        valid_choices = random.sample(train_choices, int(len(train_choices) * 0.125))
        train_choices = set(train_choices) - set(valid_choices)

        train_x = np.array([x[i] for i in train_choices])
        train_y = np.array([y[i] for i in train_choices])
        valid_x = np.array([x[i] for i in valid_choices])
        valid_y = np.array([y[i] for i in valid_choices])
        test_x = np.array([x[i] for i in test_choices])
        test_y = np.array([y[i] for i in test_choices])

        print(
            "train_X.shape:{}\ntrain_Y.shape:{}\nvalid_X.shape:{}\nvalid_Y.shape:{}\n"
            .format(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape))

        print(train_choices)

mydata = MyDataset('../dataset/concat/1213/depth_3m/x.npy', '../dataset/concat/1213/depth_3m/y.npy')
