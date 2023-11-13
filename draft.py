import numpy as np
import cv2
import os
import random


class DatasetHandle:
    def __init__(self, in_path, out_path, scope: tuple, types: tuple):

        self.in_path = in_path
        self.out_path = out_path
        self.scope = scope
        self.types = types
        self.groups = {}
        self.results = self.__gen_results__()

    def __gen_results__(self):
        result = {}
        for typ in self.types:
            for name in self.scope:
                result[typ] = {name: []}
        return result

    def load(self):
        filenames = os.listdir(self.in_path)
        for file in filenames:
            name = file[:2]
            tmp_type = file[-7:-4]
            if name in self.scope and tmp_type in self.types:
                tmp = np.load(self.in_path + file)
                print(f"Loaded {file} of shape {tmp.shape}")
                self.results[tmp_type][name].append(tmp)

        print("All loaded!")

    def normalize_images(self, threshold=3000):
        if 'img' in self.types:
            print("Normalizing images...", end='')
            for name in self.scope:
                for i, img in enumerate(self.results['img'][name]):
                    img = img.reshape((-1, 1, 128, 128))
                    img[img > threshold] = threshold
                    img = img / float(threshold)
                    self.results['img'][name][i] = img
            print("Done!")

    def filter_directions(self, condition):
        if 'loc' in self.types:
            print(f"Filtering dataset s.t. {condition}...", end='')
            self.groups[condition] = []
            for i, loc in enumerate(self.results['loc']):
                x, y = loc[:, 0], loc[:, 1]
                self.groups[condition].append(np.squeeze(np.argwhere(eval(condition))))
            print("Filtering complete!")

    def inspect_conditions(self):
        for typ in self.types:
            for name in self.scope:
                print(f"{typ}: {name} length {len(self.results[typ][name])}")
                if self.groups.items() is not None:
                    for condition in self.groups.keys():
                        print(f"{typ}: {name} length {len(self.results[typ][name][self.groups[condition]])} "
                              f"under condition {condition}")

    def save(self, save_scope=None, condition=None):

        for typ in self.types:
            tmp = []
            scope = save_scope if save_scope else self.scope
            for name in scope:
                if condition:
                    tmp.append(self.results[typ][name][self.groups[condition]])
                else:
                    tmp.append(self.results[typ][name])
            print(f"Saving {condition}{typ} of length {len(tmp)}...")
            np.save(f"{self.out_path}{condition}{typ}.npy", np.concatenate((self.results[typ].values())))

        print("All saved!")
