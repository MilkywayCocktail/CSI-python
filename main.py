# Draft by CAO
# Last edit: 2022-07-25
from CSIKit.reader import get_reader
from CSIKit.util import csitools
from CSIKit.tools.batch_graph import BatchGraph

from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel

import numpy as np
import time
import os


class MyException(Exception):
    def __init__(self, catch):
        self.catch = catch


class PathError(MyException):
    def __str__(self):
        return "Please check the path\n" + \
               "Current path is: " + str(self.catch) + \
               "\nPlease run .set_path() to specify the path"


class DataError(MyException):
    def __str__(self):
        return "Please run .load_data() to load data"


class ArgError(MyException):
    def __str__(self):
        return "Please check the argument"


class MyCsi(object):

    __credit = 'cao'

    def __init__(self, name, path=None):
        self.name = name
        self.path = path
        self.data = self._Data(name)

    def set_path(self, path):
        self.path = path
        print("path set")

    def load_data(self):
        try:
            if self.path is None or not os.path.exists(self.path):
                raise PathError(self.path)

            print(self.name, "load start - ", time.asctime(time.localtime(time.time())))
            csi_reader = get_reader(self.path)
            csi_data = csi_reader.read_file(self.path, scaled=True)
            csi_amp, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
            csi_phase, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="phase")

            print(self.name, "load complete - ", time.asctime(time.localtime(time.time())))
            self.data.amp = csi_amp
            self.data.phase = csi_phase
            self.data.timestamps = csi_data.timestamps

        except PathError as e:
            print(e)

    def show_path(self):
        try:
            if self.path is None:
                raise PathError(self.path)
            print(self.name, "path: ", self.path)

        except PathError as e:
            print(e)

    class _Data:
        def __init__(self, name):
            self.name = name
            self.amp = None
            self.phase = None
            self.timestamps = None

        def show_shape(self):
            try:
                if self.amp is None:
                    raise DataError(self.amp)

                items = ["no_frames=", "no_subcarriers=", "no_rx_ant=", "no_tx_ant="]
                plist = [a + str(b) for a, b in zip(items, self.amp.shape)]
                print(self.name, "data shape: ", *plist, sep='\n')

            except DataError as e:
                print(e)

        def vis_all_rx(self, metric="amplitude"):
            try:
                if metric == "amplitude":
                    csi_matrix = self.amp

                elif metric == "phase":
                    csi_matrix = self.phase

                else:
                    raise ArgError(metric)

                if csi_matrix is None:
                    raise DataError(self.amp)

                for rx in range(csi_matrix.shape[2]):
                    csi_matrix_squeezed = np.squeeze(csi_matrix[:, :, rx, 0])
                    BatchGraph.plot_heatmap(csi_matrix_squeezed, self.timestamps)

            except ArgError as e:
                print(e, '\n'+"Please specify \"amplitude\" or \"phase\"")

            except DataError as e:
                print(e)

    def aoa_by_music(self, theta_list):
        lightspeed = 299792458
        center_freq = 5.68e+09
        dist_antenna = 0.0264
        twopi = 2 * np.pi
        nrx = 3
        nsub = 30

        theta_list = np.array(theta_list.reshape(-1, 1))
        steering_matrix = np.exp(-1.j * twopi * dist_antenna * np.sin(theta_list))


if __name__ == '__main__':

    mypath = "data/csi0720Atake6.dat"

# CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

    today = MyCsi("a6", mypath)

    today.load_data()

    today.data.show_shape()
