# Draft by CAO
# Last edit: 2022-07-20
from CSIKit.reader import get_reader
from CSIKit.util import csitools
from CSIKit.tools.batch_graph import BatchGraph

from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel

import numpy as np
import time

class mycsi(object):

    credit = 'cao'

    def __init__(self, name, path=None):
        self.name = name
        self.path = path
        self.data = self.data()


    def set_path(self, path):
        self.path = path


    def load_data(self):
        if self.path==None:
            print("Please run .set_path() to specify a path")

        else:
            try:
                print("load start - ", time.asctime(time.localtime(time.time())))
                csi_reader = get_reader(self.path)
                csi_data = csi_reader.read_file(self.path, scaled=True)
                csi_amp, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
                csi_phase, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="phase")

                print("load complete - ", time.asctime(time.localtime(time.time())))
                self.data.amp = csi_amp
                self.data.phase = csi_phase
                self.data.timestamps = csi_data.timestamps

            except:
                print("Please check the path")


    def show_path(self):
        if self.path==None:
            print("Please run .set_path() to specify a path")

        else:
            print("path: ", self.path)


    class data:
        def __init__(self):
            self.amp = None
            self.phase = None
            self.timestamps = None


        def show_shape(self):
            try:
                items = ["no_frames=", "no_subcarriers=", "no_rx_ant=", "no_tx_ant="]
                plist = [a + str(b) for a, b in zip(items, self.data.amp.shape)]
                print("data shape: ", plist)

            except:
                print("Please run .load_data()")


        def vis_all_rx(self, metric="amplitude"):
            if metric=="amplitude":
                csi_matrix = self.amp

            elif metric=="phase":
                csi_matrix = self.phase

            try:
                for rx in range(csi_matrix.shape[2]):
                    csi_matrix_squeezed = np.squeeze(csi_matrix[:, :, rx, 0])
                    BatchGraph.plot_heatmap(csi_matrix_squeezed, self.timestamps)

            except:
                print("Please run .load_data()")


    def aoa_by_music(self):
        lightspeed = 299792458
        center_freq = 5.68e+09
        dist_antenna = 0.0264

if __name__ == '__main__':

    mypath = "data/csi0720Atake6.dat"

# CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

    today = mycsi("a6", mypath)

    today.load_data()

    print(len(today.data.timestamps))
