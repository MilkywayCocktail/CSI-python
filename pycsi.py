# Draft by CAO
# Last edit: 2022-09-15
from CSIKit.reader import get_reader
from CSIKit.util import csitools
from CSIKit.tools.batch_graph import BatchGraph

from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


class MyException(Exception):
    def __init__(self, catch):
        self.catch = catch


class PathError(MyException):
    def __str__(self):
        return "error with " + str(self.catch) + "\nPlease check the path"


class DataError(MyException):
    def __str__(self):
        return "No data"


class ArgError(MyException):
    def __str__(self):
        return "error with " + str(self.catch) + "\n Please check the argument"


class MyCsi(object):
    __credit = 'cao'

    def __init__(self, input_name, path=None):
        self.name = str(input_name)
        self.path = path
        self.data = self._Data(input_name)

    def set_path(self, path):
        self.path = path
        print(self.name, "path set")

    def show_path(self):
        try:
            if self.path is None:
                raise PathError(self.path)
            print(self.name, "path: ", self.path)

        except PathError as e:
            print(e)

    def load_data(self):
        try:
            if self.path is None or not os.path.exists(self.path):
                raise PathError(self.path)

            if self.path[-3:] == "dat":
                print(self.name, "raw load start...", time.asctime(time.localtime(time.time())))
                csi_reader = get_reader(self.path)
                csi_data = csi_reader.read_file(self.path, scaled=True)
                csi_amp, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
                csi_phase, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="phase")

                self.data.amp = csi_amp
                self.data.phase = csi_phase
                self.data.timestamps = csi_data.timestamps
                self.data.length = no_frames
                print(self.name, "raw load complete", time.asctime(time.localtime(time.time())))

            elif self.path[-3:] == "npz":
                print(self.name, "npz load start...", time.asctime(time.localtime(time.time())))
                csi_data = np.load(self.path)
                self.data.amp = csi_data['csi_amp']
                self.data.phase = csi_data['csi_phase']
                self.data.length = len(csi_data['csi_timestamps'])
                self.data.timestamps = csi_data['csi_timestamps']
                print(self.name, "npz load complete", time.asctime(time.localtime(time.time())))

        except PathError as e:
            print(e)

    def load_spectrum(self, path):
        try:
            if path is None or not os.path.exists(path):
                raise PathError(path)

            print(self.name, "spectrum load start...", time.asctime(time.localtime(time.time())))
            csi_spectrum = np.load(path)
            self.data.spectrum = csi_spectrum['csi_spectrum']
            print(self.name, "spectrum load complete", time.asctime(time.localtime(time.time())))

        except PathError as e:
            print(e)

    class _Data:
        def __init__(self, input_name):
            self.name = input_name
            self.amp = None
            self.phase = None
            self.timestamps = None
            self.length = None
            self.spectrum = None

        def show_shape(self):
            try:
                if self.amp is None:
                    raise DataError(self.amp)

                items = ["no_frames=", "no_subcarriers=", "no_rx_ant=", "no_tx_ant="]
                plist = [a + str(b) for a, b in zip(items, self.amp.shape)]
                print(self.name, "data shape: ", *plist, sep='\n')

            except DataError as e:
                print(e, "Please run .load_data() or .load_npz()")

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

                print(self.name, metric, "plotting...", time.asctime(time.localtime(time.time())))

                for rx in range(csi_matrix.shape[2]):
                    csi_matrix_squeezed = np.squeeze(csi_matrix[:, :, rx, 0])
                    BatchGraph.plot_heatmap(csi_matrix_squeezed, self.timestamps)

                print(self.name, metric, "plot complete", time.asctime(time.localtime(time.time())))

            except ArgError as e:
                print(e, '\n' + "Please specify \"amplitude\" or \"phase\"")

            except DataError as e:
                print(e, "Please run .load_data() or .load_npz()")

        def vis_spectrum(self, threshold=10):
            try:
                if self.spectrum is None:
                    raise DataError(self.spectrum)

                print(self.name, "plotting...", time.asctime(time.localtime(time.time())))

                spectrum = np.array(self.spectrum)

                if threshold != 0:
                    spectrum[spectrum > threshold] = threshold

                ax = sns.heatmap(spectrum)

                if self.spectrum.shape[0] == 360:
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(60))
                    ax.yaxis.set_major_formatter(ticker.FixedFormatter([-240, -180, -120, -60, 0, 60, 120, 180]))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(20))

                elif self.spectrum.shape[0] == 181:
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
                    ax.yaxis.set_major_formatter(ticker.FixedFormatter([-120, -90, -60, -30, 0, 30, 60, 90]))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))

                ax.set_xlabel("#timestamp")
                ax.set_ylabel("Angel / $deg$")
                ax.collections[0].colorbar.set_label('Power / $dB$')
                plt.title(self.name + " AoA Spectrum")

                print(self.name, "plot complete", time.asctime(time.localtime(time.time())))
                plt.show()

            except DataError as e:
                print(e, "\nPlease compute spectrum")

    def save_csi(self, save_name=None):
        try:
            if self.data.amp is None:
                raise DataError(self.data.amp)

            save_path = os.getcwd().replace('\\', '/') + "/npsave"

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if save_name is None:
                save_name = self.name

            # Keys: amp, phase, timestamps
            print(self.name, "csi save start...", time.asctime(time.localtime(time.time())))
            np.savez(save_path + "/" + save_name + "-csis.npz",
                     csi_amp=self.data.amp,
                     csi_phase=self.data.phase,
                     csi_timestamps=self.data.timestamps)
            print(self.name, "csi save complete", time.asctime(time.localtime(time.time())))
        except DataError as e:
            print(e, "to save")

    def save_spectrum(self, save_name=None):
        save_path = os.getcwd().replace('\\', '/') + "/npsave"

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if save_name is None:
            save_name = self.name

        # Keys: spectrum, info
        print(self.name, "spectrum save start...", time.asctime(time.localtime(time.time())))
        np.savez(save_path + "/" + save_name + "-spectrum.npz",
                 csi_spectrum=self.data.spectrum)
        print(self.name, "spectrum save complete", time.asctime(time.localtime(time.time())))

    def aoa_by_music(self, input_theta_list, smooth=False):
        lightspeed = 299792458
        center_freq = 5.67e+09  # 5.67GHz
        dist_antenna = lightspeed / center_freq / 2.  # 2.64
        mjtwopi = -1.j * 2 * np.pi
        torad = np.pi / 180
        delta_subfreq = 3.125e+05  # 312.5KHz (fixed)
        nrx = 3
        ntx = 1

        def smooth_csi(input_csi, rx=2, sub=15):
            """
            :param input_csi:  [packet, sub, rx]
            :param rx: the number of receive antennas for smoothing (default: 2 proposed in spotfi)
            :param sub: the number of subcarriers for smoothing (default: 15 proposed in spotfi)
            :return: smoothed csi
            You have to run for amplitude and phase each
            """
            nrx = input_csi.shape[1]
            nsub = input_csi.shape[0]

            input_csi = input_csi.swapaxes(0, 1)

            output = [input_csi[i:i + rx, j:j + sub].reshape(-1)
                      for i in range(nrx - rx + 1)
                      for j in range(nsub - sub + 1)]

            return np.array(output)

        # Subcarriers from -58 to 58, step = 4
        subfreq_list = np.arange(center_freq - 58 * delta_subfreq, center_freq + 62 * delta_subfreq,
                                 4 * delta_subfreq)
        antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)

        spectrum = np.zeros((len(input_theta_list), self.data.length))

        print(self.name, "AoA by MUSIC - compute start...", time.asctime(time.localtime(time.time())))
        if smooth is True:
            print("Apply Smoothing via SpotFi...")

        temp_amp = 0
        temp_phase = 0

        for i in range(self.data.length):

            invalid_flag = np.where(self.data.amp[i] == float('-inf'))

            if len(invalid_flag[0]) == 0:
                temp_amp = self.data.amp[i]
                temp_phase = self.data.phase[i]

            if len(invalid_flag[0]) != 0 and i == 0:

                j = i
                temp_flag = invalid_flag

                while len(temp_flag[0] != 0) and j < self.data.length:
                    j += 1
                    temp_flag = np.where(self.data.amp[j] == float('-inf'))

                temp_amp = self.data.amp[j]
                temp_phase = self.data.phase[j]

            if smooth is True:
                temp_amp = smooth_csi(np.squeeze(temp_amp))
                temp_phase = smooth_csi(np.squeeze(temp_phase))

            csi = np.squeeze(temp_amp) * np.exp(1.j * np.squeeze(temp_phase))

            value, vector = np.linalg.eigh(csi.T.dot(np.conjugate(csi)))
            descend_order_index = np.argsort(-value)
            vector = vector[:, descend_order_index]
            noise_space = vector[:, ntx:]

            #print(value[descend_order_index])

            for j, theta in enumerate(input_theta_list):
                if smooth is True:
                    steering_vector = np.exp([mjtwopi * dist_antenna * np.sin(theta * torad) *
                                             no_antenna * sub_freq / lightspeed
                                             for no_antenna in antenna_list[:2]
                                             for sub_freq in subfreq_list[:15]])
                else:
                    steering_vector = np.exp(mjtwopi * dist_antenna * np.sin(theta * torad) *
                                             antenna_list * center_freq / lightspeed)

                a_en = np.conjugate(steering_vector.T).dot(noise_space)
                spectrum[j, i] = 1. / np.absolute(a_en.dot(np.conjugate(a_en.T)))

        print(self.name, "AoA by MUSIC - compute complete", time.asctime(time.localtime(time.time())))
        self.data.spectrum = spectrum
        print(spectrum.shape)

    def sanitize_phase(self):
        pass

    def calibrate_aoa(self, input_mycsi):
        """
        :param input_mycsi: CSI recorded at 0-deg
        :return: calibrated phase regarding 0-deg
        """
        standard_phase = input_mycsi.data.phase
        relative_phase = standard_phase - standard_phase[:, :, 0].repeat(3, axis=2).reshape(np.shape(standard_phase))
        offset = np.angle(np.mean(np.exp(-1.j * relative_phase), axis=0))
        self.data.phase = self.data.phase - offset


if __name__ == '__main__':

    name = "0812C01"

    mypath = "data/csi" + name + ".dat"
    npzpath = "npsave/" + name + "-csis.npz"
    pmpath = "npsave/" + name + "-spectrum.npz"

    theta_list = np.arange(-90, 91, 1.)

    # CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

    today = MyCsi(name, npzpath)

    today.load_data()

#    today.data.show_shape()

#    today.save_csi(name)

    today.aoa_by_music(theta_list, smooth=False)

#    today.save_spectrum(name)

#    today.load_spectrum(pmpath)

#    print(today.data.spectrum.shape)

    today.data.vis_spectrum(2)


