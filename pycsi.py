# Draft by CAO
# Last edit: 2022-09-20
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
        return "No data of " + str(self.catch)


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

            else:
                raise DataError(self.path)

        except PathError as e:
            print(e)
        except DataError as e:
            print(e, "\nFile not supported. Please input .dat or .npz")

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
                    raise DataError("amplitude")
                if self.phase is None:
                    raise DataError("phase")

                items = ["no_frames=", "no_subcarriers=", "no_rx_ant=", "no_tx_ant="]
                plist = [a + str(b) for a, b in zip(items, self.amp.shape)]
                print(self.name, "data shape: ", *plist, sep='\n')

            except DataError as e:
                print(e, "\nPlease run .load_data() or .load_npz()")

        def vis_all_rx(self, metric="amplitude"):
            try:
                if metric == "amplitude":
                    csi_matrix = self.amp

                elif metric == "phase":
                    csi_matrix = self.phase

                else:
                    raise ArgError("metric")

                if csi_matrix is None:
                    raise DataError("amplitude")

                print(self.name, metric, "plotting...", time.asctime(time.localtime(time.time())))

                for rx in range(csi_matrix.shape[2]):
                    csi_matrix_squeezed = np.squeeze(csi_matrix[:, :, rx, 0])
                    BatchGraph.plot_heatmap(csi_matrix_squeezed, self.timestamps)

                print(self.name, metric, "plot complete", time.asctime(time.localtime(time.time())))

            except ArgError as e:
                print(e, "\nPlease specify metric=\"amplitude\" or \"phase\"")

            except DataError as e:
                print(e, "\nPlease run .load_data() or .load_npz()")

        def vis_spectrum(self, threshold=0, autosave=False, notion=None):
            """
            Plots spectrum. You can select whether save or not.
            
            :param threshold: set threshold of spectrum. Default is 0 (none).
            :param autosave: 'True' or 'False'
            :param notion: save additional information in filename if autosave
            :return: spectrum plot
            """
            try:
                if self.spectrum is None:
                    raise DataError("spectrum")
                if autosave is not True and autosave is not False:
                    raise ArgError("autosave")

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

                if autosave is False:
                    plt.show()
                elif autosave is True:
                    plt.savefig('visualization/' + self.name[:4] + '/' + self.name[4:] + notion + '.png')

            except DataError as e:
                print(e, "\nPlease compute spectrum")
            except ArgError as e:
                print(e, "\nPlease specify autosave=\"True\" or \"False\"")

    def save_csi(self, save_name=None):
        try:
            if self.data.amp is None:
                raise DataError("amplitude")

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
            print(e, "\nPlease load data")

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

    def aoa_by_music(self, input_theta_list=np.arange(-90, 91, 1.), smooth=False):
        """
        Computes AoA spectrum by MUSIC.
        
        :param input_theta_list: list of angels, default = -90~90
        :param smooth: whether apply SpotFi smoothing or not, default = False
        :return: AoA spectrum by MUSIC stored in self.data.spectrum
        """
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
            Applies SpotFi smoothing technique.
            
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

        try:
            if self.data.amp is None:
                raise DataError("amplitude")
            if self.data.phase is None:
                raise DataError("phase")
            else:
                # Subcarriers from -58 to 58, step = 4
                subfreq_list = np.arange(center_freq - 58 * delta_subfreq, center_freq + 62 * delta_subfreq,
                                         4 * delta_subfreq)
                antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)

                spectrum = np.zeros((len(input_theta_list), self.data.length))

                print(self.name, "AoA by MUSIC - compute start...", time.asctime(time.localtime(time.time())))
                if smooth is True:
                    print("Apply Smoothing via SpotFi...")

                # Replace -inf values with neighboring packets before computing

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

                    # print(value[descend_order_index])

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
        except DataError as e:
            print(e, "\nPlease load data")

    def sanitize_phase(self):
        pass

    def remove_phase_offset(self, mode='overall'):
        """
        Removes phase offset among Rx antennas. Antenna 0 as default standard.
        Using running mean.

        :param mode: 'overall' or 'running'
        :return: calibrated phase
        """
        try:
            if self.data.phase is None:
                raise DataError("phase")
            if mode != 'running' and mode != 'overall':
                raise ArgError("mode")

            else:
                print("Apply phase offset removing among antennas", time.asctime(time.localtime(time.time())))
                subtraction = np.expand_dims(self.data.phase[:, :, 0, :], axis=2).repeat(3, axis=2)
                relative_phase = self.data.phase - subtraction
                if mode == 'running':
                    offset = np.array([[np.convolve(np.squeeze(relative_phase[:, sub, antenna, :]),
                                                    np.ones(101) / 101, mode='same')
                                      for sub in range(30)]
                                      for antenna in range(3)]).swapaxes(0, 2).reshape(relative_phase.shape)
                elif mode == 'overall':
                    offset = np.mean(relative_phase, axis=0)
                self.data.phase -= offset
        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify mode=\"running\" or \"overall\"")

    def calibrate_phase(self, input_mycsi):
        """
        Calibrates phase offset between other degrees against 0 degree.

        :param input_mycsi: CSI recorded at 0 degree
        :return: calibrated phase, unwrapped
        """
        try:
            if self.data.phase is None:
                raise DataError(self.data.phase)
            if input_mycsi.data.phase is None:
                raise DataError(input_mycsi.data.phase)
            else:
                print("Apply phase calibration against " + input_mycsi.name, time.asctime(time.localtime(time.time())))
                standard_phase = input_mycsi.data.phase
                offset = np.mean(standard_phase, axis=0)
                print(offset)
                self.data.phase -= offset

        except DataError as e:
            print(e, "\nPlease load data")


if __name__ == '__main__':

    # CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

    filepath = "data/0919/"
    filenames = os.listdir(filepath)
    for file in filenames:
        name = file[:-4]
        mypath = filepath + file
        # npzpath = "npsave/csi" + name + "-csis.npz"
        # pmpath = "npsave/" + name + "-spectrum.npz"
        today = MyCsi(name, mypath)
        today.load_data()
        today.save_csi(name)

    #    today.data.show_shape()

    #    today.save_csi(name)

    # today.aoa_by_music(smooth=False)

    #    today.save_spectrum(name)

    #    today.load_spectrum(pmpath)

    #    print(today.data.spectrum.shape)

    # today.data.vis_spectrum(0)

