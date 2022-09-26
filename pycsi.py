# Draft by CAO
# Last edit: 2022-09-26
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
    def __init__(self, input_catch):
        self.catch = str(input_catch)


class PathError(MyException):
    def __str__(self):
        return "error with path: " + self.catch


class DataError(MyException):
    def __str__(self):
        return "error with data: " + self.catch


class ArgError(MyException):
    def __str__(self):
        return "error with argument: " + self.catch


class MyCsi(object):
    __credit = 'cao'

    # Neither getter nor setter is defined in MyCsi class.
    # Please get and set manually when you need to.

    def __init__(self, input_name, path=None):
        self.name = str(input_name)
        self.path = path
        self.data = self._Data(input_name)
        self.commonfunc = self._CommonFunctions

        self.lightspeed = 299792458
        self.center_freq = 5.67e+09  # 5.67GHz
        self.dist_antenna = self.lightspeed / self.center_freq / 2.  # half-wavelength (2.64)
        self.mjtwopi = -1.j * 2 * np.pi
        self.torad = np.pi / 180
        self.delta_subfreq = 3.125e+05  # 312.5KHz (fixed)
        self.nrx = 3
        self.ntx = 1
        self.nsub = 30

    def load_data(self):
        """
        Loads csi data into current MyCsi instance.
        Supports .dat (raw) and .npz (csi_amp, csi_phase, csi_timestamps).

        :return: csi data
        """
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
            print(e, "\nPlease check the path")
        except DataError as e:
            print(e, "\nFile not supported. Please input .dat or .npz")

    def load_spectrum(self, input_path=None):
        """
        Loads .npz spectrum into current MyCsi instance.

        :param input_path: the path of spectrum, usually in npsave folder
        :return: spectrum
        """
        try:
            if input_path is None or not os.path.exists(input_path):
                raise PathError(input_path)

            if input_path[-3:] != "npz":
                raise DataError(input_path)

            print(self.name, "spectrum load start...", time.asctime(time.localtime(time.time())))
            csi_spectrum = np.load(input_path)
            self.data.spectrum = csi_spectrum['csi_spectrum']
            print(self.name, "spectrum load complete", time.asctime(time.localtime(time.time())))

        except PathError as e:
            print(e, "\nPlease check the path.")
        except DataError as e:
            print(e, "\nFile not supported. Please input .npz")

    def save_csi(self, save_name=None):
        """
        Saves csi data as npz. Strongly recommended for speeding up loading.

        :param save_name: filename, defalut = self.name
        :return: save_name + '-csis.npz'
        """
        try:
            if self.data.amp is None or self.data.phase is None:
                raise DataError("csi data")

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
        """
        Saves spectrum as npz.

        :param save_name: filename, default = self.name
        :return: save_name + '-spectrum.npz'
        """
        try:
            if self.data.spectrum is None:
                raise DataError("spectrum: " + str(self.data.spectrum))

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

        except DataError as e:
            print(e, "\nPlease compute spectrum")

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
                if self.amp is None or self.phase is None:
                    raise DataError("csi data")

                items = ["no_frames=", "no_subcarriers=", "no_rx_ant=", "no_tx_ant="]
                plist = [a + str(b) for a, b in zip(items, self.amp.shape)]
                print(self.name, "data shape: ", *plist, sep='\n')

            except DataError as e:
                print(e, "\nPlease load data")

        def remove_inf_values(self):
            """
            Removes -inf values in csi amplitude which hinders further calculation.
            Replaces packets with -inf values with neighboring ones.\n
            Embodied in spectrum calculating methods.

            :return: Processed amplitude
            """
            try:
                if self.amp is None:
                    raise DataError("amplitude: " + str(self.amp))

                print("Apply invalid value removal " + time.asctime(time.localtime(time.time())))
                print("Found", len(np.where(self.amp == float('-inf'))[0]), "-inf values")

                if len(np.where(self.amp == float('-inf'))[0]) != 0:

                    for i in range(self.length):
                        invalid_flag = np.where(self.amp[i] == float('-inf'))

                        if len(invalid_flag[0]) != 0:
                            j = i
                            temp_flag = invalid_flag

                            while len(temp_flag[0] != 0) and j < self.length - 1:
                                j += 1
                                temp_flag = np.where(self.amp[j] == float('-inf'))

                            if j == self.length - 1 and len(temp_flag[0] == 0):
                                j = i
                                while len(temp_flag[0] != 0) and j >= 0:
                                    j -= 1
                                    temp_flag = np.where(self.amp[j] == float('-inf'))

                            self.amp[i] = self.amp[j]
                            self.phase[i] = self.phase[j]

            except DataError as e:
                print(e, "\nPlease load data")

        def vis_all_rx(self, metric="amplitude"):
            """
            Plots csi amplitude OR phase for all antennas.

            :param metric: 'amplitude' or 'phase'
            :return: value-time plot
            """
            try:
                if metric == "amplitude":
                    csi_matrix = self.amp

                elif metric == "phase":
                    csi_matrix = self.phase

                else:
                    raise ArgError("metric: " + str(metric))

                if csi_matrix is None:
                    raise DataError("csi data")

                print(self.name, metric, "plotting...", time.asctime(time.localtime(time.time())))

                for rx in range(csi_matrix.shape[2]):
                    csi_matrix_squeezed = np.squeeze(csi_matrix[:, :, rx, 0])
                    BatchGraph.plot_heatmap(csi_matrix_squeezed, self.timestamps)

                print(self.name, metric, "plot complete", time.asctime(time.localtime(time.time())))

            except ArgError as e:
                print(e, "\nPlease specify metric=\"amplitude\" or \"phase\"")
            except DataError as e:
                print(e, "\nPlease load data")

        def vis_spectrum(self, threshold=0, autosave=False, notion=''):
            """
            Plots spectrum. You can select whether save the image or not.

            :param threshold: set threshold of spectrum. Default is 0 (none).
            :param autosave: 'True' or 'False'
            :param notion: string, save additional information in filename if autosave
            :return: spectrum plot
            """
            try:
                if self.spectrum is None:
                    raise DataError("spectrum: " + str(self.spectrum))

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
                plt.title(self.name + " AoA Spectrum" + str(notion))

                print(self.name, "plot complete", time.asctime(time.localtime(time.time())))

                if autosave is False:
                    plt.show()

                elif autosave is True:
                    save_path = os.getcwd().replace('\\', '/') + "/visualization/" + self.name[:4] + '/'

                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

                    savename = save_path + self.name[4:] + '_AoA' + str(notion) + '.png'
                    plt.savefig(savename)
                    print(self.name, "saved as", savename, time.asctime(time.localtime(time.time())))
                    plt.close()

                else:
                    raise ArgError("autosave")

            except DataError as e:
                print(e, "\nPlease compute spectrum")
            except ArgError as e:
                print(e, "\nPlease specify autosave=\"True\" or \"False\"")

    class _CommonFunctions:
        def smooth_csi(self, input_csi, rx=2, sub=15):
            """
            Static method.\n
            Applies SpotFi smoothing technique. You have to run for amplitude and phase each.

            :param input_csi:  [packet, sub, rx]
            :param rx: the number of receive antennas for smoothing (default: 2 proposed in spotfi)
            :param sub: the number of subcarriers for smoothing (default: 15 proposed in spotfi)
            :return: smoothed csi
            """
            nrx = input_csi.shape[1]
            nsub = input_csi.shape[0]

            input_csi = input_csi.swapaxes(0, 1)

            output = [input_csi[i:i + rx, j:j + sub].reshape(-1)
                      for i in range(nrx - rx + 1)
                      for j in range(nsub - sub + 1)]

            return np.array(output)

    def aoa_by_music(self, input_theta_list=np.arange(-90, 91, 1.), smooth=False):
        """
        Computes AoA spectrum by MUSIC.

        :param input_theta_list: list of angels, default = -90~90
        :param smooth: whether apply SpotFi smoothing or not, default = False
        :return: AoA spectrum by MUSIC stored in self.data.spectrum
        """
        lightspeed = self.lightspeed
        center_freq = self.center_freq
        dist_antenna = self.dist_antenna
        mjtwopi = self.mjtwopi
        torad = self.torad
        delta_subfreq = self.delta_subfreq
        nrx = self.nrx
        ntx = self.ntx

        try:
            if self.data.amp is None:
                raise DataError("amplitude: " + str(self.data.amp))

            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase))

            # Subcarriers from -58 to 58, step = 4
            subfreq_list = np.arange(center_freq - 58 * delta_subfreq, center_freq + 62 * delta_subfreq,
                                     4 * delta_subfreq)
            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)

            spectrum = np.zeros((len(input_theta_list), self.data.length))

            print(self.name, "AoA by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

            if smooth is True:
                print("Apply Smoothing via SpotFi...")

            # Replace -inf values with neighboring packets before computing

            self.data.remove_inf_values()

            temp_amp = 0
            temp_phase = 0

            for i in range(self.data.length):

                if smooth is True:
                    temp_amp = self.commonfunc.smooth_csi(np.squeeze(self.data.amp[i]))
                    temp_phase = self.commonfunc.smooth_csi(np.squeeze(self.data.phase[i]))

                else:
                    temp_amp = self.data.amp[i]
                    temp_phase = self.data.phase[i]

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

        except DataError as e:
            print(e, "\nPlease load data")

    def doppler_by_music(self, input_velocity_list=np.arange(-5, 5.01, 0.01)):
        """
        Computes Doppler spectrum by MUSIC. Under construction.

        :param: input_velocity_list: list of velocities, default = -5~5
        :return: Doppler spectrum by MUSIC stored in self.data.spectrum
        """
        lightspeed = self.lightspeed
        center_freq = self.center_freq
        mjtwopi = self.mjtwopi
        delta_subfreq = self.delta_subfreq
        nrx = self.nrx
        ntx = self.ntx
        nsub = self.nsub
        num_samples = 100
        delta_t = 0.2208e-3

        try:
            if self.data.amp is None:
                raise DataError("amplitude: " + str(self.data.amp))

            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase))

            # Subcarriers from -58 to 58, step = 4
            subfreq_list = np.arange(center_freq - 58 * delta_subfreq, center_freq + 62 * delta_subfreq,
                                     4 * delta_subfreq)
            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)

            spectrum = np.zeros((len(input_velocity_list), self.data.length))

            print(self.name, "Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

            # Replace -inf values with neighboring packets before computing

            self.data.remove_inf_values()

            temp_amp = 0
            temp_phase = 0

            for i in range(self.data.length - num_samples):

                csi = [np.squeeze(self.data.amp[i + i_sub, :, 0]) *
                       np.exp(1.j * np.squeeze(self.data.phase[i + i_sub, :, 0]))
                       for i_sub in range(num_samples)]

                csi = np.array(csi)
                value, vector = np.linalg.eigh(csi.dot(np.conjugate(csi.T)))
                descend_order_index = np.argsort(-value)
                vector = vector[:, descend_order_index]
                noise_space = vector[:, ntx:]

                # print(value[descend_order_index])

                for j, velocity in enumerate(input_velocity_list):

                    steering_vector = np.exp([mjtwopi * center_freq * delta_t / lightspeed *
                                              m for m in range(num_samples)])

                    a_en = np.conjugate(steering_vector.T).dot(noise_space)
                    spectrum[j, i] = 1. / np.absolute(a_en.dot(np.conjugate(a_en.T)))

            print(self.name, "Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))
            self.data.spectrum = spectrum

        except DataError as e:
            print(e, "\nPlease load data")

    def sanitize_phase(self):
        pass

    def extract_dynamic(self, mode='overall'):
        """
        Removes the static component from csi.\n
        Strongly recommended when Tx is placed beside Rx.

        :param mode: 'overall' or 'running' (in terms of averaging)
        :return: phase and amplitude of dynamic component of csi
        """
        nrx = self.nrx
        ntx = self.ntx
        nsub = self.nsub

        try:
            if self.data.amp is None or self.data.phase is None:
                raise DataError("csi data")

            complex_csi = self.data.amp * np.exp(1.j * self.data.phase)
            conjugate_csi = complex_csi[:, :, 0, :, None].repeat(3, axis=2)
            hc = complex_csi * conjugate_csi

            if mode == 'overall':
                average_hc = np.mean(hc, axis=0).reshape((1, nsub, nrx, ntx))

            elif mode == 'running':
                average_hc = np.array([[np.convolve(np.squeeze(hc[:, sub, antenna, :]),
                                        np.ones(101) / 101, mode='same')
                                        for sub in range(30)]
                                      for antenna in range(3)]).swapaxes(0, 2).reshape((1, nsub, nrx, ntx))
            else:
                raise ArgError("mode: " + str(mode))

            dynamic_csi = hc - average_hc.repeat(self.data.length, axis=0)
            self.data.amp = np.abs(dynamic_csi)
            self.data.phase = np.angle(dynamic_csi)

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify mode=\"running\" or \"overall\"")

    def calibrate_phase(self, input_mycsi):
        """
        Calibrates phase offset between other degrees against 0 degree.\n
        Initial Phase Offset is removed.

        :param input_mycsi: CSI recorded at 0 degree
        :return: calibrated phase, unwrapped
        """
        nrx = self.nrx
        nsub = self.nsub

        try:
            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase))

            if not isinstance(input_mycsi, MyCsi):
                raise DataError("reference csi: " + str(input_mycsi) + "\nPlease input MyCsi instance.")

            if input_mycsi.data.phase is None:
                raise DataError("reference phase: " + str(input_mycsi.data.phase))

            print("Apply phase calibration according to " + input_mycsi.name, time.asctime(time.localtime(time.time())))

            reference_csi = np.squeeze(input_mycsi.data.amp) * np.exp(1.j * np.squeeze(input_mycsi.data.phase))
            current_csi = np.squeeze(self.data.amp) * np.exp(1.j * np.squeeze(self.data.phase))

            subtrahend = np.expand_dims(reference_csi[:, :, 0], axis=2).repeat(3, axis=2)

            relative = reference_csi * np.conjugate(subtrahend)
            offset = np.mean(relative, axis=(0, 1)).reshape((1, 1, nrx))

            offset = offset.repeat(nsub, axis=1).repeat(self.data.length, axis=0)

            calibrated_csi = current_csi * np.conjugate(offset)
            calibrated_csi = np.expand_dims(calibrated_csi, axis=3)
            print(calibrated_csi.shape)

            self.data.amp = np.abs(calibrated_csi)
            self.data.phase = np.angle(calibrated_csi)

        except DataError as e:
            print(e, "\nPlease load data")


if __name__ == '__main__':

    # CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

    filepath = "data/0919/"
    filenames = os.listdir(filepath)
    for file in filenames:
        name = file[3:-4]
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

