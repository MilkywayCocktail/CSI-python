from CSIKit.reader import get_reader
from CSIKit.util import csitools
from CSIKit.tools.batch_graph import BatchGraph

from CSIKit.filters.passband import highpass
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
    """
    Main functionalities of csi processing.
    """
    __credit = 'cao'

    # Neither getter nor setter is defined in MyCsi class.
    # Please get and set manually when you need to.

    def __init__(self, input_name='', path=None):
        self.name = str(input_name)
        self.path = path

        self.lightspeed = 299792458
        self.center_freq = 5.67e+09  # 5.67GHz
        self.dist_antenna = self.lightspeed / self.center_freq / 2.  # half-wavelength (2.64)
        self.mjtwopi = -1.j * 2 * np.pi
        self.torad = np.pi / 180
        self.delta_subfreq = 3.125e+05  # 312.5KHz (fixed)
        self.nrx = 3
        self.ntx = 1
        self.nsub = 30

        self.commonfunc = self._CommonFunctions
        self.data = self._Data(input_name, self.nrx)

    def __str__(self):
        return 'MyCsi-' + self.name

    def __repr__(self):
        return 'MyCsi-' + self.name

    def load_data(self):
        """
        Loads csi data into current MyCsi instance.\n
        Supports .dat (raw) and .npz (csi_amp, csi_phase, csi_timestamps).\n
        :return: csi data
        """
        try:
            if self.path is None or not os.path.exists(self.path):
                raise PathError(str(self.path))
            if self.path[-3:] not in ('dat', 'npz'):
                raise DataError("file: " + str(self.path))

        except PathError as e:
            print(e, "\nPlease check the path")
        except DataError as e:
            print(e, "\nFile not supported. Please input .dat or .npz")

        else:
            if self.path[-3:] == "dat":
                print(self.name, "raw load start...", time.asctime(time.localtime(time.time())))
                csi_reader = get_reader(self.path)
                csi_data = csi_reader.read_file(self.path, scaled=True)
                csi_amp, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
                csi_phase, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="phase")

                self.data.amp = csi_amp
                self.data.phase = csi_phase
                self.data.length = no_frames
                self.data.timestamps = csi_data.timestamps
                self.data.sampling_rate = self.data.length / self.data.timestamps[-1]
                print(self.name, "raw load complete", time.asctime(time.localtime(time.time())))

            elif self.path[-3:] == "npz":
                print(self.name, "npz load start...", time.asctime(time.localtime(time.time())))
                csi_data = np.load(self.path)
                self.data.amp = csi_data['csi_amp']
                self.data.phase = csi_data['csi_phase']
                self.data.length = len(csi_data['csi_timestamps'])
                self.data.timestamps = csi_data['csi_timestamps']
                self.data.sampling_rate = self.data.length / self.data.timestamps[-1]
                print(self.name, "npz load complete", time.asctime(time.localtime(time.time())))

    def load_lists(self, amp, phase, timelist):
        """
        Loads separate items into curren MyCsi instance.\n
        :param amp: input amplitude
        :param phase: input phase
        :param timelist: input timestamps
        :return: amplitude, phase, timestamps, length, sampling_rate
        """

        self.data.amp = amp
        self.data.phase = phase
        self.data.timestamps = timelist
        self.data.length = len(self.data.amp)
        self.data.sampling_rate = self.data.length / self.data.timestamps[-1]

    def load_spectrum(self, input_path=None):
        """
        Loads .npz spectrum into current MyCsi instance.\n
        :param input_path: the path of spectrum, usually in 'npsave' folder\n
        :return: spectrum
        """
        print(self.name, "spectrum load start...", time.asctime(time.localtime(time.time())))

        try:
            if input_path is None or not os.path.exists(input_path):
                raise PathError(str(input_path))

            if input_path[-3:] != "npz":
                raise DataError("file: " + str(input_path))

        except PathError as e:
            print(e, "\nPlease check the path.")
        except DataError as e:
            print(e, "\nFile not supported. Please input .npz")

        else:
            csi_spectrum = np.load(input_path)
            self.data.spectrum = csi_spectrum['csi_spectrum']
            self.data.algorithm = csi_spectrum['csi_algorithm']
            print(self.name, "spectrum load complete", time.asctime(time.localtime(time.time())))

    def save_csi(self, save_name=None):
        """
        Saves csi data as npz. Strongly recommended for speeding up loading.\n
        :param save_name: filename, defalut = self.name
        :return: save_name + '-csis.npz'
        """
        print(self.name, "csi save start...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.amp is None or self.data.phase is None:
                raise DataError("csi data")

        except DataError as e:
            print(e, "\nPlease load data")

        else:
            save_path = "../npsave/" + self.name[:4] + '/csi/'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if save_name is None:
                save_name = self.name

            # Keys: amp, phase, timestamps
            np.savez(save_path + save_name + "-csis.npz",
                     csi_amp=self.data.amp,
                     csi_phase=self.data.phase,
                     csi_timestamps=self.data.timestamps)
            print(self.name, "csi save complete", time.asctime(time.localtime(time.time())))

    def save_spectrum(self, notion=''):
        """
        Saves spectrum as npz.\n
        :param notion: additional information in the savename, default is empty
        :return: save_name + '-spectrum.npz'
        """
        print(self.name, "spectrum save start...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.spectrum is None:
                raise DataError("spectrum")

        except DataError as e:
            print(e, "\nPlease compute spectrum")

        else:
            save_path = "../npsave/" + self.name[:4] + '/spectrum/'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Keys: spectrum, algorithm
            np.savez(save_path + self.name + self.data.algorithm + "-spectrum" + notion + ".npz",
                     csi_spectrum=self.data.spectrum,
                     csi_algorithm=self.data.algorithm)
            print(self.name, "spectrum save complete", time.asctime(time.localtime(time.time())))

    class _Data:
        def __init__(self, input_name, input_nrx):
            self.name = input_name
            self.nrx = input_nrx
            self.amp = None
            self.phase = None
            self.timestamps = None
            self.sampling_rate = None
            self.length = None
            self.spectrum = None
            self.xlabels = None
            self.algorithm = None
            self.commonfunc = MyCsi._CommonFunctions

        def show_shape(self):
            """
            Shows dimesionality information of csi data.\n
            :return: csi data shape
            """

            try:
                if self.amp is None and self.phase is None:
                    raise DataError("csi data")

            except DataError as e:
                print(e, "\nPlease load data")

            else:
                items = ["no_frames=", "no_subcarriers=", "no_rx_ant=", "no_tx_ant="]
                _list = [a + str(b) for a, b in zip(items, self.amp.shape)]
                print(self.name, "data shape: ", *_list, sep='\n')

        def show_antenna_strength(self):
            """
            Shows the average of absolute values of each antenna.\n
            :return: nrx * ntx matrix
            """

            try:
                if self.amp is None:
                    raise DataError("amplitude")

            except DataError as e:
                print(e, "\nPlease load data")

            else:
                mean_abs = np.mean(np.abs(self.amp), axis=(0, 1))
                return mean_abs

        def rearrange_antenna(self):
            """
            Rearranges rx antenna permutation as [1, 2, 0].\n
            :return: rearranged csi
            """

            try:
                if self.amp is None and self.phase is None:
                    raise DataError("data")

            except DataError as e:
                print(e, "\nPlease load data")

            else:
                print(self.name, "Apply antenna order rearrangement...", time.asctime(time.localtime(time.time())))
                self.amp = self.amp[:, :, [1, 2, 0], :]
                self.phase = self.phase[:, :, [1, 2, 0], :]

        def remove_inf_values(self):
            """
            Removes -inf values in csi amplitude which hinders further calculation.\n
            Replaces packets having -inf values with neighboring ones.\n
            Embodied in spectrum calculating methods.\n
            :return: Processed amplitude
            """
            print("  Apply invalid value removal...", time.asctime(time.localtime(time.time())))

            try:
                if self.amp is None:
                    raise DataError("amplitude: " + str(self.amp))

            except DataError as e:
                print(e, "\nPlease load data")

            else:
                print("  Found", len(np.where(self.amp == float('-inf'))[0]), "-inf values")

                if len(np.where(self.amp == float('-inf'))[0]) != 0:

                    _amp = self.amp.reshape((self.length, -1))
                    good_package_counter = 0
                    for k in range(_amp.shape[0]):
                        flag = any(_amp[k, :] == float('-inf'))
                        if flag is False:
                            good_package_counter += 1

                    if good_package_counter < int(self.length * 0.8):
                        print('  Bad sample detected')
                        return 'bad'

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

        def view_phase_diff(self, packet1=None, packet2=None, ref_antenna=None,
                            autosave=False, notion='', folder_name=''):

            if ref_antenna is None:
                ref_antenna = np.argmax(self.show_antenna_strength())
            if packet1 is None:
                packet1 = np.random.randint(self.length)
            if packet2 is None:
                packet2 = np.random.randint(self.length)

            antennas = list(range(self.nrx))
            antennas.remove(int(ref_antenna))
            recon = self.commonfunc.reconstruct_csi
            csi = recon(self.amp, self.phase)
            phase_diff = np.unwrap(np.angle(csi * csi[:, :, ref_antenna, :][:, :, np.newaxis, :].conj()), axis=1)

            plt.title("Phase Difference of " + str(self.name))
            plt.plot(phase_diff[packet1, :, antennas[0], 0],
                     label='antenna' + str(antennas[0]) + '-' + str(ref_antenna) + ' #' + str(packet1), color='b')
            plt.plot(phase_diff[packet1, :, antennas[1], 0],
                     label='antenna' + str(antennas[1]) + '-' + str(ref_antenna) + ' #' + str(packet1), color='r')
            plt.plot(phase_diff[packet2, :, antennas[0], 0],
                     label='antenna' + str(antennas[0]) + '-' + str(ref_antenna) + ' #' + str(packet2), color='b',
                     linestyle='--')
            plt.plot(phase_diff[packet2, :, antennas[1], 0],
                     label='antenna' + str(antennas[1]) + '-' + str(ref_antenna) + ' #' + str(packet2), color='r',
                     linestyle='--')
            plt.xlabel('#Subcarrier', loc='right')
            plt.ylabel('PhaseDiff / $rad$')
            plt.legend()

            if autosave is True:
                save_path = "../visualization/" + self.name[:4] + '/' + folder_name + '/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_name = save_path + self.name[4:] + notion + '.png'
                plt.savefig(save_name)
                print(self.name, "saved as", save_name, time.asctime(time.localtime(time.time())))
                plt.close()
                return save_name

            else:
                plt.show()
                return 'No saving'

        def view_all_rx(self, metric="amplitude"):
            """
            Plots csi amplitude OR phase for all antennas.\n
            :param metric: 'amplitude' or 'phase'
            :return: value-time plot
            """
            print(self.name, metric, "plotting...", time.asctime(time.localtime(time.time())))

            try:
                if metric == "amplitude":
                    csi_matrix = self.amp

                elif metric == "phase":
                    csi_matrix = self.phase

                else:
                    raise ArgError("metric: " + str(metric))

                if csi_matrix is None:
                    raise DataError("csi data")

            except ArgError as e:
                print(e, "\nPlease specify metric=\"amplitude\" or \"phase\"")

            except DataError as e:
                print(e, "\nPlease load data")

            else:
                for rx in range(csi_matrix.shape[2]):
                    csi_matrix_squeezed = np.squeeze(csi_matrix[:, :, rx, 0])
                    BatchGraph.plot_heatmap(csi_matrix_squeezed, self.timestamps)

                print(self.name, metric, "plot complete", time.asctime(time.localtime(time.time())))

        def view_spectrum(self, threshold=0, sid=0, num_ticks=11, autosave=False, notion='', folder_name=''):
            """
            Plots spectrum. You can select whether save the image or not.\n
            :param threshold: set threshold of spectrum, default is 0 (none)
            :param sid: index of spectrum (for AoA-ToF and AoA-Doppler)
            :param num_ticks: set number of ticks to be plotted in the figure, must be larger than 2. Default is 11
            :param autosave: True or False. Default is False
            :param notion: string, save additional information in filename if autosave
            :param folder_name: if entered, create a folder if autosave is True
            :return: spectrum plot
            """
            print(self.name, "plotting...", time.asctime(time.localtime(time.time())))

            try:
                if self.spectrum is None:
                    raise DataError("spectrum: " + str(self.spectrum))

                if not isinstance(num_ticks, int) or num_ticks < 3:
                    raise ArgError("num_ticks: " + str(num_ticks) + "\nPlease specify an integer larger than 3")

            except DataError as e:
                print(e, "\nPlease compute spectrum")
            except ArgError as e:
                print(e)

            else:
                spectrum = np.array(self.spectrum)
                replace = self.commonfunc.replace_labels

                if threshold != 0:
                    spectrum[spectrum > threshold] = threshold

                if self.algorithm == '_aoa':
                    ax = sns.heatmap(spectrum)
                    label0, label1 = replace(self.timestamps, self.length, num_ticks)

                    ax.yaxis.set_major_formatter(ticker.FixedFormatter([-240, -180, -120, -60, 0, 60, 120, 180]))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(60))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(20))
                    plt.xticks(label0, label1)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_xlabel("Time / $s$")
                    ax.set_ylabel("AoA / $deg$")
                    plt.title(self.name + " AoA Spectrum" + str(notion))

                elif self.algorithm == '_tof':
                    ax = sns.heatmap(spectrum)
                    label0, label1 = replace(self.timestamps, self.length, num_ticks)

                    ax.yaxis.set_major_formatter(ticker.FixedFormatter([-20, 0, 20, 40, 60, 80, 100]))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(40))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
                    plt.xticks(label0, label1)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_xlabel("Time / $s$")
                    ax.set_ylabel("ToF / $ns$")
                    plt.title(self.name + " ToF Spectrum" + str(notion))

                elif self.algorithm == '_doppler':
                    ax = sns.heatmap(spectrum)
                    label0, label1 = replace(self.xlabels, len(self.xlabels), num_ticks)

                    ax.yaxis.set_major_formatter(ticker.FixedFormatter([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
                    plt.xticks(label0, label1)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_xlabel("Time / $s$")
                    ax.set_ylabel("Velocity / $m/s$")
                    plt.title(self.name + " Doppler Spectrum" + str(notion))

                elif self.algorithm == '_aoatof':
                    ax = sns.heatmap(spectrum[sid])
                    ax.yaxis.set_major_formatter(ticker.FixedFormatter([-240, -180, -120, -60, 0, 60, 120, 180]))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(60))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(20))
                    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
                               [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_xlabel("ToF / $ns$")
                    ax.set_ylabel("AoA / $deg$")
                    plt.title(self.name + " AoA-ToF Spectrum" + str(notion))

                elif self.algorithm == '_aoadoppler':
                    ax = sns.heatmap(spectrum[sid])

                    ax.yaxis.set_major_formatter(ticker.FixedFormatter([-240, -180, -120, -60, 0, 60, 120, 180]))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(60))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(20))
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
                    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
                    ax.set_xlabel("Velocity / $m/s$")
                    ax.set_ylabel("AoA / $deg$")
                    plt.title(self.name + " AoA-Doppler Spectrum" + str(notion))

                ax.collections[0].colorbar.set_label('Power / $dB$')

                print(self.name, "plot complete", time.asctime(time.localtime(time.time())))

                if autosave is False:
                    plt.show()
                    return "No saving"

                elif autosave is True:
                    save_path = "../visualization/" + self.name[:4] + '/' + str(folder_name) + '/'

                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

                    savename = save_path + self.name[4:] + self.algorithm + str(notion) + '.png'
                    plt.savefig(savename)
                    print(self.name, "saved as", savename, time.asctime(time.localtime(time.time())))
                    plt.close()
                    return savename

                else:
                    raise ArgError("autosave\nPlease specify autosave=\"True\" or \"False\"")

    class _CommonFunctions:
        """
        Collection of static methods that may be used in other methods.\n
        """

        @staticmethod
        def smooth_csi(input_csi, rx=2, sub=15):
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

        @staticmethod
        def reconstruct_csi(input_amp, input_phase):
            """
            Static method.\n
            Reconstructs csi data as complex numbers.\n
            :param input_amp: csi amplitude
            :param input_phase: csi phase
            :return: reconstructed csi
            """
            reconstruct_csi = np.squeeze(input_amp) * np.exp(1.j * np.squeeze(input_phase))

            return reconstruct_csi.reshape(input_amp.shape)

        @staticmethod
        def noise_space(input_csi, ntx=1):
            """
            Static method.\n
            Calculates self-correlation and eigen vectors of given csi.\n
            :param input_csi: complex csi
            :param ntx: number of tx antenna, default is 1
            :return: noise space vectors
            """
            if len(input_csi.shape) > 2:
                input_csi = np.squeeze(input_csi)
            value, vector = np.linalg.eigh(input_csi.T.dot(np.conjugate(input_csi)))
            descend_order_index = np.argsort(-value)
            vector = vector[:, descend_order_index]
            noise_space = vector[:, ntx:]

            return noise_space

        @staticmethod
        def replace_labels(input_timestamps, input_length, input_ticks):
            """
            Static method.\n
            Generates a list of timestamps to plot as x-axis labels.\n
            :param input_timestamps: ordinarily input self.data.timestamps
            :param input_length: ordinarily input self.data.length
            :param input_ticks: how many labels you need (including start and end)
            :return: a list of timestamps
            """

            indices = [i * input_length // (input_ticks - 1) for i in range(input_ticks - 1)]
            indices.append(input_length - 1)

            labels = indices if len(np.where(np.array(input_timestamps) < 0)[0]) > 0 else [
                float('%.3f' % x) for x in input_timestamps[indices]]

            return indices, labels

    def aoa_by_music(self, input_theta_list=np.arange(-180, 181, 1.), smooth=False):
        """
        Computes AoA spectrum by MUSIC.\n
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
        recon = self.commonfunc.reconstruct_csi
        smoothing = self.commonfunc.smooth_csi
        noise = self.commonfunc.noise_space

        print(self.name, "AoA by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.amp is None:
                raise DataError("amplitude: " + str(self.data.amp) + "\nPlease load data")

            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase) + "\nPlease load data")

            if smooth not in (True, False):
                raise ArgError("smooth:" + str(smooth))

            # Replace -inf values with neighboring packets before computing
            if self.data.remove_inf_values() == 'bad':
                raise DataError("values: too many -inf values\nSample aborted")

            if smooth is True:
                print(self.name, "apply Smoothing via SpotFi...")

            # Subcarriers from -58 to 58, step = 4
            subfreq_list = np.arange(center_freq - 58 * delta_subfreq, center_freq + 62 * delta_subfreq,
                                     4 * delta_subfreq).reshape(-1, 1)
            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            theta_list = np.array(input_theta_list).reshape(-1, 1)
            spectrum = np.zeros((len(input_theta_list), self.data.length))

            for i in range(self.data.length):

                if smooth is True:
                    temp_amp = smoothing(np.squeeze(self.data.amp[i]))
                    temp_phase = smoothing(np.squeeze(self.data.phase[i]))

                else:
                    temp_amp = self.data.amp[i]
                    temp_phase = self.data.phase[i]

                csi = np.squeeze(recon(temp_amp, temp_phase))
                noise_space = noise(csi, ntx)

                if smooth is True:
                    steering_vector = np.exp([mjtwopi * dist_antenna * (np.sin(theta_list * torad) *
                                              no_antenna).dot(sub_freq) / lightspeed
                                              for no_antenna in antenna_list[:2]
                                              for sub_freq in subfreq_list[:15]])
                else:
                    steering_vector = np.exp(mjtwopi * dist_antenna * np.sin(theta_list * torad).dot(
                                                antenna_list.T) * center_freq / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.data.spectrum = np.log(spectrum)
            self.data.algorithm = '_aoa'
            print(self.name, "AoA by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")

    def tof_by_music(self, input_dt_list=np.arange(0, 1.e-7, 5.e-10)):

        mjtwopi = self.mjtwopi
        delta_subfreq = self.delta_subfreq
        center_freq = self.center_freq
        ntx = self.ntx
        recon = self.commonfunc.reconstruct_csi
        noise = self.commonfunc.noise_space

        print(self.name, "ToF by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.amp is None:
                raise DataError("amplitude: " + str(self.data.amp) + "\nPlease load data")

            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase) + "\nPlease load data")

            # Replace -inf values with neighboring packets before computing
            if self.data.remove_inf_values() == 'bad':
                raise DataError("values: too many -inf values\nSample aborted")

            subcarrier_list = np.arange(-58, 62, 4)
            subfreq_list = np.arange(center_freq - 58 * delta_subfreq, center_freq + 62 * delta_subfreq,
                                     4 * delta_subfreq).reshape(-1, 1)
            dt_list = np.array(input_dt_list).reshape(-1, 1)
            spectrum = np.zeros((len(input_dt_list), self.data.length))

            for i in range(self.data.length):

                csi = np.squeeze(recon(self.data.amp[i], self.data.phase[i]))
                noise_space = noise(csi.T, ntx)

                steering_vector = np.exp(mjtwopi * dt_list.dot(subfreq_list.T))

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.data.spectrum = np.log(spectrum)
            self.data.algorithm = '_tof'
            print(self.name, "ToF by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def doppler_by_music(self, input_velocity_list=np.arange(-5, 5.05, 0.05),
                         window_length=500,
                         stride=500,
                         raw_timestamps=False,
                         raw_window=False):
        """
        Computes Doppler spectrum by MUSIC.\n
        Involves self-calibration, windowed dynamic component extraction and resampling (if specified).\n
        :param input_velocity_list: list of velocities. Default = -5~5
        :param window_length: window length for each step
        :param stride: stride for each step
        :param raw_timestamps: whether use original timestamps. Default is False
        :param raw_window: whether skip extracting dynamic CSI. Default is False
        :return: Doppler spectrum by MUSIC stored in self.data.spectrum
        """
        lightspeed = self.lightspeed
        center_freq = self.center_freq
        mjtwopi = self.mjtwopi
        ntx = self.ntx
        recon = self.commonfunc.reconstruct_csi
        noise = self.commonfunc.noise_space

        print(self.name, "Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.amp is None:
                raise DataError("amplitude: " + str(self.data.amp) + "\nPlease load data")

            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase)+ "\nPlease load data")

            # Replace -inf values with neighboring packets before computing
            if self.data.remove_inf_values() == 'bad':
                raise DataError("values: too many -inf values\nSample aborted")

            # Each window has ts of packets (1 / sampling_rate * window_length = t)
            delay_list = np.arange(0, window_length, 1.).reshape(-1, 1) / self.data.sampling_rate
            velocity_list = np.array(input_velocity_list).reshape(-1, 1)

            pick_antenna = np.argmin(self.data.show_antenna_strength())

            spectrum = np.zeros((len(input_velocity_list), (self.data.length - window_length) // stride))
            temp_timestamps = np.zeros((self.data.length - window_length) // stride)
            csi = np.squeeze(recon(self.data.amp, self.data.phase))

            for i in range((self.data.length - window_length) // stride):

                csi_windowed = csi[i * stride: i * stride + window_length, :, pick_antenna]

                if raw_window is True:
                    noise_space = noise(csi_windowed.T, ntx)
                else:
                    # Using windowed dynamic extraction
                    noise_space = noise((csi_windowed - np.mean(csi_windowed, axis=0)).T, ntx)

                if raw_timestamps is True:
                    # Using original timestamps (possibly uneven intervals)
                    delay_list = self.data.timestamps[i * stride: i * stride + window_length] - \
                                 self.data.timestamps[i * stride]

                steering_vector = np.exp(mjtwopi * center_freq * velocity_list.dot(delay_list.T) / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

                temp_timestamps[i] = self.data.timestamps[i * stride]

            self.data.spectrum = np.log(spectrum)
            self.data.algorithm = '_doppler'
            self.data.xlabels = temp_timestamps

            print(self.name, "Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def aoa_tof_by_music(self, input_theta_list=np.arange(-180, 181, 1.),
                         input_dt_list=np.arange(0, 1.e-7, 5.e-10),
                         smooth=False):
        """
        Computes AoA-ToF spectrum by MUSIC.\n
        :param input_theta_list:  list of angels, default = -90~90
        :param input_dt_list: list of time measurements, default = 0~8e-8
        :param smooth:  whether apply SpotFi smoothing or not, default = False
        :return:  AoA-ToF spectrum by MUSIC stored in self.data.spectrum
        """

        lightspeed = self.lightspeed
        center_freq = self.center_freq
        dist_antenna = self.dist_antenna
        mjtwopi = self.mjtwopi
        torad = self.torad
        delta_subfreq = self.delta_subfreq
        nsub = self.nsub
        nrx = self.nrx
        ntx = self.ntx
        recon = self.commonfunc.reconstruct_csi
        smoothing = self.commonfunc.smooth_csi
        noise = self.commonfunc.noise_space

        print(self.name, "AoA-ToF by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.amp is None:
                raise DataError("amplitude: " + str(self.data.amp) + "\nPlease load data")

            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase) + "\nPlease load data")

            if smooth not in (True, False):
                raise ArgError("smooth:" + str(smooth))

            # Replace -inf values with neighboring packets before computing
            if self.data.remove_inf_values() == 'bad':
                raise DataError("values: too many -inf values\nSample aborted")

            if smooth is True:
                print(self.name, "apply Smoothing via SpotFi...")

            # Subcarriers from -58 to 58, step = 4
            subcarrier_list = np.arange(-58, 62, 4)
            subfreq_list = np.arange(center_freq - 58 * delta_subfreq, center_freq + 62 * delta_subfreq,
                                     4 * delta_subfreq)
            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            theta_list = np.array(input_theta_list).reshape(-1, 1)
            dt_list = np.array(input_dt_list).reshape(-1, 1)

            steering_aoa = np.exp(mjtwopi * dist_antenna * np.sin(theta_list * torad).dot(
                antenna_list.T) * center_freq / lightspeed).reshape(-1, 1)

            spectrum = np.zeros((self.data.length, len(input_theta_list), len(input_dt_list)))

            for i in range(self.data.length):

                if smooth is True:
                    temp_amp = smoothing(np.squeeze(self.data.amp[i]))
                    temp_phase = smoothing(np.squeeze(self.data.phase[i]))

                else:
                    temp_amp = self.data.amp[i]
                    temp_phase = self.data.phase[i]

                csi = np.squeeze(recon(temp_amp, temp_phase)).reshape(1, -1)  # nrx * nsub columns
                noise_space = noise(csi, ntx)

                for j, tof in enumerate(dt_list):

                    if smooth is True:
                        steering_vector = np.exp(mjtwopi * dist_antenna * np.sin(theta_list * torad).dot(
                            antenna_list[:2].dot(subfreq_list[:15])) / lightspeed)
                    else:
                        steering_tof = np.exp(mjtwopi * subfreq_list * tof).reshape(-1, 1)
                        steering_vector = steering_tof.dot(steering_aoa.T).reshape(nsub, len(input_theta_list), nrx)
                        steering_vector = steering_vector.swapaxes(0, 1).reshape(len(input_theta_list), nrx * nsub)

                    a_en = np.conjugate(steering_vector).dot(noise_space)
                    spectrum[i, :, j] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.data.spectrum = np.log(spectrum)
            self.data.algorithm = '_aoatof'
            print(self.name, "AoA-ToF by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")

    def aoa_doppler_by_music(self, input_theta_list=np.arange(-180, 181, 1.),
                             input_velocity_list=np.arange(-5, 5.05, 0.05),
                             window_length=500,
                             stride=500,
                             raw_timestamps=False,
                             raw_window=False):
        """
        Computes AoA-Doppler spectrum by MUSIC.\n
        :param input_theta_list:  list of angels, default = -90~90
        :param input_velocity_list: list of velocities. Default = -5~5
        :param window_length: window length for each step
        :param stride: stride for each step
        :param raw_timestamps: whether use original timestamps. Default is False
        :param raw_window: whether skip extracting dynamic CSI. Default is False
        :return:  AoA-Doppler spectrum by MUSIC stored in self.data.spectrum
        """

        lightspeed = self.lightspeed
        center_freq = self.center_freq
        dist_antenna = self.dist_antenna
        mjtwopi = self.mjtwopi
        torad = self.torad
        nrx = self.nrx
        ntx = self.ntx
        nsub = self.nsub
        recon = self.commonfunc.reconstruct_csi
        noise = self.commonfunc.noise_space

        print(self.name, "AoA-Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.amp is None:
                raise DataError("amplitude: " + str(self.data.amp) + "\nPlease load data")

            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase) + "\nPlease load data")

            # Replace -inf values with neighboring packets before computing
            if self.data.remove_inf_values() == 'bad':
                raise DataError("values: too many -inf values\nSample aborted")

            # Each window has ts of packets (1 / sampling_rate * window_length = t)
            delay_list = np.arange(0, window_length, 1.).reshape(-1, 1) / self.data.sampling_rate
            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            theta_list = np.array(input_theta_list).reshape(-1, 1)
            velocity_list = np.array(input_velocity_list).reshape(-1, 1)

            steering_aoa = np.exp(mjtwopi * dist_antenna * np.sin(theta_list * torad).dot(
                antenna_list.T) * center_freq / lightspeed).reshape(-1, 1)
            spectrum = np.zeros(((self.data.length - window_length) // stride, len(input_theta_list),
                                 len(input_velocity_list)))
            temp_timestamps = np.zeros((self.data.length - window_length) // stride)
            csi = np.squeeze(recon(self.data.amp, self.data.phase))

            # Using windowed dynamic extraction
            for i in range((self.data.length - window_length) // stride):

                csi_windowed = csi[i * stride: i * stride + window_length, :, :]

                if raw_window is True:
                    csi_dynamic = csi_windowed
                else:
                    # Using windowed dynamic extraction
                    csi_dynamic = csi_windowed - np.mean(csi_windowed, axis=0)

                csi_dynamic = csi_dynamic.swapaxes(0, 1).reshape(nsub, window_length * nrx)
                noise_space = noise(csi_dynamic, ntx)

                if raw_timestamps is True:
                    # Using original timestamps (possibly uneven intervals)
                    delay_list = self.data.timestamps[i * stride: i * stride + window_length] - \
                                 self.data.timestamps[i * stride]

                for j, velocity in enumerate(velocity_list):

                    steering_doppler = np.exp(mjtwopi * center_freq * delay_list * velocity / lightspeed).reshape(-1, 1)
                    steering_vector = steering_doppler.dot(steering_aoa.T).reshape(len(delay_list), len(input_theta_list), nrx)
                    steering_vector = steering_vector.swapaxes(0, 1).reshape(len(input_theta_list), nrx * len(delay_list))

                    a_en = np.conjugate(steering_vector).dot(noise_space)
                    spectrum[i, :, j] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.data.spectrum = np.log(spectrum)
            self.data.algorithm = '_aoadoppler'
            self.data.xlabels = temp_timestamps
            print(self.name, "AoA-Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")

    def self_calibrate(self, ref_antenna=None):
        """
        Self-calibrate using given antenna or the strongest antenna. Default is the strongest antenna.\n
        :param: ref_antenna: If not specified, use the strongest antenna
        :return: self-calibrated amplitude and phase
        """
        recon = self.commonfunc.reconstruct_csi
        nsub = self.nsub

        if ref_antenna is None:
            strengths = self.data.show_antenna_strength()
            ref_antenna = np.argmax(strengths)

        _csi = recon(self.data.amp, self.data.phase) * recon(
            self.data.amp, self.data.phase)[:, :, ref_antenna, :][:, :, np.newaxis, :].conj()

        self.data.amp = np.abs(_csi)
        self.data.phase = np.angle(_csi)

        return _csi

    def sanitize_phase(self):
        """
        Also known as SpotFi Algorithm1.\n
        Removes Sampling Time Offset shared by all rx antennas.\n
        :return: sanitized phase
        """

        nrx = self.nrx
        nsub = self.nsub

        print(self.name, "apply SpotFi Algorithm1 to remove STO", time.asctime(time.localtime(time.time())))

        try:
            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase))

            fit_x = np.concatenate([np.arange(0, nsub) for i in range(nrx)])
            fit_y = np.unwrap(np.squeeze(self.data.phase), axis=1).swapaxes(1, 2).reshape(self.data.length, -1)

            A = np.stack((fit_x, np.ones_like(fit_x)), axis=-1)
            fit = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(fit_y.T).T
            # fit = np.array([np.polyfit(fit_x, fit_y[i], 1) for i in range(self.data.length)])

            self.data.phase = np.unwrap(self.data.phase, axis=1) - \
                              np.arange(nsub).reshape(1, nsub, 1, 1) * fit[:, 0].reshape(self.data.length, 1, 1, 1)

        except DataError as e:
            print(e, "\nPlease load data")

    def calibrate_phase(self, reference_antenna=0, cal_dict=None):
        """
        Calibrates phase with reference csi data files.\n
        Multiple files is supported.\n
        Reference files are recommended to be collected at 50cm at certain degrees (eg. 0, +-30, +-60).\n
        Removes Initial Phase Offset.\n
        :param reference_antenna: select one antenna with which to calculate phase difference between antennas.
        Default is 0
        :param cal_dict: formatted as "{'xx': MyCsi}", where xx is degrees
        :return: calibrated phase
        """
        nrx = self.nrx
        nsub = self.nsub
        mjtwopi = self.mjtwopi
        distance_antenna = self.dist_antenna
        torad = self.torad
        lightspeed = self.lightspeed
        center_freq = self.center_freq
        recon = self.commonfunc.reconstruct_csi

        print(self.name, "apply phase calibration according to", cal_dict.keys(), "...",
              time.asctime(time.localtime(time.time())))

        try:
            if self.data.phase is None:
                raise DataError("phase: " + str(self.data.phase))

            if reference_antenna not in (0, 1, 2):
                raise ArgError("reference_antenna: " + str(reference_antenna))

            if cal_dict is None:
                raise DataError("reference: " + str(cal_dict))

            ipo = []
            # cal_dict: "{'xx': MyCsi}"

            for key, value in cal_dict.items():

                if not isinstance(value, MyCsi):
                    raise DataError("reference csi: " + str(value) + "\nPlease input MyCsi instance.")

                if value.data.phase is None:
                    raise DataError("reference phase: " + str(value.data.phase))

                ref_angle = eval(key)

                ref_csi = recon(value.data.amp, value.data.phase)
                ref_diff = np.mean(ref_csi * ref_csi[:, :, reference_antenna, :][:, :, np.newaxis, :].conj(),
                                   axis=(0, 1))
                true_diff = np.exp([mjtwopi * distance_antenna * antenna * center_freq * np.sin(
                    ref_angle * torad) / lightspeed for antenna in range(nrx)]).reshape(-1, 1)

                ipo.append(ref_diff.reshape(-1, 1) * true_diff.conj())

            ipo = np.squeeze(np.mean(ipo, axis=0))
            print(ipo)

            current_csi = recon(self.data.amp, self.data.phase)
            calibrated_csi = current_csi * ipo[np.newaxis, np.newaxis, :, np.newaxis].conj()

            self.data.amp = np.abs(calibrated_csi)
            self.data.phase = np.angle(calibrated_csi)

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify an integer from 0~2")

    def extract_dynamic(self, mode='overall', window_length=31, reference_antenna=0):
        """
        Removes the static component from csi.\n
        :param mode: 'overall' or 'running' (in terms of averaging) or 'highpass'. Default is 'overall'
        :param window_length: if mode is 'running', specify a window length for running mean. Default is 31
        :param reference_antenna: select one antenna with which to remove random phase offsets. Default is 0
        :return: phase and amplitude of dynamic component of csi
        """
        nrx = self.nrx
        ntx = self.ntx
        nsub = self.nsub
        sampling_rate = self.data.sampling_rate
        recon = self.commonfunc.reconstruct_csi

        print(self.name, "apply dynamic component extraction...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.amp is None or self.data.phase is None:
                raise DataError("csi data")

            if reference_antenna not in range(nrx):
                raise ArgError("reference_antenna: " + str(reference_antenna) + "\nPlease specify an integer from 0~2")

            if not isinstance(window_length, int) or window_length < 1 or window_length > self.data.length:
                raise ArgError("window_length: " + str(window_length) + "\nPlease specify an integer larger than 0 and"
                                                                        "not larger than data length")

            strengths = self.data.show_antenna_strength()
            ref_antenna = np.argmax(strengths)

            complex_csi = recon(self.data.amp, self.data.phase)
            conjugate_csi = np.conjugate(complex_csi[:, :, ref_antenna, None]).repeat(3, axis=2)
            hc = (complex_csi * conjugate_csi).reshape((-1, nsub, nrx))

            if mode == 'overall':
                average_hc = np.mean(hc, axis=0).reshape((1, nsub, nrx)).repeat(self.data.length, axis=0)

            elif mode == 'running':
                average_hc = np.array([[np.convolve(np.squeeze(hc[:, sub, antenna, :]),
                                        np.ones(window_length) / window_length, mode='same')
                                        for sub in range(nsub)]
                                      for antenna in range(nrx)]).swapaxes(0, 2).reshape((-1, nsub, nrx))
            elif mode == 'highpass':
                for packet in range(self.data.length):
                    for antenna in range(nrx):
                        hc[packet, :, antenna] = highpass(hc[packet, :, antenna], cutoff=10, fs=sampling_rate, order=5)
                average_hc = 0 + 0j

            else:
                raise ArgError("mode: " + str(mode) + "\nPlease specify mode=\"overall\", \"running\" or \"highpass\"")

            dynamic_csi = hc - average_hc
            self.data.amp = np.abs(dynamic_csi)
            self.data.phase = np.angle(dynamic_csi)

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e)

    def resample_packets(self, sampling_rate=100):
        """
        Resample from raw CSI to reach a specified sampling rate.\n
        Strongly recommended when uniform interval is required.\n
        :param sampling_rate: sampling rate in Hz after resampling. Must be less than 3965.
        Default is 100
        :return: Resampled csi data
        """
        print(self.name, "resampling at " + str(sampling_rate) + "Hz...", time.asctime(time.localtime(time.time())))

        try:
            if self.data.amp is None or self.data.phase is None:
                raise DataError("csi data")

            if not isinstance(sampling_rate, int):
                raise ArgError("sampling_rate: " + str(sampling_rate))

            new_interval = 1. / sampling_rate

            if len(np.where(self.data.timestamps < 0)[0]) > 0:
                print(self.name, "Timestamping bug detected!")
                return 'bad'

            else:
                new_length = int(self.data.timestamps[-1] * sampling_rate) + 1  # Flooring
            resample_indicies = []

            for i in range(new_length):

                index = np.searchsorted(self.data.timestamps, i * new_interval)

                if index > 0 and (
                        index == self.data.length or
                        abs(self.data.timestamps[index] - i * new_interval) >
                        abs(self.data.timestamps[index - 1] - i * new_interval)):
                    index -= 1

                resample_indicies.append(index)

            self.data.amp = self.data.amp[resample_indicies]
            self.data.phase = self.data.phase[resample_indicies]
            self.data.timestamps = self.data.timestamps[resample_indicies]
            self.data.length = new_length
            self.data.sampling_rate = sampling_rate

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify an integer less than 3965")


if __name__ == '__main__':

    # CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

    # Raw CSI naming: csi-XXXX<date>-Y<A or B>-ZZ<#exp>-.dat

    filepath = "../data/1030/"
    filenames = os.listdir(filepath)
    for file in filenames:
        name = file[3:-4]
        mypath = filepath + file
        # npzpath = "npsave/csi" + name + "-csis.npz"
        # pmpath = "npsave/" + name + "-spectrum.npz"
        _csi = MyCsi(name, mypath)
        _csi.load_data()
        _csi.save_csi(name)
