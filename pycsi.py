from CSIKit.tools.batch_graph import BatchGraph
from CSIKit.filters.passband import highpass

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import signal
import csi_loader


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


class MyConfigs:
    def __init__(self, center_freq=5.32, bandwidth=20, sampling_rate=1000):
        self.lightspeed = 299792458
        self.center_freq = center_freq * 1e+09  # in GHz
        self.bandwidth = bandwidth  # in MHz
        self.sampling_rate = sampling_rate
        self.dist_antenna = self.lightspeed / self.center_freq / 2.  # half-wavelength
        self.torad = np.pi / 180
        self.delta_subfreq = 3.125e+05  # 312.5KHz (fixed)
        self.nrx = 3
        self.ntx = 1
        self.nsub = 30
        self.subfreq_list = self.__gen_subcarrier__()
        self.antenna_list = np.arange(0, self.nrx, 1.).reshape(-1, 1)
        self.tx_rate = 0x4101

    def __gen_subcarrier__(self):
        if self.bandwidth == 40:
            return np.arange(self.center_freq - 58 * self.delta_subfreq,
                             self.center_freq + 62 * self.delta_subfreq,
                             4 * self.delta_subfreq).reshape(-1, 1)
        elif self.bandwidth == 20:
            return np.arange(self.center_freq - 28 * self.delta_subfreq,
                             self.center_freq + 32 * self.delta_subfreq,
                             2 * self.delta_subfreq).reshape(-1, 1)
        else:
            return None


class MyCommonFuncs:
    """
    Collection of static methods that may be used in other methods.\n
    """
    def __init__(self):
        pass

    @staticmethod
    def smooth_csi(input_csi, rx=2, sub=15):
        """
        Applies SpotFi smoothing technique.\n
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
    def noise_space(input_csi):
        """
        Calculates self-correlation and eigen vectors of given csi.\n
        For AoA, please input CSI of (nsub, nrx).\n
        For ToF and Doppler, please input CSI of (nrx, nsub).\n
        :param input_csi: complex csi
        :param ntx: number of tx antenna, default is 1
        :return: noise space vectors
        """

        input_csi = np.squeeze(input_csi)

        value, vector = np.linalg.eigh(input_csi.T.dot(np.conjugate(input_csi)))
        descend_order_index = np.argsort(-value)
        vector = vector[:, descend_order_index]
        noise_space = vector[:, 1:]

        return noise_space

    @staticmethod
    def dynamic(input_csi, ref, reference_antenna, subtract_mean=True):
        if ref == 'rx':
            phase_diff = input_csi * input_csi[:, :, reference_antenna, :][..., np.newaxis, :].conj().repeat(3, axis=2)
        elif ref == 'tx':
            phase_diff = input_csi * input_csi[:, :, :, reference_antenna][..., np.newaxis].conj().repeat(3, axis=3)
        if subtract_mean is True:
            static = np.mean(phase_diff, axis=0)
            dynamic = phase_diff - static
        else:
            dynamic = phase_diff
        return dynamic

    @staticmethod
    def windowed_divison(input_csi, ref, reference_antenna, subtract_mean=True):

        re_csi = (np.abs(input_csi) + 1.e-6) * np.exp(1.j * np.abs(input_csi))
        if ref == 'rx':
            phase_diff = input_csi / re_csi[:, :, reference_antenna, :][..., np.newaxis, :].repeat(3, axis=2)
        elif ref == 'tx':
            phase_diff = input_csi / re_csi[:, :, :, reference_antenna][..., np.newaxis].repeat(3, axis=3)

        if subtract_mean is True:
            static = np.mean(phase_diff, axis=0)
            dynamic = phase_diff - static
        else:
            dynamic = phase_diff
        return dynamic

    @staticmethod
    def highpass(fs=1000, cutoff=2, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a


class MySpectrumViewer:

    def __init__(self, name, spectrum, timestamps, num_ticks=11):
        self.name = name
        self.spectrum = spectrum
        self.timestamps = timestamps
        self.num_ticks = num_ticks
        self.algorithm = None

    def view(self, threshold=0, autosave=False, *args, **kwargs):
        print(self.name, "plotting...", end='')

        try:
            if self.spectrum is None:
                raise DataError("spectrum: " + str(self.spectrum))

        except DataError as e:
            print(e, "\nPlease compute spectrum")
        except ArgError as e:
            print(e)

        else:
            if threshold != 0:
                self.spectrum[self.spectrum > threshold] = threshold
            self.show(*args, **kwargs)
            print("Done")

        if autosave is False:
            plt.tight_layout()
            plt.show()
            return "No saving"

        elif autosave is True:
            notion = str(kwargs['notion']) if 'notion' in kwargs.keys() else ''
            folder = str(kwargs['folder']) + '/' if 'folder' in kwargs.keys() else ''
            save_path = os.path.join("../visualization", self.name[:4], folder)
            save_name = self.name[4:] + self.algorithm + notion + '.png'

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
            print(self.name, "saved as", save_name, time.asctime(time.localtime(time.time())))
            plt.close()
            return save_name

        else:
            raise ArgError("autosave\nPlease specify autosave=\"True\" or \"False\"")

    def show(self, *args, **kwargs):
        pass

    @staticmethod
    def replace(input_timestamps: list, input_ticks: int):
        """
        Generates a list of timestamps to be plotted as x-axis labels.\n
        :param input_timestamps: timestamps
        :param input_ticks: number of ticks (including start and end)
        :return: a list of timestamps
        """

        indices = [i * len(input_timestamps) // (input_ticks - 1) for i in range(input_ticks - 1)]
        indices.append(len(input_timestamps) - 1)

        labels = [float('%.1f' % input_timestamps[x]) for x in indices]

        return indices, labels


class AoAViewer(MySpectrumViewer):

    def __init__(self, *args, **kwargs):
        MySpectrumViewer.__init__(self, *args, **kwargs)
        self.algorithm = '_AoA'

    def show(self, srange=None, notion=''):
        if isinstance(srange, list):
            ax = sns.heatmap(self.spectrum[srange])
            label0, label1 = self.replace(self.timestamps[srange], self.num_ticks)
        else:
            ax = sns.heatmap(self.spectrum)
            label0, label1 = self.replace(self.timestamps, self.num_ticks)

        ax.yaxis.set_major_formatter(ticker.FixedFormatter([120, 90, 60, 30, 0, -30, -60, -90]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
        plt.xticks(label0, label1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Time / $s$")
        ax.set_ylabel("AoA / $deg$")
        plt.title(self.name + " AoA Spectrum" + str(notion))
        ax.collections[0].colorbar.set_label('Power / $dB$')


class AoDViewer(MySpectrumViewer):

    def __init__(self, *args, **kwargs):
        MySpectrumViewer.__init__(self, *args, **kwargs)
        self.algorithm = '_AoD'

    def show(self, srange=None, notion=''):
        if isinstance(srange, list):
            ax = sns.heatmap(self.spectrum[srange])
            label0, label1 = self.replace(self.timestamps[srange], self.num_ticks)
        else:
            ax = sns.heatmap(self.spectrum)
            label0, label1 = self.replace(self.timestamps, self.num_ticks)

        ax.yaxis.set_major_formatter(ticker.FixedFormatter([120, 90, 60, 30, 0, -30, -60, -90]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
        plt.xticks(label0, label1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Time / $s$")
        ax.set_ylabel("AoD / $deg$")
        plt.title(self.name + " AoD Spectrum" + str(notion))
        ax.collections[0].colorbar.set_label('Power / $dB$')


class ToFViewer(MySpectrumViewer):

    def __init__(self, *args, **kwargs):
        MySpectrumViewer.__init__(self, *args, **kwargs)
        self.algorithm = '_ToF'

    def show(self, srange=None, notion=''):
        if isinstance(srange, list):
            ax = sns.heatmap(self.spectrum[srange])
            label0, label1 = self.replace(self.timestamps[srange], self.num_ticks)
        else:
            ax = sns.heatmap(self.spectrum)
            label0, label1 = self.replace(self.timestamps, self.num_ticks)

        ax.yaxis.set_major_formatter(ticker.FixedFormatter([450, 400, 300, 200, 100, 0, -100]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(20))
        plt.xticks(label0, label1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Time / $s$")
        ax.set_ylabel("ToF / $ns$")
        plt.title(self.name + " ToF Spectrum" + str(notion))
        ax.collections[0].colorbar.set_label('Power / $dB$')


class DopplerViewer(MySpectrumViewer):

    def __init__(self, xlabels, *args, **kwargs):
        MySpectrumViewer.__init__(self, *args, **kwargs)
        self.xlabels = xlabels
        self.algorithm = '_Doppler'

    def show(self, notion=''):
        ax = sns.heatmap(self.spectrum)
        label0, label1 = self.replace(self.xlabels, self.num_ticks)

        ax.yaxis.set_major_formatter(ticker.FixedFormatter([6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
        plt.xticks(label0, label1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel("Time / $s$")
        ax.set_ylabel("Velocity / $m/s$")
        plt.title(self.name + " Doppler Spectrum" + str(notion))
        ax.collections[0].colorbar.set_label('Power / $dB$')


class AoAToFViewer(MySpectrumViewer):

    def __init__(self, *args, **kwargs):
        MySpectrumViewer.__init__(self, *args, **kwargs)
        self.algorithm = '_AoAToF'

    def show(self, sid=0, notion=''):
        ax = sns.heatmap(self.spectrum[sid])

        ax.yaxis.set_major_formatter(ticker.FixedFormatter([120, 90, 60, 30, 0, -30, -60, -90]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        ax.set_xlabel("Velocity / $m/s$")
        ax.set_ylabel("AoA / $deg$")
        plt.title(self.name + " AoA-Doppler Spectrum" + str(notion))
        ax.collections[0].colorbar.set_label('Power / $dB$')


class AoADopplerViewer(MySpectrumViewer):

    def __init__(self, *args, **kwargs):
        MySpectrumViewer.__init__(self, *args, **kwargs)
        self.algorithm = '_AoADoppler'

    def show(self, sid=0, notion=''):
        ax = sns.heatmap(self.spectrum[sid])

        ax.yaxis.set_major_formatter(ticker.FixedFormatter([120, 90, 60, 30, 0, -30, -60, -90]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        ax.set_xlabel("Velocity / $m/s$")
        ax.set_ylabel("AoA / $deg$")
        plt.title(self.name + " AoA-Doppler Spectrum" + str(notion))
        cb = ax.collections[0].colorbar
        cb.set_label('Power / $dB$')
        cb.ax.set_yticklabels(["{:.2f}".format(i) for i in cb.get_ticks()])


class MyCsi:
    """
    Main functionalities of csi processing.
    """

    def __init__(self, configs: MyConfigs, input_name='', path=None):
        self.name = str(input_name)
        self.path = path

        self.csi = None
        self.timestamps = None
        self.abs_timestamps = None
        self.length = None
        self.actual_sr = None
        self.labels = None

        self.spectrum = None
        self.xlabels = None
        self.algorithm = None
        self.viewer = None

        self.configs = configs
        self.commonfunc = MyCommonFuncs()

    def __str__(self):
        return 'MyCsi-' + self.name

    def __repr__(self):
        return 'MyCsi-' + self.name

    def load_data(self, remove_sm=False):
        """
        Loads csi data into current MyCsi instance.\n
        Supports .dat (raw) and .npz (csi_amp, csi_phase, csi_timestamps).\n
        :return: csi data
        """
        try:
            if self.path is None or not os.path.exists(self.path):
                raise PathError(str(self.path))
            if self.path[-3:] not in ('dat', 'npy'):
                raise DataError("file: " + str(self.path))

        except PathError as e:
            print(e, "\nPlease check the path")
        except DataError as e:
            print(e, "\nFile not supported. Please input .dat or .npy")

        else:
            csi = None
            t = None
            if self.path[-3:] == "dat":
                print(self.name, "raw load start...", time.asctime(time.localtime(time.time())))
                csi, t = csi_loader.dat2npy(self.path, None, autosave=False)

            elif self.path[-3:] == "npy":
                print(self.name, "npy load start...", time.asctime(time.localtime(time.time())))
                csi, t, r, d = csi_loader.load_npy(self.path)

            self.length = len(csi)
            if remove_sm is True and self.configs.ntx != 1:
                print('Removing sm...', end='')
                for i in range(self.length):
                    csi[i] = csi_loader.remove_sm(csi[i], self.configs.tx_rate)
                print('Done')

            self.csi = csi
            self.timestamps = np.array(t - t[0]) / 1.e6
            self.actual_sr = self.length / self.timestamps[-1]
            print(self.name, self.csi.shape, "load complete", time.asctime(time.localtime(time.time())))

    def load_label(self, path):
        """
        Loads label from .csv file.\n
        If the CSI sample is the environment, input 'env' to apply static labels to the CSI.\n
        :param path: label file path, or 'env'
        :return: list of labels corresponding to all packets
        """
        print('Loading label file...', end='')
        labels = []
        if path == 'env':
            print('Done')
            print('Environment label')
            self.labels = {'static': list(range(self.length)),
                           'dynamic': [],
                           'period': []
                           }
        else:
            with open(path) as f:
                for i, line in enumerate(f):
                    if i > 0:
                        labels.append([eval(line.split(',')[0]), eval(line.split(',')[1])])

            labels = np.array(labels)
            print('Done')

            print('Labeling...', end='')
            dyn = []
            period = []
            segments = []

            for (start, end) in labels:
                start_id = np.searchsorted(self.timestamps, start)
                end_id = np.searchsorted(self.timestamps, end)
                dyn.extend(list(range(start_id, end_id)))
                period.append([self.timestamps[start_id], self.timestamps[end_id]])
                segments.append(list(range(start_id, end_id)))

            # static = list(set(full).difference(set(dyn)))
            self.labels = {'dynamic': dyn,
                           'times': period,
                           'segments': segments
                           }
        print('Done')

    def slice_by_label(self, segment='all', overwrite=True):
        """
        Slice the full-length CSI by loaded labels.
        :param segment:
        :param overwrite:
        :return:
        """
        print('Slicing...', end='')
        if self.labels is not None:
            if segment == 'all':
                seg = self.labels['dynamic'][:]
            else:
                seg = []
                for s in segment:
                    seg.extend(self.labels['segments'][s])

            if overwrite is True:
                self.csi = self.csi[seg]
                self.timestamps = self.timestamps[seg]
                self.length = len(self.csi)
            else:
                return self.csi[seg], self.timestamps[seg]
        print('Done')

    def load_lists(self, csilist=None, timelist=None):
        """
        Loads separate items into current MyCsi instance.\n
        :param csilist: input csi
        :param timelist: input timestamps
        :return: amplitude, phase, timestamps, length, sampling_rate
        """

        if self.path is not None:
            dic = np.load(self.path, allow_pickle=True).item()
            self.csi = dic['csi']
            self.timestamps = dic['time']
        else:
            self.csi = csilist
            self.timestamps = timelist
        self.length = len(self.csi)
        self.actual_sr = self.length / self.timestamps[-1]

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

            if input_path[-3:] != "npy":
                raise DataError("file: " + str(input_path))

        except PathError as e:
            print(e, "\nPlease check the path.")
        except DataError as e:
            print(e, "\nFile not supported. Please input .npz")

        else:
            csi_spectrum = np.load(input_path)
            self.spectrum = csi_spectrum['csi_spectrum']
            self.algorithm = csi_spectrum['csi_algorithm']
            print(self.name, "spectrum load complete", time.asctime(time.localtime(time.time())))

    def save_spectrum(self, notion=''):
        """
        Saves spectrum as npz.\n
        :param notion: additional information in the savename, default is empty
        :return: save_name + '-spectrum.npz'
        """
        print(self.name, "spectrum save start...", time.asctime(time.localtime(time.time())))

        try:
            if self.spectrum is None:
                raise DataError("spectrum")

        except DataError as e:
            print(e, "\nPlease compute spectrum")

        else:
            save_path = "../npsave/" + self.name[:4] + '/spectrum/'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Keys: spectrum, algorithm
            data = {'csi_spectrum': self.spectrum,
                    'csi_algorithm': self.algorithm}
            np.save(save_path + self.name + self.algorithm + "-spectrum" + notion + ".npy", data)
            print(self.name, "spectrum save complete", time.asctime(time.localtime(time.time())))

    def save_csi(self):
        save_path = "../npsave/" + self.name[:4] + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save = {'csi': self.csi,
                'time': self.timestamps}

        np.save(save_path + self.name + "-csis" + ".npy", save)

    def show_antenna_strength(self):
        """
        Shows the average of absolute values of each antenna.\n
        :return: nrx * ntx matrix
        """

        try:
            if self.csi is None:
                raise DataError("csi")

        except DataError as e:
            print(e, "\nPlease load data")

        else:
            mean_abs = np.mean(np.abs(self.csi), axis=(0, 1))
            return mean_abs

    def view_phase_diff(self, packet1=None, packet2=None, autosave=False, notion='', folder_name=''):

        if packet1 is None:
            packet1 = np.random.randint(self.length)
            print(packet1)
        if packet2 is None:
            packet2 = np.random.randint(self.length)
            print(packet2)

        plt.title("Phase Difference of " + str(self.name))
        plt.plot(np.unwrap(np.angle(self.csi[packet1, :, 0] * self.csi[packet1, :, 1].conj())),
                 label='antenna 0-1 #' + str(packet1), color='b')
        plt.plot(np.unwrap(np.angle(self.csi[packet2, :, 1] * self.csi[packet2, :, 2].conj())),
                 label='antenna 1-2 #' + str(packet1), color='r')
        plt.plot(np.unwrap(np.angle(self.csi[packet1, :, 0] * self.csi[packet1, :, 1].conj())),
                 label='antenna 0-1 #' + str(packet2), color='b',
                 linestyle='--')
        plt.plot(np.unwrap(np.angle(self.csi[packet2, :, 1] * self.csi[packet2, :, 2].conj())),
                 label='antenna 1-2 #' + str(packet2), color='r',
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

    def windowed_phase_difference(self, window_length=100, stride=100, folder_name=''):

        print(self.name, "Plotting phase difference...\n",)

        save_path = "../visualization/" + self.name[:4] + '/' + self.name[4:7] + folder_name + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for step in range((self.length - window_length) // stride):
            print('\rSaving figure', step, end='')
            plt.figure(figsize=(12, 9))

            ax1 = plt.subplot(2, 1, 1)
            ax1.set_title('Phasediff_0_1')
            ax1.plot(np.unwrap(np.angle(self.csi[step * stride: step * stride + window_length, 14, 0, 0] *
                                        self.csi[step * stride: step * stride + window_length, 14, 1, 0].conj())))
            ax1.set_xlabel('#Packet')
            ax1.set_ylim([-30, 30])

            ax2 = plt.subplot(2, 1, 2)
            ax2.set_title('Phasediff_1_2')
            ax2.plot(np.unwrap(np.angle(self.csi[step * stride: step * stride + window_length, 14, 1, 0] *
                                        self.csi[step * stride: step * stride + window_length, 14, 2, 0].conj())))
            ax2.set_xlabel('#Packet')
            ax2.set_ylim([-30, 30])
            plt.suptitle(str(self.name) + ' Phase Difference - packet ' + str(step * stride))
            plt.tight_layout()

            save_name = save_path + str(self.timestamps[step * stride]) + '.jpg'
            plt.savefig(save_name)
            plt.close()

        print('Done')

    def verbose_packet(self, index=None, notion=''):
        """
        Plot the unwrapped phase of a certain packet.
        :param index: packet index
        :param notion: Figure title
        :return: Figure
        """
        nsub = self.configs.nsub
        nrx = self.configs.nrx
        ntx = self.configs.ntx

        if index is None:
            index = np.random.randint(self.length)
        _csi = self.csi[index].reshape(nsub, ntx * nrx)
        _phs = np.unwrap(np.angle(_csi), axis=0)
        for ll in range(9):
            plt.plot(_phs[:, ll], label=ll)
        plt.legend()
        plt.grid()
        plt.title(notion)
        plt.xlabel("sub")
        plt.ylabel("phase")
        plt.show()

    def verbose_series(self, start=0, end=None, sub=14, rx=0, tx=0, notion=''):

        if end is None:
            end = self.length

        plt.plot(np.unwrap(np.angle(self.csi[start:end, sub, rx, tx]), axis=0))
        plt.grid()
        plt.title(notion)
        plt.xlabel("packet")
        plt.ylabel("phase")
        plt.show()

    def view_all_rx(self, metric="amplitude"):
        """
        Plots csi amplitude OR phase for all antennas.\n
        :param metric: 'amplitude' or 'phase'
        :return: value-time plot
        """
        print(self.name, metric, "plotting...", time.asctime(time.localtime(time.time())))

        try:
            if metric == "amplitude":
                csi_matrix = np.abs(self.csi)

            elif metric == "phase":
                csi_matrix = np.angle(self.csi)

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

    def aoa_by_music(self, input_theta_list=np.arange(-90, 91, 1.), smooth=False, pick_tx=0):
        """
        Computes AoA spectrum by MUSIC.\n
        :param input_theta_list: list of angels, default = -90~90
        :param smooth: whether apply SpotFi smoothing or not, default = False
        :param pick_tx: select 1 tx antenna, default is 0
        :return: AoA spectrum by MUSIC stored in self.data.spectrum
        """
        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        dist_antenna = self.configs.dist_antenna
        torad = self.configs.torad
        subfreq_list = self.configs.subfreq_list
        smoothing = self.commonfunc.smooth_csi
        noise = self.commonfunc.noise_space

        print(self.name, "AoA by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("csi: " + str(self.csi) + "\nPlease load data")

            if smooth is True:
                print(self.name, "apply Smoothing via SpotFi...")

            antenna_list = self.configs.antenna_list
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            spectrum = np.zeros((len(input_theta_list), self.length))

            for i in range(self.length):

                if smooth is True:
                    pass

                noise_space = noise(self.csi[i, :, :, pick_tx])

                if smooth is True:
                    steering_vector = np.exp([-1.j * 2 * np.pi * dist_antenna * (np.sin(theta_list * torad) *
                                              no_antenna).dot(sub_freq) / lightspeed
                                              for no_antenna in antenna_list[:2]
                                              for sub_freq in subfreq_list[:15]])
                else:
                    steering_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                                                antenna_list.T) * center_freq / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = AoAViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print(self.name, "AoA by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def aod_by_music(self, input_theta_list=np.arange(-90, 91, 1.), pick_rx=0):
        """
        Computes AoA spectrum by MUSIC.\n
        :param input_theta_list: list of angels, default = -90~90
        :param pick_rx: select 1 tx antenna, default is 0
        :return: AoA spectrum by MUSIC stored in self.data.spectrum
        """
        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        dist_antenna = self.configs.dist_antenna
        torad = self.configs.torad
        noise = self.commonfunc.noise_space

        print(self.name, "AoD by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("csi: " + str(self.csi) + "\nPlease load data")

            antenna_list = self.configs.antenna_list
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            spectrum = np.zeros((len(input_theta_list), self.length))

            for i in range(self.length):

                noise_space = noise(self.csi[i, :, pick_rx, :])

                steering_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                                            antenna_list.T) * center_freq / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = AoDViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print(self.name, "AoD by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def tof_by_music(self, input_dt_list=np.arange(-1.e-7, 4.e-7, 1.e-9), pick_tx=0):
        """
        Computes AoA spectrum by MUSIC.\n
        :param input_dt_list: list of tofs, default = -0.5e-7~2e-7
        :param pick_tx: select 1 tx antenna, default is 0
        :return: ToF spectrum by MUSIC stored in self.data.spectrum
        """

        subfreq_list = self.configs.subfreq_list - self.configs.center_freq
        noise = self.commonfunc.noise_space

        print(self.name, "ToF by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("amplitude: " + str(self.csi) + "\nPlease load data")

            dt_list = np.array(input_dt_list[::-1]).reshape(-1, 1)
            spectrum = np.zeros((len(input_dt_list), self.length))

            for i in range(self.length):

                noise_space = noise(self.csi[i, :, :, pick_tx].T)

                steering_vector = np.exp(-1.j * 2 * np.pi * dt_list.dot(subfreq_list.T))

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = ToFViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print(self.name, "ToF by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def doppler_by_music(self, input_velocity_list=np.arange(-5, 5.01, 0.01),
                         window_length=100,
                         stride=100,
                         pick_rx=0,
                         pick_tx=0,
                         ref_antenna=1,
                         raw_timestamps=False,
                         raw_window=False):
        """
        Computes Doppler spectrum by MUSIC.\n
        Involves self-calibration, windowed dynamic component extraction and resampling (if specified).\n
        :param input_velocity_list: list of velocities. Default = -5~5
        :param window_length: window length for each step
        :param stride: stride for each step
        :param pick_rx: select 1 rx antenna, default is 0. (You can also Specify 'strong' or 'weak')
        :param pick_tx: select 1 tx antenna, default is 0
        :param ref_antenna: select 2 rx antenna for dynamic extraction, default is 1
        :param raw_timestamps: whether to use original timestamps. Default is False
        :param raw_window: whether to use raw CSI or dynamic CSI. Default is False
        :return: Doppler spectrum by MUSIC stored in self.data.spectrum
        """
        sampling_rate = self.configs.sampling_rate
        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        noise = self.commonfunc.noise_space
        dynamic = self.commonfunc.dynamic

        print(self.name, "Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("amplitude: " + str(self.csi) + "\nPlease load data")

            # Each window has (window_length / sampling_rate) seconds of packets
            delay_list = np.arange(0, window_length, 1.).reshape(-1, 1) / sampling_rate
            velocity_list = np.array(input_velocity_list[::-1]).reshape(-1, 1)
            total_strides = (self.length - window_length) // stride

            if pick_rx == 'strong':
                pick_rx = np.argmax(self.show_antenna_strength())
            elif pick_rx == 'weak':
                pick_rx = np.argmin(self.show_antenna_strength())

            spectrum = np.zeros((len(input_velocity_list), total_strides))
            temp_timestamps = np.zeros(total_strides)

            for i in range(total_strides):

                csi_windowed = self.csi[i * stride: i * stride + window_length]

                if raw_window is True:
                    noise_space = noise(csi_windowed[:, :, pick_rx, pick_tx].T)
                else:
                    # Using windowed dynamic extraction
                    csi_dynamic = dynamic(csi_windowed, ref='rx', reference_antenna=ref_antenna)
                    noise_space = noise(csi_dynamic[:, :, pick_rx, pick_tx].T)

                if raw_timestamps is True:
                    # Using original timestamps (possibly uneven intervals)
                    delay_list = self.timestamps[i * stride: i * stride + window_length] - \
                                 self.timestamps[i * stride]

                steering_vector = np.exp(-1.j * 2 * np.pi * center_freq * velocity_list.dot(delay_list.T) / lightspeed)

                a_en = steering_vector.conj().dot(noise_space)
                spectrum[:, i] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

                temp_timestamps[i] = self.timestamps[i * stride]

            self.spectrum = np.log(spectrum)
            self.viewer = DopplerViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps,
                                        xlabels=temp_timestamps)

            print(self.name, "Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)

    def aoa_tof_by_music(self, input_theta_list=np.arange(-90, 91, 1.),
                         input_dt_list=np.arange(-1.e-7, 2.e-7, 1.e-9),
                         smooth=False):
        """
        Computes AoA-ToF spectrum by MUSIC.\n
        :param input_theta_list:  list of angels, default = -90~90
        :param input_dt_list: list of time measurements, default = 0~8e-8
        :param smooth:  whether apply SpotFi smoothing or not, default = False
        :return:  AoA-ToF spectrum by MUSIC stored in self.data.spectrum
        """

        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        dist_antenna = self.configs.dist_antenna
        torad = self.configs.torad
        subfreq_list = self.configs.subfreq_list
        nsub = self.configs.nsub
        nrx = self.configs.nrx
        smoothing = self.commonfunc.smooth_csi
        noise = self.commonfunc.noise_space

        print(self.name, "AoA-ToF by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("amplitude: " + str(self.csi) + "\nPlease load data")

            if smooth not in (True, False):
                raise ArgError("smooth:" + str(smooth))

            if smooth is True:
                print(self.name, "apply Smoothing via SpotFi...")

            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            dt_list = np.array(input_dt_list).reshape(-1, 1)

            steering_aoa = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                            antenna_list.T) * center_freq / lightspeed).reshape(-1, 1)

            spectrum = np.zeros((self.length, len(input_theta_list), len(input_dt_list)))

            for i in range(self.length):

                if smooth is True:
                    pass

                noise_space = noise(self.csi[i].reshape(1, -1))   # nrx * nsub columns

                for j, tof in enumerate(dt_list):

                    if smooth is True:
                        steering_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                            antenna_list[:2].dot(subfreq_list[:15])) / lightspeed)
                    else:
                        steering_tof = np.exp(-1.j * 2 * np.pi * subfreq_list * tof).reshape(-1, 1)
                        steering_vector = steering_tof.dot(steering_aoa.T).reshape(nsub, len(input_theta_list), nrx)
                        steering_vector = steering_vector.swapaxes(0, 1).reshape(len(input_theta_list), nrx * nsub)

                    a_en = np.conjugate(steering_vector).dot(noise_space)
                    spectrum[i, :, j] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = AoAToFViewer(name=self.name, spectrum=self.spectrum, timestamps=self.timestamps)
            print(self.name, "AoA-ToF by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")

    def aoa_doppler_by_music(self, input_theta_list=np.arange(-90, 91, 1.),
                             input_velocity_list=np.arange(-5, 5.05, 0.05),
                             window_length=100,
                             stride=100,
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

        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq
        dist_antenna = self.configs.dist_antenna
        sampling_rate = self.configs.sampling_rate
        torad = self.configs.torad
        nrx = self.configs.nrx
        nsub = self.configs.nsub
        noise = self.commonfunc.noise_space
        dynamic = self.commonfunc.dynamic

        print(self.name, "AoA-Doppler by MUSIC - compute start...", time.asctime(time.localtime(time.time())))

        try:
            if self.csi is None:
                raise DataError("amplitude: " + str(self.csi) + "\nPlease load data")

            # Each window has ts of packets (1 / sampling_rate * window_length = t)
            delay_list = np.arange(0, window_length, 1.).reshape(-1, 1) / sampling_rate
            antenna_list = np.arange(0, nrx, 1.).reshape(-1, 1)
            theta_list = np.array(input_theta_list[::-1]).reshape(-1, 1)
            velocity_list = np.array(input_velocity_list).reshape(-1, 1)

            steering_aoa = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list * torad).dot(
                antenna_list.T) * center_freq / lightspeed).reshape(-1, 1)
            spectrum = np.zeros(((self.length - window_length) // stride, len(input_theta_list),
                                 len(input_velocity_list)))
            temp_timestamps = np.zeros((self.length - window_length) // stride)

            # Using windowed dynamic extraction
            for i in range((self.length - window_length) // stride):

                csi_windowed = self.csi[i * stride: i * stride + window_length]

                if raw_window is True:
                    noise_space = noise(csi_windowed.swapaxes(0, 1).reshape(nsub, window_length * nrx))
                else:
                    # Using windowed dynamic extraction
                    csi_dynamic = dynamic(csi_windowed, ref='rx', reference_antenna=2)
                    noise_space = noise(csi_dynamic.swapaxes(0, 1).reshape(nsub, window_length * nrx))

                if raw_timestamps is True:
                    # Using original timestamps (possibly uneven intervals)
                    delay_list = self.timestamps[i * stride: i * stride + window_length] - \
                                 self.timestamps[i * stride]

                for j, velocity in enumerate(velocity_list):

                    steering_doppler = np.exp(-1.j * 2 * np.pi * center_freq * delay_list * velocity /
                                              lightspeed).reshape(-1, 1)
                    steering_vector = steering_doppler.dot(steering_aoa.T
                                                           ).reshape(len(delay_list), len(input_theta_list), nrx)
                    steering_vector = steering_vector.swapaxes(0, 1
                                                               ).reshape(len(input_theta_list), nrx * len(delay_list))

                    a_en = np.conjugate(steering_vector).dot(noise_space)
                    spectrum[i, :, j] = 1. / np.absolute(np.diagonal(a_en.dot(a_en.conj().T)))

            self.spectrum = np.log(spectrum)
            self.viewer = AoADopplerViewer(name=self.name, spectrum=self.spectrum, timestamps=temp_timestamps)
            print(self.name, "AoA-Doppler by MUSIC - compute complete", time.asctime(time.localtime(time.time())))

        except DataError as e:
            print(e)
        except ArgError as e:
            print(e, "\nPlease specify smooth=True or False")

    def sanitize_phase(self):
        """
        Also known as SpotFi Algorithm1.\n
        Removes Sampling Time Offset shared by all rx antennas.\n
        :return: sanitized phase
        """

        nrx = self.configs.nrx
        nsub = self.configs.nsub

        print(self.name, "apply SpotFi Algorithm1 to remove STO...", end='')

        try:
            if self.csi is None:
                raise DataError("phase: " + str(self.csi))

            fit_x = np.concatenate([np.arange(0, nsub) for _ in range(nrx)])
            fit_y = np.unwrap(np.squeeze(self.csi), axis=1).swapaxes(1, 2).reshape(self.length, -1)

            a = np.stack((fit_x, np.ones_like(fit_x)), axis=-1)
            fit = np.linalg.inv(a.T.dot(a)).dot(a.T).dot(fit_y.T).T
            # fit = np.array([np.polyfit(fit_x, fit_y[i], 1) for i in range(self.data.length)])

            phase = np.unwrap(np.angle(self.csi), axis=1) - np.arange(nsub).reshape(
                (1, nsub, 1, 1)) * fit[:, 0].reshape(self.length, 1, 1, 1)
            print("Done")

            self.csi = np.abs(self.csi) * np.exp(1.j * phase)

        except DataError as e:
            print(e, "\nPlease load data")

    def remove_ipo(self, reference_antenna=0, cal_dict=None):
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
        nrx = self.configs.nrx
        distance_antenna = self.configs.dist_antenna
        torad = self.configs.torad
        lightspeed = self.configs.lightspeed
        center_freq = self.configs.center_freq

        print(self.name, "apply phase calibration according to", str(cal_dict.keys())[10:-1], "...", end='')

        try:
            if self.csi is None:
                raise DataError("csi: " + str(self.csi))

            if reference_antenna not in (0, 1, 2):
                raise ArgError("reference_antenna: " + str(reference_antenna))

            if cal_dict is None:
                raise DataError("reference: " + str(cal_dict))

            ipo = []
            # cal_dict: "{'xx': MyCsi}"

            for key, value in cal_dict.items():

                if not isinstance(value, MyCsi):
                    raise DataError("reference csi: " + str(value) + "\nPlease input MyCsi instance.")

                if value.csi is None:
                    raise DataError("reference phase: " + str(value.csi))

                ref_angle = eval(key)

                ref_csi = value.csi
                ref_diff = np.mean(ref_csi * ref_csi[:, :, reference_antenna][:, :, np.newaxis].conj(),
                                   axis=(0, 1))
                true_diff = np.exp([-1.j * 2 * np.pi * distance_antenna * antenna * center_freq * np.sin(
                    ref_angle * torad) / lightspeed for antenna in range(nrx)]).reshape(-1, 1)

                ipo.append(ref_diff.reshape(-1, 1) * true_diff.conj())

            ipo = np.squeeze(np.mean(ipo, axis=0))

            self.csi = self.csi * ipo[np.newaxis, np.newaxis, :, np.newaxis].conj()

            print("Done")

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify an integer from 0~2")

    def remove_csd(self, HT=False):
        """
        Remove CSD based on values in 802.11 standard.\n
        Requires 3 tx.\n
        non-HT: -200ns, -100ns\n
        HT: -400ns, -200ns\n
        :param HT: Default is False
        :return: CSI with CSD removed
        """

        print(self.name, "removing CSD...", end='')

        try:
            if self.csi is None:
                raise DataError("csi: " + str(self.csi))

            if self.configs.ntx != 3:
                raise DataError(str(self.csi) + 'does not have multiple tx')
            else:
                if HT:
                    csd_1 = np.exp(2.j * np.pi * self.configs.subfreq_list * (-400) * 1.e-9)
                    csd_2 = np.exp(2.j * np.pi * self.configs.subfreq_list * (-200) * 1.e-9)
                else:
                    csd_1 = np.exp(2.j * np.pi * self.configs.subfreq_list * (-200) * 1.e-9)
                    csd_2 = np.exp(2.j * np.pi * self.configs.subfreq_list * (-100) * 1.e-9)

            self.csi[:, :, :, 1] = self.csi[:, :, :, 1] * csd_1
            self.csi[:, :, :, 2] = self.csi[:, :, :, 2] * csd_2

            print("Done")

        except DataError as e:
            print(e, "\nPlease load data")

    def show_csd(self):
        if self.configs.ntx != 3:
            return
        else:
            csd_1 = self.csi[..., 0] * self.csi[..., 1].conj()
            csd_2 = self.csi[..., 0] * self.csi[..., 2].conj()

            csd_1 = np.unwrap(np.squeeze(np.angle(np.mean(csd_1, axis=0)))) / (2 * np.pi * self.configs.subfreq_list) * 1.e9
            csd_2 = np.unwrap(np.squeeze(np.angle(np.mean(csd_2, axis=0)))) / (2 * np.pi * self.configs.subfreq_list) * 1.e9

            plt.subplot(2, 1, 1)

            for rx in range(self.configs.nrx):
                plt.plot(csd_1[:, rx], label='rx'+str(rx))
            plt.xlabel('Sub')
            plt.ylabel('CSD/ns')
            plt.title('CSD_1')
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            for rx in range(self.configs.nrx):
                plt.plot(csd_2[:, rx], label='rx' + str(rx))
            plt.xlabel('Sub')
            plt.ylabel('CSD/ns')
            plt.title('CSD_2')
            plt.legend()
            plt.grid()

            plt.suptitle('CSD')
            plt.tight_layout()
            plt.show()

    def extract_dynamic(self, mode='overall-multiply',
                        ref='rx',
                        reference_antenna=0,
                        window_length=100,
                        stride=100,
                        **kwargs):
        """
        Removes the static component from csi.\n
        :param mode: 'overall' or 'running' (in terms of averaging) or 'highpass'. Default is 'overall'
        :param ref: 'rx' or 'tx'
        :param window_length: if mode is 'running', specify a window length for running mean. Default is 100
        :param stride: if mode is 'running', specify a stride for running mean. Default is 100
        :param reference_antenna: select one antenna with which to remove random phase offsets. Default is 0
        :return: phase and amplitude of dynamic component of csi
        """
        nrx = self.configs.nrx
        nsub = self.configs.nsub
        ntx = self.configs.ntx
        sampling_rate = self.configs.sampling_rate
        dynamic = self.commonfunc.dynamic
        division = self.commonfunc.windowed_divison
        highpass = self.commonfunc.highpass

        print(self.name, "apply dynamic component extraction...", end='')

        try:
            if self.csi is None:
                raise DataError("csi data")

            if reference_antenna not in range(nrx):
                raise ArgError("reference_antenna: " + str(reference_antenna) + "\nPlease specify an integer from 0~2")

            if reference_antenna is None:
                strengths = self.show_antenna_strength()
                reference_antenna = np.argmax(strengths)

            if mode == 'overall-multiply':
                if ref == 'rx':
                    conjugate_csi = self.csi[..., reference_antenna, :][..., np.newaxis, :].repeat(nrx, axis=2).conj()
                elif ref == 'tx':
                    conjugate_csi = self.csi[..., reference_antenna][..., np.newaxis].repeat(ntx, axis=3).conj()
                hc = (self.csi * conjugate_csi).reshape((-1, nsub, nrx, ntx))
                average_hc = np.mean(hc, axis=0)[np.newaxis, ...].repeat(self.length, axis=0)
                dynamic_csi = hc - average_hc

            elif mode == 'overall-divide':
                re_csi = (np.abs(self.csi) + 1.e-6) * np.exp(1.j * np.abs(self.csi))
                if ref == 'rx':
                    dynamic_csi = self.csi / re_csi[..., reference_antenna, :][..., np.newaxis, :].repeat(nrx, axis=2)
                elif ref == 'tx':
                    dynamic_csi = self.csi / re_csi[..., reference_antenna][..., np.newaxis].repeat(ntx, axis=3)

            elif mode == 'running-multiply':
                dynamic_csi = np.zeros((self.length, self.configs.nsub, self.configs.nrx, self.configs.ntx), dtype=complex)
                for step in range((self.length - window_length) // stride):
                    dynamic_csi[step * stride: step * stride + window_length] = dynamic(
                        self.csi[step * stride: step * stride + window_length], ref, reference_antenna, **kwargs)

            elif mode == 'running-divide':
                dynamic_csi = np.zeros((self.length, self.configs.nsub, self.configs.nrx, self.configs.ntx), dtype=complex)
                for step in range((self.length - window_length) // stride):
                    dynamic_csi[step * stride: step * stride + window_length] = division(
                        self.csi[step * stride: step * stride + window_length], ref, reference_antenna, **kwargs)

            elif mode == 'highpass':
                b, a = highpass(**kwargs)
                dynamic_csi = np.zeros_like(self.csi)
                for sub in range(nsub):
                    for rx in range(nrx):
                        for tx in range(ntx):
                            dynamic_csi[:, sub, rx, tx] = signal.filtfilt(b, a, self.csi[:, sub, rx, tx])

            else:
                raise ArgError("mode: " + str(mode) +
                               "\nPlease specify mode=\"overall-multiply\", \"overall-divide\", \"running-divide\"or "
                               "\"highpass\"")

            self.csi = dynamic_csi
            print("Done")

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
        print(self.name, "resampling at " + str(sampling_rate) + "Hz...", end='')

        try:
            if self.csi is None:
                raise DataError("csi data")

            if not isinstance(sampling_rate, int) or sampling_rate >= self.actual_sr:
                raise ArgError("sampling_rate: " + str(sampling_rate))

            new_interval = 1. / sampling_rate

            new_length = int(self.timestamps[-1] * sampling_rate) + 1  # Flooring
            resample_indicies = []

            for i in range(new_length):

                index = np.searchsorted(self.timestamps, i * new_interval)

                if index > 0 and (
                        index == self.length or
                        abs(self.timestamps[index] - i * new_interval) >
                        abs(self.timestamps[index - 1] - i * new_interval)):
                    index -= 1

                resample_indicies.append(index)

            self.csi = self.csi[resample_indicies]
            self.timestamps = self.timestamps[resample_indicies]
            self.length = new_length
            self.actual_sr = sampling_rate

            print("Done")

        except DataError as e:
            print(e, "\nPlease load data")
        except ArgError as e:
            print(e, "\nPlease specify an integer less than the current sampling rate")


if __name__ == '__main__':

    mycon = MyConfigs(5.32, 20)
    mycon.ntx = 3
    mycsi = MyCsi(mycon, '0307A04', '../npsave/0307/0307A04-csio.npy')
    mycsi.load_data(remove_sm=True)
    #mycsi.load_lists()
    mycsi.load_label('../sense/0307/04_labels.csv')
    mycsi.slice_by_label(overwrite=True)
    #mycsi.show_csd()
    mycsi.remove_csd()
    mycsi.extract_dynamic(mode='overall-divide', ref='tx', reference_antenna=1, subtract_mean=False)
    mycsi.extract_dynamic(mode='highpass')
    #mycsi.calibrate_phase(reference_antenna=0, cal_dict={'0': ref})
    #mycsi.windowed_phase_difference(folder_name='phasediff_dyn')
    mycsi.aoa_by_music()
    mycsi.viewer.view()
