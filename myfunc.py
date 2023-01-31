import numpy as np
import time
import os
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
import seaborn as sns


class CountClass(object):
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print("*" * 20 + str(self.func.__name__) + ": run #" + str(self.count) + "*" * 20)

        return self.func(*args, **kwargs)


class MyFunc(object):

    def __init__(self, data_date=None, test_title=None, reference=None, subject=None):

        self.date = data_date
        self.title = str(test_title)
        self.self_cal = False
        self.calibrate = False
        self.sanitize = False
        self.extract = False
        self.resample = False
        self.sampling_rate = 0
        self.reference = reference
        self.subject = subject
        self.autosave = False
        self.notion = ''
        self.suptitle = ''

    def __str__(self):
        return 'My Test Functions'

    def preprocess(self):

        if self.self_cal is True:
            self.subject.self_calibrate()

        if self.calibrate is True:
            self.subject.calibrate_phase(cal_dict=self.reference)

        if self.sanitize is True:
            self.subject.sanitize_phase()

        if self.extract is True:
            self.subject.extract_dynamic()

        if self.resample is False:
            self.sampling_rate = 0

        if self.resample is True and self.sampling_rate > 0:

            if self.subject.resample_packets(sampling_rate=self.sampling_rate) == 'bad':
                return 'Subject skipped due to resampling error'

    def func(self):
        pass

    def myplot(self, **kwargs):
        pass

    def mysubplot(self, **kwargs):
        pass

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save_show_figure(self):
        plt.suptitle(self.suptitle)

        if self.autosave is True:
            save_path = "../visualization/" + self.subject.name[:4] + '/' + self.title + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = save_path + self.subject.name[4:] + self.notion + '.png'
            plt.savefig(save_name)
            print(self.subject.name, "saved as", save_name, time.asctime(time.localtime(time.time())))
            plt.close()
            return save_name

        else:
            plt.show()
            return 'No saving'


class PhaseCompare(MyFunc):
    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)
        self.ref_antenna = 1
        self.antennas = list(range(self.subject.nrx))
        self.recursive = False
        self.packet1 = int(self.subject.data.length / 2)
        self.packet2 = int(self.subject.data.length / 2) + 1
        self.method = 'calibration'
        self.title1 = 'Before'
        self.title2 = 'After'
        self.suptitle = self.subject.name

    def __str__(self):
        return 'Phase Comparison Method'

    def get_phase(self):
        return 0

    def antenna_list(self):
        pass

    def func(self):

        self.title1 = 'Before ' + self.method.title()
        self.title2 = 'After ' + self.method.title()
        self.antenna_list()
        self.preprocess()

        count = self.subject.data.length if self.recursive is True else 1

        for i in range(count):

            if count == 1:
                self.packet1 = int(self.subject.data.length / 2)
                self.packet2 = int(self.subject.data.length / 2) + 10
            else:
                self.suptitle += "_" + str(i)
                self.packet1 = i
                self.packet2 = i

            print(self.subject.name, i, "of", self.subject.data.length, "plotting...",
                  time.asctime(time.localtime(time.time())))

            phase = self.get_phase()

            fig, ax = plt.subplots(2, 1)
            self.mysubplot(ax[0], self.title1, phase)

            if self.method == 'sanitization':
                self.subject.sanitize_phase()

            elif self.method == 'calibration':
                self.subject.calibrate_phase(self.ref_antenna, self.reference)

            elif self.method == 'calibration + sanitization':
                self.subject.calibrate_phase(self.ref_antenna, self.reference)
                self.subject.sanitize_phase()

            phase = self.get_phase()

            self.mysubplot(ax[1], self.title2, phase)
            print(self.subject.name, i, "of", self.subject.data.length, "plot complete",
                  time.asctime(time.localtime(time.time())))

            r = self.save_show_figure()

        return r


@CountClass
class _TestPhase(PhaseCompare):

    def __str__(self):
        return 'Test Phase'

    def mysubplot(self, axis, title, phase):

        axis.set_title(title)
        axis.plot(phase[self.packet1, :, 0, 0], label='antenna0 #' + str(self.packet1), color='b')
        axis.plot(phase[self.packet1, :, 1, 0], label='antenna1 #' + str(self.packet1), color='r')
        axis.plot(phase[self.packet1, :, 2, 0], label='antenna2 #' + str(self.packet1), color='y')
        axis.plot(phase[self.packet2, :, 0, 0], label='antenna0 #' + str(self.packet2), color='b', linestyle='--')
        axis.plot(phase[self.packet2, :, 1, 0], label='antenna1 #' + str(self.packet2), color='r', linestyle='--')
        axis.plot(phase[self.packet2, :, 2, 0], label='antenna2 #' + str(self.packet2), color='y', linestyle='--')
        axis.set_xlabel('#Subcarrier', loc='right')
        axis.set_ylabel('Phase / $rad$')
        axis.legend()

    def get_phase(self):
        csi = self.subject.data.amp * np.exp(1.j * self.subject.data.phase)
        phase = np.unwrap(np.angle(csi), axis=1)
        return phase


@CountClass
class _TestPhaseDiff(PhaseCompare):

    def __str__(self):
        return 'Test Phase Difference'

    def mysubplot(self, axis, title, phase):

        axis.set_title(title)
        axis.plot(phase[0][self.packet1],
                  label='antenna 0-1 #' + str(self.packet1),
                  color='b')
        axis.plot(phase[1][self.packet1],
                  label='antenna 1-2 #' + str(self.packet1),
                  color='r')
        axis.plot(phase[0][self.packet2],
                  label='antenna 0-1 #' + str(self.packet2),
                  color='b',
                  linestyle='--')
        axis.plot(phase[1][self.packet2],
                  label='antenna 1-2 #' + str(self.packet2),
                  color='r',
                  linestyle='--')
        axis.set_xlabel('#Subcarrier', loc='right')
        axis.set_ylabel('Phase Difference / $rad$')
        axis.legend()

    def get_phase(self):
        csi = self.subject.data.amp * np.exp(1.j * self.subject.data.phase)
        phase_diff1 = np.unwrap(np.angle(csi[:, :, 0, :] * csi[:, :, 1, :].conj()), axis=1)
        phase_diff2 = np.unwrap(np.angle(csi[:, :, 1, :] * csi[:, :, 2, :].conj()), axis=1)
        print('0-1:', phase_diff1[self.packet1])
        print('1-2:', phase_diff2[self.packet1])
        return [phase_diff1, phase_diff2]

    def antenna_list(self):
        #self.antennas.remove(int(self.ref_antenna))
        #self.suptitle = self.subject.name + '_ref' + str(self.ref_antenna)
        pass


@CountClass
class _TestResampling(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.resample = True
        self.sampling_rate = 100
        self.antenna = 0
        self.subcarrier = 0
        self.label = 'sub' + str(self.antenna) + ' antenna' + str(self.antenna)
        self.suptitle = 'Resampling of ' + self.subject.name + ' @' + str(self.sampling_rate) + 'Hz'

    def __str__(self):
        return 'Test Resamping at Given Sampling Rate'

    def mysubplot(self, axis, title, amp):
        axis.set_title(title)
        axis.plot(np.squeeze(amp), label=self.label)
        axis.set_xlabel('Timestamp/s', loc='right')
        plt.sca(axis)
        labels1 = [i * self.subject.data.length // 10 for i in range(10)]
        labels1.append(self.subject.data.length - 1)
        l1 = [float('%.6f' % x) for x in self.subject.data.timestamps[labels1]]
        plt.xticks(labels1, l1)
        axis.set_ylabel('Amplitude')
        axis.legend()

    def func(self):

        self.preprocess()

        print('Length before resampling:', self.subject.data.length)
        print(self.subject.name, "test_resampling plotting...", time.asctime(time.localtime(time.time())))

        fig, ax = plt.subplots(2, 1)
        self.mysubplot(ax[0], "Before Resampling", self.subject.data.amp[:, self.subcarrier, self.antenna, 0])

        self.subject.resample_packets(self.sampling_rate)
        print('Length after resampling:', self.subject.data.length)

        self. mysubplot(ax[1], "After Resampling", self.subject.data.amp[:, self.subcarrier, self.antenna, 0])
        print(self.subject.name, "test_resampling plot complete", time.asctime(time.localtime(time.time())))

        return self.save_show_figure()


@CountClass
class _TestAoA(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.threshold = 0
        self.calibrate = True
        self.smooth = False
        self.num_ticks = 11

    def __str__(self):
        return 'Plot AoA Spectrum'

    def func(self):

        self.preprocess()
        self.subject.extract_dynamic()
        self.subject.aoa_by_music(smooth=self.smooth)

        return self.subject.viewer.view(threshold=self.threshold, autosave=self.autosave, notion=self.notion)


@CountClass
class _TestDoppler(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.resample = False
        self.sampling_rate = 100
        self.threshold = -4.4
        self.window_length = 100
        self.stride = 100
        self.num_ticks = 11

    def __str__(self):
        return 'Plot Doppler Spectrum'

    def func(self):

        self.preprocess()

        #self.subject.extract_dynamic()

        self.subject.doppler_by_music(window_length=self.window_length, stride=self.stride, raw_window=False)

        return self.subject.viewer.view(threshold=self.threshold, autosave=self.autosave, notion=self.notion)


@CountClass
class _TestAoAToF(MyFunc):

    # Not finished

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.threshold = 0
        self.start = 0
        self.end = self.subject.data.length
        self.calibrate = True
        self.sanitize = True
        self.extract = True
        self.resample = True
        self.sampling_rate = 100
        self.smooth = False
        self.num_ticks = 11

    def __str__(self):
        return 'Plot AoA-ToF Spectrum'

    def func(self):

        self.preprocess()

        if 0 <= self.start <= self.end <= self.subject.data.length:
            self.subject.data.length = self.end - self.start
            self.subject.data.amp = self.subject.data.amp[self.start: self.end]
            self.subject.data.phase = self.subject.data.phase[self.start: self.end]

        self.subject.aoa_tof_by_music()

        return_name = []

        for i, spectrum in enumerate(self.subject.data.spectrum):
            return_name = self.subject.data.view_spectrum(self.threshold, i, None, self.num_ticks, self.autosave,
                                                          self.notion + '_' + str(i).zfill(5), self.title)

        return return_name


@CountClass
class _TestAoADoppler(MyFunc):

    # Not finished

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.threshold = 0
        self.start = 0
        self.end = self.subject.data.length
        self.self_cal = True
        self.resample = False
        self.sampling_rate = 0
        self.num_ticks = 11

    def __str__(self):
        return 'Plot AoA-Doppler Spectrum'

    def func(self):

        self.preprocess()

        if 0 <= self.start <= self.end <= self.subject.data.length:
            self.subject.data.length = self.end - self.start
            self.subject.data.amp = self.subject.data.amp[self.start: self.end]
            self.subject.data.phase = self.subject.data.phase[self.start: self.end]

        self.subject.aoa_doppler_by_music()

        return_name = []

        for i, spectrum in enumerate(self.subject.data.spectrum):
            return_name = self.subject.data.view_spectrum(self.threshold, i, None, self.num_ticks, self.autosave,
                                                          self.notion + '_' + str(i).zfill(5), self.title)

        return return_name
