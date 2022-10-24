import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pycsi


class CountClass(object):
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print("*" * 20 + str(self.func.__name__) + ": run #" + str(self.count) + "*" * 20)

        return self.func(*args, **kwargs)


class MyFunc(object):

    def __init__(self, data_date=None, test_title=None, rearrange=False, reference=None, subject=None):

        self.date = data_date
        self.title = str(test_title)
        self.rearrange = rearrange
        self.calibrate = False
        self.extract = False
        self.sanitize = False
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
        if self.rearrange is True:
            print("  Apply antenna order rearrangement...", time.asctime(time.localtime(time.time())))
            self.subject.data.rearrange_antenna()

            for value in self.reference.values():
                value.data.rearrange_antenna()

        if self.calibrate is True:
            self.subject.calibrate_phase(cal_dict=self.reference)

        if self.sanitize is True:
            self.subject.sanitize_phase()

        if self.extract is True:
            self.subject.extract_dynamic()

        if self.resample is False:
            self.sampling_rate = 0

        if self.resample is True and 0 < self.sampling_rate < 3965:

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
            save_path = os.getcwd().replace('\\', '/') + "/visualization/" + self.subject.name[
                                                                             :4] + '/' + self.title + '/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_name = save_path + self.subject.name[4:] + self.notion + '.png'
            plt.savefig(save_name)
            print(self.subject.name, "saved as", save_name, time.asctime(time.localtime(time.time())))
            plt.close()
            return save_name

        else:
            plt.show()
            return 'No saving'


@CountClass
class _TestPhaseDiff(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.ref_antenna = np.argmax(self.subject.data.show_antenna_strength())
        self.antennas = list(range(self.subject.nrx))
        self.antennas.remove(int(self.ref_antenna))
        self.packet1 = np.random.randint(self.subject.data.length)
        self.packet2 = np.random.randint(self.subject.data.length)
        self.title1 = 'Before'
        self.title2 = 'After'
        self.suptitle = 'Calibration of ' + self.subject.name

    def __str__(self):
        return 'Test Phase Difference After Calibration'

    def mysubplot(self, axis, title, phase):

        axis.set_title(title)
        axis.plot(phase[self.packet1, :, self.antennas[0], 0],
                  label='antenna' + str(self.antennas[0]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet1),
                  color='b')
        axis.plot(phase[self.packet1, :, self.antennas[1], 0],
                  label='antenna' + str(self.antennas[1]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet1),
                  color='r')
        axis.plot(phase[self.packet2, :, self.antennas[0], 0],
                  label='antenna' + str(self.antennas[0]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet2),
                  color='b',
                  linestyle='--')
        axis.plot(phase[self.packet2, :, self.antennas[1], 0],
                  label='antenna' + str(self.antennas[1]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet2),
                  color='r',
                  linestyle='--')
        axis.set_xlabel('#Subcarrier', loc='right')
        axis.set_ylabel('Phase Difference / $rad$')
        axis.legend()

    def func(self):

        self.preprocess()

        csi = self.subject.data.amp * np.exp(1.j * self.subject.data.phase)
        phase_diff = np.angle(csi * csi[:, :, self.ref_antenna, :][:, :, np.newaxis, :].conj())

        print(self.subject.name, "test_phase_diff plotting...", time.asctime(time.localtime(time.time())))

        fig, ax = plt.subplots(2, 1)
        self.mysubplot(ax[0], self.title1, phase_diff)

        self.subject.calibrate_phase(self.ref_antenna, self.reference)

        csi = self.subject.data.amp * np.exp(1.j * self.subject.data.phase)
        phase_diff = np.angle(csi * csi[:, :, self.ref_antenna, :][:, :, np.newaxis, :].conj())

        self.mysubplot(ax[1], self.title2, phase_diff)
        print(self.subject.name, "test_phase_diff plot complete", time.asctime(time.localtime(time.time())))

        return self.save_show_figure()


@CountClass
class _TestResampling(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

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
class _TestSanitize(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.packet1 = np.random.randint(self.subject.data.length)
        self.packet2 = np.random.randint(self.subject.data.length)
        self.suptitle = 'Sanitization of ' + self.subject.name

    def __str__(self):
        return 'Test Sanitization (SpotFi Algorithm1)'

    def mysubplot(self, axis, title, phase):
        axis.set_title(title)
        axis.plot(np.unwrap(np.squeeze(phase[self.subject, :, 0, :])), label='antenna0 #' + str(self.packet1),
                  color='b')
        axis.plot(np.unwrap(np.squeeze(phase[self.subject, :, 1, :])), label='antenna1 #' + str(self.packet1),
                  color='r')
        axis.plot(np.unwrap(np.squeeze(phase[self.subject, :, 2, :])), label='antenna2 #' + str(self.packet1),
                  color='y')
        axis.plot(np.unwrap(np.squeeze(phase[self.packet2, :, 0, :])), label='antenna0 #' + str(self.packet2),
                  color='b', linestyle='--')
        axis.plot(np.unwrap(np.squeeze(phase[self.packet2, :, 1, :])), label='antenna1 #' + str(self.packet2),
                  color='r', linestyle='--')
        axis.plot(np.unwrap(np.squeeze(phase[self.packet2, :, 2, :])), label='antenna2 #' + str(self.packet2),
                  color='y', linestyle='--')
        axis.set_xlabel('Subcarrier', loc='right')
        axis.set_ylabel('Phase Difference')
        axis.legend()

    def func(self):

        self.preprocess()

        # self.subject.data.phase -= np.mean(self.subject.data.phase, axis=1).reshape(-1, 1, 3, 1)

        print(self.subject.name, "test_sanitization plotting...", time.asctime(time.localtime(time.time())))

        fig, ax = plt.subplots(2, 1)
        self.mysubplot(ax[0], "Before Sanitization", self.subject.data.phase)

        self.subject.sanitize_phase()
        # csi.data.phase -= np.mean(csi.data.phase, axis=1).reshape(-1, 1, 3, 1)

        self.mysubplot(ax[1], "After Sanitization", self.subject.data.phase)
        print(self.subject.name, "test_sanitization plot complete", time.asctime(time.localtime(time.time())))

        return self.save_show_figure()


@CountClass
class _TestAoA(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.threshold = 0
        self.calibrate = True
        self.num_ticks = 11

    def __str__(self):
        return 'Plot AoA Spectrum'

    def func(self):

        self.preprocess()
        self.subject.aoa_by_music()

        return self.subject.data.view_spectrum(self.threshold, self.num_ticks, self.autosave, self.notion)


@CountClass
class _TestDoppler(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.threshold = 0
        self.window_length = 500
        self.stride = 500
        self.num_ticks = 11

    def __str__(self):
        return 'Plot Doppler Spectrum'

    def func(self):

        self.preprocess()

        self.subject.doppler_by_music(resample=self.sampling_rate, window_length=self.window_length, stride=self.stride)

        return self.subject.data.view_spectrum(self.threshold, self.num_ticks, self.autosave, self.notion)


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

        for i, spectrum in enumerate(self.subject.data.spectrum):
            return self.subject.data.view_spectrum(self.threshold, spectrum, self.num_ticks, self.autosave,
                                                   self.notion + '_' + str(i))


@CountClass
class _TestAoADoppler(MyFunc):

    # Not finished

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.threshold = 0
        self.start = 0
        self.end = self.subject.data.length
        self.extract = True
        self.resample = True
        self.sampling_rate = 100
        self.num_ticks = 11
        self.self_cal = True

    def __str__(self):
        return 'Plot AoA-Doppler Spectrum'

    def func(self):

        self.preprocess()

        if 0 <= self.start <= self.end <= self.subject.data.length:
            self.subject.data.length = self.end - self.start
            self.subject.data.amp = self.subject.data.amp[self.start: self.end]
            self.subject.data.phase = self.subject.data.phase[self.start: self.end]

        self.subject.aoa_doppler_by_music(self_cal=self.self_cal)

        for spectrum in self.subject.spectrum.length:
            return self.subject.data.view_spectrum(self.threshold, spectrum, self.num_ticks, self.autosave, self.notion)

