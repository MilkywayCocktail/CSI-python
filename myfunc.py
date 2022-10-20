import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pycsi
from functools import wraps


class MyFunc(object):

    def __init__(self, data_date=None, test_title=None, rearrange=False, reference=None, subject=None):

        self.date = data_date
        self.title = str(test_title)
        self.rearrange = rearrange
        self.reference = reference
        self.subject = subject

    def __str__(self):
        return 'My Test Functions'

    def func(self):
        pass

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _TestPhaseDiff(MyFunc):

    def __init__(self, *args, **kwargs):
        MyFunc.__init__(self, *args, **kwargs)

        self.ref_antenna = np.argmax(self.subject.data.show_antenna_strength())
        self.packet1 = np.random.randint(self.subject.data.length)
        self.packet2 = np.random.randint(self.subject.data.length)
        self.title1 = 'Before'
        self.title2 = 'After'
        self.suptitle = 'Calibration of ' + self.subject.name
        self.autosave = False
        self.notion = ''

    def __str__(self):
        return 'Test Phase Difference After Calibration'

    def func(self):

        antennas = list(range(self.subject.nrx))
        antennas.remove(int(self.ref_antenna))

        if self.rearrange is True:
            self.subject.data.rearrange_antenna()
        csi = self.subject.data.amp * np.exp(1.j * self.subject.data.phase)
        phase_diff = np.angle(csi * csi[:, :, self.ref_antenna, :][:, :, np.newaxis, :].conj())

        print(self.subject.name, "test_phase_diff plotting...", time.asctime(time.localtime(time.time())))

        fig, ax = plt.subplots(2, 1)
        ax[0].set_title(self.title1)
        ax[0].plot(phase_diff[self.packet1, :, antennas[0], 0],
                   label='antenna' + str(antennas[0]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet1), color='b')
        ax[0].plot(phase_diff[self.packet1, :, antennas[1], 0],
                   label='antenna' + str(antennas[1]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet1), color='r')
        ax[0].plot(phase_diff[self.packet2, :, antennas[0], 0],
                   label='antenna' + str(antennas[0]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet2), color='b',
                   linestyle='--')
        ax[0].plot(phase_diff[self.packet2, :, antennas[1], 0],
                   label='antenna' + str(antennas[1]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet2), color='r',
                   linestyle='--')
        ax[0].set_xlabel('#Subcarrier', loc='right')
        ax[0].set_ylabel('Phase Difference / $rad$')
        ax[0].legend()

        if self.rearrange is True:
            for ref in self.reference.values():
                ref.data.rearrange_antenna()

        self.subject.calibrate_phase(self.ref_antenna, self.reference)

        csi = self.subject.data.amp * np.exp(1.j * self.subject.data.phase)
        phasediff = np.angle(csi * csi[:, :, self.ref_antenna, :][:, :, np.newaxis, :].conj())

        ax[1].set_title(self.title2)
        ax[1].plot(phasediff[self.packet1, :, antennas[0], 0],
                   label='antenna' + str(antennas[0]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet1), color='b')
        ax[1].plot(phasediff[self.packet1, :, antennas[1], 0],
                   label='antenna' + str(antennas[1]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet1), color='r')
        ax[1].plot(phasediff[self.packet2, :, antennas[0], 0],
                   label='antenna' + str(antennas[0]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet2), color='b',
                   linestyle='--')
        ax[1].plot(phasediff[self.packet2, :, antennas[1], 0],
                   label='antenna' + str(antennas[1]) + '-' + str(self.ref_antenna) + ' #' + str(self.packet2), color='r',
                   linestyle='--')

        ax[1].set_xlabel('Subcarrier', loc='right')
        ax[1].set_ylabel('Phase Difference')
        ax[1].legend()

        plt.suptitle(self.suptitle)
        print(self.subject.name, "test_phase_diff plot complete", time.asctime(time.localtime(time.time())))

        if self.autosave is True:
            save_path = os.getcwd().replace('\\', '/') + "/visualization/" + self.subject.name[:4] + '/' + self.title + '/'
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
