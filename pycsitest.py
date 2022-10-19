import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pycsi
from functools import wraps

class MyTest(object):
    """
    A higher-level structure over MyCsi.\n
    Allows employing multiple MyCsi entities.\n
    Collects testing methods.\n
    """

    def __init__(self, input_title='', input_subject=None, input_date=None, input_reference=None,
                 input_path=None, batch=False):
        self.subject = input_subject
        self.reference = input_reference
        self.date = input_date
        self.title = input_title
        self.path = input_path
        self.batch_trigger = batch
        self.log = []
        self.methods = [method for method in dir(self) if method.startswith('__') is False and method.startswith('_') is True]

    @staticmethod
    def npzloader(input_name, input_path):
        """
        A loader that loads npz files into MyCsi object.\n
        :param input_name: name of the MyCsi object
        :param input_path: folder path of npz file (excluding filename)
        :return: csi data loaded into MyCsi object
        """
        if input_path is None:
            file = "npsave/" + input_name[:4] + '/' + input_name + "-csis.npz"
        else:
            file = input_path + input_name + "-csis.npz"

        _csi = pycsi.MyCsi(input_name, file)
        _csi.load_data()
        _csi.data.remove_inf_values()
        return _csi

    def logger(self, *args):
        """
        Logs message into log
        :param args: any information
        :return: updated log
        """
        log_path = os.getcwd().replace('\\', '/') + "/data/" + str(self.date) + '/'

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        logfile = open(log_path + str(self.title) + '.txt', mode='a', encoding='utf-8')

        for message in args:
            if isinstance(message, dict):
                for key, value in message.items():
                    logfile.write(str(key) + ' : ' + str(value) + '\n')
            logfile.write(str(message) + '\n')

        return log_path + str(self.title) + '.txt'

    def load_all_references(self, rearrange=False):
        if self.reference is not None:
            for key, value in self.reference.items():
                degref = value if isinstance(value, pycsi.MyCsi) else self.npzloader(value, self.path)
                if rearrange is True:
                    degref.data.rearrange_antenna()
                self.reference[key] = degref

    def show_all_methods(self):
        for i, method in enumerate(self.methods):
            print(i, ':', self.methods)

    def test(self, rearrange=False, input_index=None, input_name=None, *args, **kwargs):

        if input_index is not None:
            func = eval('self._' + self.methods[input_index])
        if input_name is not None and input_name in self.methods:
            func = eval('self._' + input_name)

        func = eval('self._' + input_name)
        params = locals()

        self.log.append(os.getcwd().replace('\\', '/') + "/logs/" + str(self.date) + '/')
        self.logger(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '###TEST START###', params)

        self.load_all_references(rearrange=rearrange)
        self.logger('reference:', self.reference)

        if self.batch_trigger is True:
            print("- Enabling batch processing -")
            self.logger('##Batch process##')

            filenames = os.listdir(self.path)

            for file in filenames:
                name = file[:-9]
                self.logger(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), name)
                subject = self.npzloader(name, self.path)
                kwargs['subject'] = subject
                kwargs['reference'] = self.reference
                func(self, *args, **kwargs)

            print("- Batch processing complete -")

        else:
            self.logger(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), )
            subject = self.npzloader(self.subject, self.path)
            kwargs['subject'] = subject
            kwargs['reference'] = self.reference
            func(self, *args, **kwargs)

        self.logger(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '###TEST FINISH###')

    @staticmethod
    def _test_phase_diff(self, *args, **kwargs):
        try:
            if 'subject' not in kwargs.keys() or kwargs['subject'] is None:
                raise
            if 'reference' not in kwargs.keys():
                raise

        except Exception:
            pass

        else:
            subject = kwargs['subject']
            ref_antenna = np.argmax(subject.data.show_antenna_strength())
            packet1 = np.random.randint(subject.data.length)
            packet2 = np.random.randint(subject.data.length)
            title1 = 'Before'
            title2 = 'After'
            suptitle = 'Calibration of ' + subject.name
            autosave = False
            notion = ''

            print(subject.data.show_antenna_strength())
            antennas = list(range(subject.nrx))
            antennas.remove(int(ref_antenna))
            print(antennas)

            csi = subject.data.amp * np.exp(1.j * subject.data.phase)
            phase_diff = np.angle(csi * csi[:, :, ref_antenna, :][:, :, np.newaxis, :].conj())
            print(phase_diff[packet1, :, antennas[0], 0])
            print(subject.name, "test_phase_diff plotting...", time.asctime(time.localtime(time.time())))

            fig, ax = plt.subplots(2, 1)
            ax[0].set_title(title1)
            ax[0].plot(phase_diff[packet1, :, antennas[0], 0],
                       label='antenna' + str(antennas[0]) + '-' + str(ref_antenna) + ' #' + str(packet1), color='b')
            ax[0].plot(phase_diff[packet1, :, antennas[1], 0],
                       label='antenna' + str(antennas[1]) + '-' + str(ref_antenna) + ' #' + str(packet1), color='r')
            ax[0].plot(phase_diff[packet2, :, antennas[0], 0],
                       label='antenna' + str(antennas[0]) + '-' + str(ref_antenna) + ' #' + str(packet2), color='b',
                       linestyle='--')
            ax[0].plot(phase_diff[packet2, :, antennas[1], 0],
                       label='antenna' + str(antennas[1]) + '-' + str(ref_antenna) + ' #' + str(packet2), color='r',
                       linestyle='--')
            ax[0].set_xlabel('#Subcarrier', loc='right')
            ax[0].set_ylabel('Phase Difference / $rad$')
            ax[0].legend()

            subject.calibrate_phase(cal_dict=kwargs['reference'])

            csi = subject.data.amp * np.exp(1.j * subject.data.phase)
            phasediff = np.angle(csi * csi[:, :, ref_antenna, :][:, :, np.newaxis, :].conj())

            ax[1].set_title(title2)
            ax[1].plot(phasediff[packet1, :, antennas[0], 0],
                       label='antenna' + str(antennas[0]) + '-' + str(ref_antenna) + ' #' + str(packet1), color='b')
            ax[1].plot(phasediff[packet1, :, antennas[1], 0],
                       label='antenna' + str(antennas[1]) + '-' + str(ref_antenna) + ' #' + str(packet1), color='r')
            ax[1].plot(phasediff[packet2, :, antennas[0], 0],
                       label='antenna' + str(antennas[0]) + '-' + str(ref_antenna) + ' #' + str(packet2), color='b',
                       linestyle='--')
            ax[1].plot(phasediff[packet2, :, antennas[1], 0],
                       label='antenna' + str(antennas[1]) + '-' + str(ref_antenna) + ' #' + str(packet2), color='r',
                       linestyle='--')

            ax[1].set_xlabel('Subcarrier', loc='right')
            ax[1].set_ylabel('Phase Difference')
            ax[1].legend()

            plt.suptitle(suptitle)
            print(subject.name, "test_phase_diff plot complete", time.asctime(time.localtime(time.time())))

            if autosave is True:
                save_path = os.getcwd().replace('\\', '/') + "/visualization/" + subject.name[:4] + '/' + self.title + '/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_name = save_path + subject.name[4:] + notion + '.png'
                plt.savefig(save_name)
                print(subject.name, "saved as", save_name, time.asctime(time.localtime(time.time())))
                plt.close()

            else:
                plt.show()
