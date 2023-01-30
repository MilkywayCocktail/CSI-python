import numpy as np
import time
import os
import csi_loader
import pycsi
import myfunc
import time
from functools import wraps


class MyTest(object):
    """
    A higher-level structure over MyCsi.\n
    Allows employing multiple MyCsi entities.\n
    Collects testing methods.\n
    """

    def __init__(self, title='',
                 date=None,
                 subject=None,
                 reference=None,
                 path=None,
                 batch=False,
                 func_index=0,
                 func_name=None,
                 sub_range=None):

        self.subject = subject
        self.reference = reference
        self.date = date
        self.title = str(title)
        self.path = path
        self.batch_trigger = batch
        self.testfunc = None
        self.sub_range = sub_range

        self.methods = [method for method in dir(myfunc) if method.startswith('_T') is True]
        self.methods = {i: m for i, m in enumerate(self.methods)}

        # Specify testing function
        if func_index is not None:
            self.select_func = self.methods[func_index]
        if func_name is not None:
            self.select_func = func_name
        if self.select_func is None:
            raise

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def npzloader(input_name, input_path, fc, bw):
        """
        A loader that loads npz files into MyCsi object.\n
        :param input_name: name of the MyCsi object (filename without '.npz')
        :param input_path: folder path of npz file (excluding filename)
        :param fc: center frequency of CSI (in GHz)
        :param bw: bandwidth of CSI (in MHz)
        :return: csi data loaded into MyCsi object
        """
        if input_path is None:
            print('Please specify path')
            return
        else:
            filepath = input_path + input_name + '-csio.npy'
            if not os.path.exists(filepath):
                filepath = input_path + input_name + '-csis.npz'
            _csi = pycsi.MyCsi(input_name, filepath, fc, bw)
            _csi.load_data()

        return _csi

    def logger(self, *args):
        """
        Logs message into log
        :param args: any information
        :return: updated log
        """
        log_path = "../logs/" + str(self.date) + '/'

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        logfile = open(log_path + self.title + '.txt', mode='a', encoding='utf-8')

        for message in args:
            if isinstance(message, dict):
                for key, value in message.items():
                    logfile.write(str(key) + ' : ' + str(value) + '\n')
            else:
                logfile.write(str(message) + '\n')

        return log_path + str(self.title) + '.txt'

    def load_all_references(self, fc, bw, rearrange=False):
        if self.reference is not None:
            for key, value in self.reference.items():
                degref = value if isinstance(value, pycsi.MyCsi) else self.npzloader(value, self.path, fc, bw)
                if rearrange is True:
                    degref.data.rearrange_antenna()
                self.reference[key] = degref

    def show_all_methods(self):
        for key, value in self.methods.items():
            print(key, ':', value)

    def run(self, fc=5.67, bw=40, **kwargs):
        self.logger(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + str(
            self.select_func) + ' ----TEST START----')

        print("######", self.title, "Test Start", time.asctime(time.localtime(time.time())))

        if self.batch_trigger is False and self.subject is not None:
            self.load_all_references(fc, bw)
            self.subject = self.npzloader(self.subject, self.path, fc, bw) \
                if not isinstance(self.subject, pycsi.MyCsi) else self.subject
            self.logger(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' ' + self.subject.name)

            self.testfunc = eval('myfunc.' + self.select_func +
                                 '(test_title=self.title, reference=self   .reference, subject=self.subject)')
            self.testfunc.set_params(**kwargs)
            self.logger(self.testfunc.__dict__)
            self.logger(self.testfunc.func())

        elif self.batch_trigger is True and self.path is not None:
            print("- Enabling batch processing -")
            self.logger('----Batch process----')

            filenames = os.listdir(self.path)
            self.load_all_references(fc, bw)

            for file in filenames:
                name = file[:-9]

                if self.sub_range is not None and name[-3:] not in self.sub_range:
                    continue

                else:
                    self.subject = self.npzloader(name, self.path,fc, bw)
                    self.logger(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' ' + self.subject.name)

                    self.testfunc = eval('myfunc.' + self.select_func +
                                         '(test_title=self.title, reference=self.reference, subject=self.subject)')
                    self.testfunc.set_params(**kwargs)
                    self.logger(self.testfunc.__dict__)
                    self.logger(self.testfunc.func())

            print("- Batch processing complete -")

        self.logger(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + str(
            self.select_func) + ' ----TEST FINISH----\n')

        print("######", self.title, "Test Complete", time.asctime(time.localtime(time.time())))


if __name__ == '__main__':

    sub = "0124A00"

    npzpath = '../npsave/0124/'

    cal = {'0': '0124A00'}

    # sub_range = ['A' + str(x).zfill(2) for x in range(0, 12)]

    # test0 = MyTest()
    # test0.show_all_methods()

    mytest = MyTest(title='aoa', date='0124', subject=sub, reference=cal, path=npzpath, batch=False,
                    func_index=0)
    mytest.run(fc=5.32, bw=20, calibrate=False, recursive=False, autosave=True,
               method='calibration + sanitization', notion='_dyn')





'''
0 : _TestAoA
1 : _TestAoADoppler
2 : _TestAoAToF
3 : _TestDoppler
4 : _TestPhase
5 : _TestPhaseDiff
6 : _TestResampling'''