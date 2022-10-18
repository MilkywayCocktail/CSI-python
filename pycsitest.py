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
    __credit = 'cao'

    def __int__(self, input_theme='', input_subjectlist=[], input_referencelist=[], path=None):
        self.subject = input_subjectlist
        self.reference = input_referencelist
        self.theme = input_theme

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



