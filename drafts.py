import pycsi
import numpy as np
import random
import matplotlib.pyplot as plt

nrx = 3
ntx = 1
nsub = 30
center_freq = 5.67e+09
lightspeed = 299792458
dist_antenna = self.lightspeed / self.center_freq / 2.
bandwidth = 40e+06
delta_subfreq = 3.125e+05
length = length
sampling_rate = sampling_rate
amp = None
phase = None
csi = None
timestamps = np.arange(0, self.length, 1.) / self.sampling_rate
subfreq_list = np.arange(self.center_freq - 58 * self.delta_subfreq,
                              self.center_freq + 62 * self.delta_subfreq,
                              4 * self.delta_subfreq).reshape(-1, 1)