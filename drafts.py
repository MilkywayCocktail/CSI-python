import pycsi
import csitest
import numpy as np
import random
import matplotlib.pyplot as plt

path = '../npsave/GT1/csi/'

name = 'GT1'

loader = csitest.MyTest.npzloader

simu = loader(name, path)

simu.doppler_by_music(window_length=100, stride=100, raw_timestamps=False, raw_window=False)
simu.data.view_spectrum(threshold=40)