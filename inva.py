import numpy as np
import pycsi
import csitest
import os
import matplotlib.pyplot as plt

loader = csitest.MyTest.npzloader

npzpath = 'npsave/1030/csi/'

a = '1030A00'

csi = loader(a, npzpath)
csi.data.show_shape()
csi.data.view_all_rx()