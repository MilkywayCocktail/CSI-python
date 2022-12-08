import numpy as np
import pycsi
import csitest
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import csi_loader

kim = pycsi.MyCsi('kim')
#csi_loader.dat2npy("D:/CAO/KIM/csi_programs/sample/sample_data_AoA/20180824233433.dat", '../npsave/1205/')
a, b, c, d = csi_loader.load_npy('../npsave/1205/20180824233433-csio.npy')
kim.load_lists(np.abs(a).swapaxes(1,3), np.angle(a).swapaxes(1,3), b)
kim.aoa_by_music()
kim.data.view_spectrum()
