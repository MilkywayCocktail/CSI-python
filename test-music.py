import numpy as np
import pycsi
import os

# Processing
'''
name = "0919A00c"

mypath = "data/0919/csi" + name + ".dat"
npzpath = "npsave/csi" + name + "-csis.npz"
pmpath = "npsave/" + name + "-spectrum.npz"
  
today = pycsi.MyCsi(name, npzpath)
#today.load_data()
#today.aoa_by_music()
#today.data.vis_spectrum(0, autosave=True)
'''


filepath = "npsave/0919/"
filenames = os.listdir(filepath)
for file in filenames:
    name = file[3:-9]
    npzpath = filepath + file
    # pmpath = "npsave/" + name + "-spectrum.npz"
    today = pycsi.MyCsi(name, npzpath)
    today.load_data()
    today.aoa_by_music()
    today.data.vis_spectrum(0, autosave=True)
