import numpy as np
import pycsi
import os
import matplotlib.pyplot as plt

# Processing

name0 = "0919A00f"
npzpath0 = "npsave/" + name0[:4] + '/csi' + name0 + "-csis.npz"

name1 = "0919A03"
npzpath1 = "npsave/" + name1[:4] + '/csi' + name1 + "-csis.npz"

today = pycsi.MyCsi(name0, npzpath0)
today.load_data()
today.data.remove_inf_values()

standard = pycsi.MyCsi(name1, npzpath1)
standard.load_data()
standard.data.remove_inf_values()



today.calibrate_phase(standard)
#today.extract_dynamic()
#today.doppler_by_music()
#today.data.vis_spectrum(0, autosave=False)



#plt.plot(timediff * 1000)
#plt.title("timestamp intervals of " + name)
#plt.xlabel("#Packet")
#plt.ylabel("Interval/ms")
#plt.show()
#today.aoa_by_music()
#today.data.vis_spectrum(0, autosave=True)


