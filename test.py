import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pycsi

'''
'[7.19159539e+03 2.84124463e+03 1.75162060e+03 9.27874518e+02
' 6.88463775e+02 4.34973030e+02 4.65195045e+01 3.36275633e+01
' 1.74068612e+01 1.36320811e+01 1.21766247e+01 9.20933349e+00
' 6.59235218e+00 5.97109779e+00 4.53847191e+00 4.15139574e+00
' 3.95309017e+00 3.67496986e+00 2.94431685e+00 2.10873623e+00
' 1.83352551e+00 1.56853077e+00 1.36217416e+00 1.05220583e+00
' 8.99966127e-01 8.17047732e-01 5.52063264e-01 2.00270376e-01
' 9.85997021e-02 1.69925570e-02]'
'''

a = np.array([11,12,13,14,21,22,23,24,31,32,33,34])

#print(a.reshape((3,4)))
#print(a.reshape((4,3)))
#print(a.reshape((3,4)).swapaxes(0,1))


# Simulation
'''
num_antenna = 3
ground_truth_AoA = 20  # deg
center_freq = 5.67e+09
light_speed = 299792458
dist_antenna = 0.025
csi = np.exp(-1.j * 2. * np.pi * np.arange(3) * dist_antenna * np.sin(
    np.deg2rad(ground_truth_AoA)) * center_freq / light_speed)
csi = csi[np.newaxis, :, np.newaxis] * np.ones((10, 1, 30))
csi = csi.transpose(0, 2, 1)
print(csi.shape)  # [packet, sub, rx]

today = pycsi.MyCsi("test", "./")
today.data.amp = np.abs(csi)
today.data.phase = np.angle(csi)
today.data.timestamps = np.arange(len(csi))
today.data.length = len(csi)
theta_list = np.arange(-90, 91, 1.)
today.aoa_by_music(theta_list, smooth=False)

plt.plot(theta_list, today.data.spectrum[:, 0])
plt.show()
'''
# Processing

name = "0812C01"

mypath = "data/csi" + name + ".dat"
npzpath = "npsave/" + name + "-csis.npz"
pmpath = "npsave/" + name + "-spectrum.npz"


# CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

today = pycsi.MyCsi(name, npzpath)

#standard = pycsi.MyCsi("0812C01", "npsave/0812C01-csis.npz")

today.load_data()

#standard.load_data()

#plt.plot(np.unwrap(np.squeeze(standard.data.phase[:,0,0,0])))
#plt.show()


plt.subplot(2, 1, 1)
plt.title("Removing antenna phase offset - overall mean")
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,0,0])), label='antenna0')
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,1,0])), label='antenna1')
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,2,0])), label='antenna2')
plt.legend()


# Calibration

#today.data.vis_all_rx("phase")
#standard.data.vis_all_rx("phase")

today.remove_phase_offset()


#standard.remove_phase_offset()

#today.calibrate_phase(standard)
#today.aoa_by_music(theta_list)

#today.data.vis_spectrum(0)

plt.subplot(2, 1, 2)
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,0,0])), label='antenna0')
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,1,0])), label='antenna1')
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,2,0])), label='antenna2')
plt.legend()

plt.show()

