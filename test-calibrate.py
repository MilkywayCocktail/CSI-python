import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pycsi


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

name0 = "0919A00f"
npzpath0 = "npsave/" + name0[:4] + '/csi' + name0 + "-csis.npz"

name1 = "0919A13"
npzpath1 = "npsave/" + name1[:4] + '/csi' + name1 + "-csis.npz"



# CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

today = pycsi.MyCsi(name1, npzpath1)

standard = pycsi.MyCsi(name0, npzpath0)

today.load_data()

standard.load_data()

#standard.load_data()

#plt.plot(np.unwrap(np.squeeze(standard.data.phase[:,0,0,0])))
#plt.show()


plt.subplot(2, 1, 1)
plt.title("Calibration")
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,0,0])), label='antenna0')
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,1,0])), label='antenna1')
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,2,0])), label='antenna2')
plt.legend()


# Calibration

#today.data.vis_all_rx("phase")
#standard.data.vis_all_rx("phase")

#today.remove_phase_offset()
today.calibrate_phase(standard)
today.extract_dynamic()

plt.subplot(2, 1, 2)
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,0,0])), label='antenna0')
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,1,0])), label='antenna1')
plt.plot(np.unwrap(np.squeeze(today.data.phase[:,20,2,0])), label='antenna2')
plt.legend()

plt.show()

today.aoa_by_music()
today.data.vis_spectrum()
