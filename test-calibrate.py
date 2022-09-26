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

# Processing

name0 = "0919A00f"
npzpath0 = "npsave/" + name0[:4] + '/csi' + name0 + "-csis.npz"

name1 = "0919A11"
npzpath1 = "npsave/" + name1[:4] + '/csi' + name1 + "-csis.npz"

# CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

today = pycsi.MyCsi(name1, npzpath1)

standard = pycsi.MyCsi(name0, npzpath0)

today.load_data()

standard.load_data()

#standard.load_data()

#plt.plot(np.unwrap(np.squeeze(standard.data.phase[:,0,0,0])))
#plt.show()

packet1 = np.random.randint(today.data.length)

packet2 = np.random.randint(today.data.length)

plt.title("Before calibration")
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet1, :, 0, 0])), label='packet1 antenna0', color='b')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet1, :, 1, 0])), label='packet1 antenna1', color='y')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet1, :, 2, 0])), label='packet1 antenna2', color='r')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet2, :, 0, 0])), label='packet2 antenna0', linestyle='--', color='b')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet2, :, 1, 0])), label='packet2 antenna1', linestyle='--', color='y')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet2, :, 2, 0])), label='packet2 antenna2', linestyle='--', color='r')
plt.legend()
plt.xlabel('Subcarrier Index')
plt.ylabel('Unwrapped CSI Phase')
plt.show()


# Calibration

#today.data.vis_all_rx("phase")
#standard.data.vis_all_rx("phase")

today.data.remove_inf_values()
standard.data.remove_inf_values()

today.calibrate_phase(standard)

plt.title("After calibration")
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet1, :, 0, 0])), label='packet1 antenna0', color='b')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet1, :, 1, 0])), label='packet1 antenna1', color='y')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet1, :, 2, 0])), label='packet1 antenna2', color='r')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet2, :, 0, 0])), label='packet2 antenna0', linestyle='--', color='b')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet2, :, 1, 0])), label='packet2 antenna1', linestyle='--', color='y')
plt.plot(np.unwrap(np.squeeze(today.data.phase[packet2, :, 2, 0])), label='packet2 antenna2', linestyle='--', color='r')
plt.legend()
plt.xlabel('Subcarrier Index')
plt.ylabel('Unwrapped CSI Phase')
plt.show()


#print(today.data.length)
#print(np.where(today.data.amp==float('-inf'))[0])



#today.aoa_by_music()
#today.data.vis_spectrum()
