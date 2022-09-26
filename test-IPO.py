import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pycsi

# Processing

name0 = "0919A00f"
npzpath0 = "npsave/" + name0[:4] + '/csi' + name0 + "-csis.npz"

name1 = "0919A11"
npzpath1 = "npsave/" + name1[:4] + '/csi' + name1 + "-csis.npz"

# CSI data composition: [no_frames, no_subcarriers, no_rx_ant, no_tx_ant]

csi = pycsi.MyCsi(name1, npzpath1)

standard = pycsi.MyCsi(name0, npzpath0)

csi.load_data()

standard.load_data()

packet1 = np.random.randint(csi.data.length)
packet2 = np.random.randint(csi.data.length)

plt.title("Before calibration")
diff_phase11 = csi.data.phase[packet1, :, 1, :] - csi.data.phase[packet1, :, 0, :]
diff_phase12 = csi.data.phase[packet1, :, 2, :] - csi.data.phase[packet1, :, 0, :]
diff_phase21 = csi.data.phase[packet2, :, 1, :] - csi.data.phase[packet2, :, 0, :]
diff_phase22 = csi.data.phase[packet2, :, 2, :] - csi.data.phase[packet2, :, 0, :]

plt.plot(np.unwrap(np.squeeze(diff_phase11)), label='packet1 diff antenna0-1', color='b')
plt.plot(np.unwrap(np.squeeze(diff_phase12)), label='packet1 diff antenna0-2', color='r')
plt.plot(np.unwrap(np.squeeze(diff_phase21)), label='packet2 diff antenna0-1', color='b', linestyle='--')
plt.plot(np.unwrap(np.squeeze(diff_phase22)), label='packet2 diff antenna0-2', color='r', linestyle='--')
plt.legend()
plt.xlabel('Subcarrier Index')
plt.ylabel('Phase difference')
plt.show()


# Calibration

csi.data.remove_inf_values()
standard.data.remove_inf_values()

csi.calibrate_phase(standard)

plt.title("After calibration")
diff_phase11 = csi.data.phase[packet1, :, 1, :] - csi.data.phase[packet1, :, 0, :]
diff_phase12 = csi.data.phase[packet1, :, 2, :] - csi.data.phase[packet1, :, 0, :]
diff_phase21 = csi.data.phase[packet2, :, 1, :] - csi.data.phase[packet2, :, 0, :]
diff_phase22 = csi.data.phase[packet2, :, 2, :] - csi.data.phase[packet2, :, 0, :]

plt.plot(np.unwrap(np.squeeze(diff_phase11)), label='packet1 diff antenna0-1', color='b')
plt.plot(np.unwrap(np.squeeze(diff_phase12)), label='packet1 diff antenna0-2', color='r')
plt.plot(np.unwrap(np.squeeze(diff_phase21)), label='packet2 diff antenna0-1', color='b', linestyle='--')
plt.plot(np.unwrap(np.squeeze(diff_phase22)), label='packet2 diff antenna0-2', color='r', linestyle='--')
plt.legend()
plt.xlabel('Subcarrier Index')
plt.ylabel('Phase difference')
plt.show()
