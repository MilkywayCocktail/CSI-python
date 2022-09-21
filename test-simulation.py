import numpy as np
import matplotlib.pyplot as plt
import pycsi

# Simulation
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
