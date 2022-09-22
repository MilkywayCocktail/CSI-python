import numpy as np
import matplotlib.pyplot as plt
import pycsi

# Simulation
num_antenna = 3
ground_truth_AoA = 0  # deg
center_freq = 5.67e+09
light_speed = 299792458
dist_antenna = 0.0264
csi = np.exp(-1.j * 2. * np.pi * np.arange(3) * dist_antenna * np.sin(
    np.deg2rad(ground_truth_AoA)) * center_freq / light_speed)
csi = csi[np.newaxis, :, np.newaxis] * np.ones((10, 1, 30))
csi = csi.transpose(0, 2, 1)
print(csi.shape)  # [packet, sub, rx]

def white_noise(X, N, snr):
    noise = np.random.randn(N)
    snr = 10 ** (snr/10)
    power = np.mean(np.square(X))
    npower = power / snr
    noise = noise * np.sqrt(npower)

today = pycsi.MyCsi("test", "./")
today.data.amp = np.abs(csi)
today.data.phase = np.angle(csi)
today.data.timestamps = np.arange(len(csi))
today.data.length = len(csi)
theta_list = np.arange(-90, 91, 1.)
today.aoa_by_music()

plt.plot(theta_list, today.data.spectrum[:, 0])
plt.show()

plt.plot(today.data.phase[:,20,0], label='antenna0')
plt.plot(today.data.phase[:,20,1], label='antenna1')
plt.plot(today.data.phase[:,20,2], label='antenna2')
plt.legend()
plt.show()
