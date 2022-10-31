import numpy as np
import pycsi
import csitest
import os
import matplotlib.pyplot as plt

loader = csitest.MyTest.npzloader

npzpath = '../npsave/1030/csi/'

cal = {'0': "1030B12",
       '30': "1030B13",
       '60': "1030B14",
       '90': "1030B15",
       '120': "1030B16",
       '150': "1030B17",
       '180': "1030B18",
       '210': "1030B19",
       '240': "1030B20",
       '270': "1030B21",
       '300': "1030B22",
       '330': "1030B23",
       }

diffs = []
for key, value in cal.items():
    ref = loader(value, npzpath)
    if ref.data.remove_inf_values() == 'bad':
        diffs.append([np.nan, np.nan, np.nan])
        continue

    ref_angle = eval(key)

    ref_csi = ref.data.amp * np.exp(1.j * ref.data.phase)
    ref_diff = np.mean(ref_csi * ref_csi[:, :, 0, :][:, :, np.newaxis, :].conj(), axis=(0, 1))
    true_diff = np.exp([-1.j * np.pi * antenna * np.sin(ref_angle * np.pi / 180) for antenna in range(3)]).reshape(-1, 1)
    #true_diff = np.exp(0)

    diffs.append(np.squeeze(np.angle(ref_diff.reshape(-1, 1) * true_diff.conj())).tolist())

print(diffs)
diffs = np.array(diffs)

print(np.mean(diffs, axis=0))
x = list(range(0, 360, 30))
plt.scatter(x, diffs[:, 0], c='r', label='0-0')
plt.scatter(x, diffs[:, 1], c='b', label='1-0')
plt.scatter(x, diffs[:, 2], c='g', label='2-0')
plt.title("Initial Phase Offsets")

plt.xlabel('Position / $deg$')
plt.ylabel('Phase Difference / $rad$')
plt.legend()
plt.show()


