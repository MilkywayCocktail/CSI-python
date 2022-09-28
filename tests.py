import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pycsi


class MyTest(pycsi.MyCsi):
    """
    Inherits full functionalities of MyCsi.\n
    Extra methods added for testing.
    """


def test_calibration(input_mycsi, input_standard):

    name0 = "0919A00f"
    npzpath0 = "npsave/" + name0[:4] + '/csi' + name0 + "-csis.npz"

    name1 = "0919A11"
    npzpath1 = "npsave/" + name1[:4] + '/csi' + name1 + "-csis.npz"

    csi = pycsi.MyCsi(name1, npzpath1)
    standard = pycsi.MyCsi(name0, npzpath0)

    csi.load_data()
    standard.load_data()

    packet1 = np.random.randint(csi.data.length)

    packet2 = np.random.randint(csi.data.length)

    ax = plt.subplot(2,1,1)
    ax.set_title("Before calibration")
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 0, 0])), label='packet1 antenna0', color='b')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 1, 0])), label='packet1 antenna1', color='y')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 2, 0])), label='packet1 antenna2', color='r')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 0, 0])), label='packet2 antenna0', linestyle='--',
            color='b')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 1, 0])), label='packet2 antenna1', linestyle='--',
            color='y')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 2, 0])), label='packet2 antenna2', linestyle='--',
            color='r')
    ax.legend()
    ax.set_xlabel('Subcarrier Index')
    ax.set_ylabel('Unwrapped CSI Phase')

    csi.data.remove_inf_values()
    standard.data.remove_inf_values()

    csi.calibrate_phase(standard)

    ax = plt.subplot(2,1,2)
    ax.set_title("After calibration")
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 0, 0])), label='packet1 antenna0', color='b')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 1, 0])), label='packet1 antenna1', color='y')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 2, 0])), label='packet1 antenna2', color='r')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 0, 0])), label='packet2 antenna0', linestyle='--',
            color='b')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 1, 0])), label='packet2 antenna1', linestyle='--',
            color='y')
    ax.plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 2, 0])), label='packet2 antenna2', linestyle='--',
            color='r')
    ax.legend()
    ax.set_xlabel('Subcarrier Index')
    ax.set_ylabel('Unwrapped CSI Phase')
    plt.show()


def test_resampling(sampling_rate=100):

    name0 = "0919B25"
    npzpath0 = "npsave/" + name0[:4] + '/csi' + name0 + "-csis.npz"

    csi = pycsi.MyCsi(name0, npzpath0)
    csi.load_data()
    print(csi.data.length)

    fig, ax = plt.subplots(2,1)
    ax[0].set_title("Before resampling")
    ax[0].plot(np.squeeze(csi.data.amp[:, 0, 0, 0]), label='sub0 antenna0')
    ax[0].set_xlabel('Timestamp/s', loc='right')
    plt.sca(ax[0])
    labels = [i * csi.data.length // 10 for i in range(10)]
    labels.append(csi.data.length - 1)
    l = [float('%.6f' % x) for x in csi.data.timestamps[labels]]
    plt.xticks(range(0, csi.data.length, csi.data.length//10), l)
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()

    csi.resample_packets(sampling_rate)

    # fig, ax = plt.subplots(2,1,2)
    ax[1].set_title("After resampling")
    ax[1].plot(np.squeeze(csi.data.amp[:, 0, 0, 0]), label='sub0 antenna0')
    ax[1].set_xlabel('Timestamp/s', loc='right')
    plt.sca(ax[1])
    labels = [i * csi.data.length // 10 for i in range(10)]
    labels.append(csi.data.length - 1)
    l = [float('%.6f' % x) for x in csi.data.timestamps[labels]]
    plt.xticks(range(0, csi.data.length, csi.data.length//10), l)
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()
    plt.suptitle('Resampling of ' + name0[4:] + ' @ 100Hz')
    plt.show()


def test_doppler():

    name0 = "0919A00f"
    npzpath0 = "npsave/" + name0[:4] + '/csi' + name0 + "-csis.npz"

    name1 = "0919A15"
    npzpath1 = "npsave/" + name1[:4] + '/csi' + name1 + "-csis.npz"

    csi = pycsi.MyCsi(name1, npzpath1)
    csi.load_data()
    csi.data.remove_inf_values()

    standard = pycsi.MyCsi(name0, npzpath0)
    standard.load_data()
    standard.data.remove_inf_values()

    csi.calibrate_phase(standard)
    csi.extract_dynamic()
    csi.resample_packets()

    csi.doppler_by_music()
    csi.data.view_spectrum()


def test_phase():

    name0 = "0919B00f"
    npzpath0 = "npsave/" + name0[:4] + '/csi' + name0 + "-csis.npz"

    csi = pycsi.MyCsi(name0, npzpath0)
    csi.load_data()
    csi.data.remove_inf_values()
    csilist = csi.data.amp * np.exp(1.j * csi.data.phase)
    print(csilist.shape)

    diff_csilist = csilist * csilist[:, :, 0, :][:, :, np.newaxis, :].conj()
    plt.plot(np.angle(diff_csilist[13000, :, :, 0]))
    print(np.angle(diff_csilist[13000, :, :, 0]))
    plt.xlabel('subcarrier')
    plt.ylabel('difference of phase')
    plt.show()


if __name__ == '__main__':

    test_doppler()
