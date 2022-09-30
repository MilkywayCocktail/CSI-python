import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pycsi
from functools import wraps


class MyTest(pycsi.MyCsi):
    """
    Inherits full functionalities of MyCsi.\n
    Extra methods added for testing.
    """


def npzloader(name, path=None):

    if path is None:
        file = "npsave/" + name[:4] + '/' + name + "-csis.npz"
    else:
        file = path + name + "-csis.npz"

    csi = pycsi.MyCsi(name, file)
    csi.load_data()
    return csi


def timereporter(csi_name=None, func_name=None):
    """
    A decorator that prints currently processed csi and function name, plus start and end time.

    :param csi_name: csi name
    :param func_name: function name
    :return: decorator
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(csi_name, func_name, "start...", time.asctime(time.localtime(time.time())))
            result = func(*args, **kwargs)
            print(csi_name, func_name, "complete", time.asctime(time.localtime(time.time())))
            return result
        return wrapper
    return decorator


def batch_tool(folder_path, func, *args, **kwargs):
    """
    Specify a path that contains csi data files (.npz). The specified function will walk through all files.

    :param folder_path: folder path that contains csi data files (.npz)
    :param func: a test function from the menu.
    :return: iterative function that enables batch processing
    """
    print("- Enabling batch processing -")

    filenames = os.listdir(folder_path)

    for file in filenames:
        name = file[:-9]
        func(name1=name, path=folder_path, *args, **kwargs)

    print("- Batch processing complete -")
    return


def test_calibration(name0, name1, path):
    """
    Plots phase difference of 30 subcarriers of antenna1-0 and 2-0 from 2 random packets.

    :param name0: reference csi
    :param name1: subject csi
    :return:
    """

    standard = name0 if isinstance(name0, pycsi.MyCsi) else npzloader(name0, path)
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

    csi.data.remove_inf_values()
    standard.data.remove_inf_values()

    csilist = csi.data.amp * np.exp(1.j * csi.data.phase)
    diff_csilist = csilist * csilist[:, :, 0, :][:, :, np.newaxis, :].conj()

    packet1 = np.random.randint(csi.data.length)
    packet2 = np.random.randint(csi.data.length)

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("Before calibration")
    ax[0].plot(np.angle(diff_csilist[packet1, :, 1, 0]), label='antenna0-1 #' + str(packet1), color='b')
    ax[0].plot(np.angle(diff_csilist[packet1, :, 2, 0]), label='antenna0-2 #' + str(packet1), color='r')
    ax[0].plot(np.angle(diff_csilist[packet2, :, 1, 0]), label='antenna0-1 #' + str(packet2), color='b', linestyle='--')
    ax[0].plot(np.angle(diff_csilist[packet2, :, 2, 0]), label='antenna0-2 #' + str(packet2), color='r', linestyle='--')
    ax[0].set_xlabel('Subcarrier', loc='right')
    ax[0].set_ylabel('Phase Difference')
    ax[0].legend()

    csi.calibrate_phase(standard)

    csilist = csi.data.amp * np.exp(1.j * csi.data.phase)
    diff_csilist = csilist * csilist[:, :, 0, :][:, :, np.newaxis, :].conj()

    ax[1].set_title("After calibration")
    ax[1].plot(np.angle(diff_csilist[packet1, :, 1, 0]), label='antenna0-1 #' + str(packet1), color='b')
    ax[1].plot(np.angle(diff_csilist[packet1, :, 2, 0]), label='antenna0-2 #' + str(packet1), color='r')
    ax[1].plot(np.angle(diff_csilist[packet2, :, 1, 0]), label='antenna0-1 #' + str(packet2), color='b', linestyle='--')
    ax[1].plot(np.angle(diff_csilist[packet2, :, 2, 0]), label='antenna0-2 #' + str(packet2), color='r', linestyle='--')
    ax[1].set_xlabel('Subcarrier', loc='right')
    ax[1].set_ylabel('Phase Difference')
    ax[1].legend()

    plt.suptitle('Calibration of ' + name1[4:] + ' vs ' + name0[4:])
    plt.show()


def test_resampling(name1, path, sampling_rate=100):
    """
    Plots amplitudes of antemma0 subcarrier0 before and after resampling.

    :param name1: subject csi
    :param sampling_rate: default is 100
    :return:
    """

    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

    print(csi.data.length)

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("Before resampling")
    ax[0].plot(np.squeeze(csi.data.amp[:, 0, 0, 0]), label='sub0 antenna0')
    ax[0].set_xlabel('Timestamp/s', loc='right')
    plt.sca(ax[0])
    labels1 = [i * csi.data.length // 10 for i in range(10)]
    labels1.append(csi.data.length - 1)
    l1 = [float('%.6f' % x) for x in csi.data.timestamps[labels1]]
    plt.xticks(labels1, l1)
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()

    csi.resample_packets(sampling_rate)

    # fig, ax = plt.subplots(2,1,2)
    ax[1].set_title("After resampling")
    ax[1].plot(np.squeeze(csi.data.amp[:, 0, 0, 0]), label='sub0 antenna0')
    ax[1].set_xlabel('Timestamp/s', loc='right')
    plt.sca(ax[1])
    labels2 = [i * csi.data.length // 10 for i in range(10)]
    labels2.append(csi.data.length - 1)
    print(labels2)
    l2 = [float('%.6f' % x) for x in csi.data.timestamps[labels2]]
    print(l2)
    plt.xticks(labels2, l2)
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()
    plt.suptitle('Resampling of ' + name1[4:] + ' @ 100Hz')
    plt.show()


def test_doppler(name0, name1, path):
    """
    Plots doppler spectrum. Walks through calibration, dynamic extraction and resampling.

    :param name0: reference csi
    :param name1: subject csi
    :return:
    """

    standard = name0 if isinstance(name0, pycsi.MyCsi) else npzloader(name0, path)
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

    csi.data.remove_inf_values()

    standard.data.remove_inf_values()

    csi.calibrate_phase(standard, reference_antenna=2)

    csi.extract_dynamic(reference_antenna=2)

    if csi.resample_packets() == 'bad':
        pass
    else:
        csi.doppler_by_music(pick_antenna=0)
        csi.data.view_spectrum(autosave=True, notion='_an0_cal2')


def test_aoa(name0, name1, path):
    """
    Plots aoa spectrum. Walks through  calibration, dynamic extraction and resampling.

    :param name0: reference csi
    :param name1: subject csi
    :return:
    """

    standard = name0 if isinstance(name0, pycsi.MyCsi) else npzloader(name0, path)
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
    csi.data.remove_inf_values()
    standard.data.remove_inf_values()

    csi.calibrate_phase(standard)
    csi.extract_dynamic()
    csi.resample_packets()

    csi.aoa_by_music()
    csi.data.view_spectrum()


def test_phasediff(name1, path):
    """
    Plots phase difference of 30 subcarriers of antenna1-0 and 2-0 from a random packet.

    :param name1: subject csi
    :return:
    """

    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
    csi.data.remove_inf_values()
    csilist = csi.data.amp * np.exp(1.j * csi.data.phase)

    diff_csilist = csilist * csilist[:, :, 0, :][:, :, np.newaxis, :].conj()
    plt.plot(np.angle(diff_csilist[13000, :, :, 0]))
    plt.xlabel('subcarrier')
    plt.ylabel('difference of phase')
    plt.title(name1 + 'phasediff')
    plt.show()


def test_simulation():
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
        snr = 10 ** (snr / 10)
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

    plt.plot(today.data.phase[:, 20, 0], label='antenna0')
    plt.plot(today.data.phase[:, 20, 1], label='antenna1')
    plt.plot(today.data.phase[:, 20, 2], label='antenna2')
    plt.legend()
    plt.show()


def test_times(name1, path):

    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
    plt.plot(csi.data.timestamps)
    plt.show()


def order(index, name0, name1, path):

    if index == 1:
        print("test_calibration")
        test_calibration(name0, name1, path)
    elif index == 2:
        print("test_resampling")
        test_resampling(name1, path)
    elif index == 3:
        print("test_doppler")
        test_doppler(name0, name1, path)
    elif index == 4:
        print("test_aoa")
        test_aoa(name0, name1, path)
    elif index == 5:
        print("test_phasediff")
        test_phasediff(name1, path)
    elif index == 6:
        print('test_simulation')
        test_simulation()
    elif index == 7:
        print('test_times')
        test_times(name1, path)


if __name__ == '__main__':

    menu = {1: 'test_calibration',
            2: 'test_resampling',
            3: 'test_doppler',
            4: 'test_aoa',
            5: 'test_phasediff',
            6: 'test_simulation',
            7: 'test_times'}

    n0 = "0919A00f"
    n1 = "0919A11"

    mypath = 'npsave/0919/A/'
    ref = npzloader(n0, mypath)

    batch_tool(mypath, order, 3, ref)

    # order(3, n0, n1)

