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


def test_calibration(name0, name1, path):
    """
    Plots phase difference of 30 subcarriers of antenna1-0 and 2-0 from 2 random packets before and after calibration.
    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
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


def test_resampling(name1, path, sampling_rate=100, name0=None):
    """
    Plots amplitudes of antemma0 subcarrier0 before and after resampling.
    :param name1: subject csi
    :param sampling_rate: default is 100
    :param path: folder path that contains batch csi files
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
    Plots doppler spectrum. Walks through dynamic extraction and resampling.
    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """

    standard = name0 if isinstance(name0, pycsi.MyCsi) else npzloader(name0, path)
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

    csi.data.remove_inf_values()

    standard.data.remove_inf_values()

    csi.extract_dynamic(mode='running', window_length=91, reference_antenna=0)

    if csi.resample_packets() == 'bad':
        pass
    else:
        csi.doppler_by_music(pick_antenna=0)
        csi.data.view_spectrum(autosave=True, notion='_an0_ref0_running91')


def test_aoa(name0, name1, path):
    """
    Plots aoa spectrum. Walks through calibration and dynamic extraction.

    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
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
    csi.data.view_spectrum(threshold=10, autosave=True)


def test_aoatof(name0, name1, path):
    """
    Plots aoa-tof spectrum. Walks through calibration and dynamic extraction.

    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """

    standard = name0 if isinstance(name0, pycsi.MyCsi) else npzloader(name0, path)
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

    csi.data.remove_inf_values()
    standard.data.remove_inf_values()

    csi.calibrate_phase(standard)
    csi.extract_dynamic()

    csi.data.length = 10
    csi.data.amp = csi.data.amp[:10]
    csi.data.phase = csi.data.phase[:10]

    csi.aoa_tof_by_music()
    print(csi.data.spectrum.shape)
    csi.data.view_spectrum()


def test_phasediff(name1, path, name0=None):
    """
    Plots phase difference of 30 subcarriers of antenna1-0 and 2-0 from a random packet.
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """

    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
    csi.data.remove_inf_values()
    csilist = csi.data.amp * np.exp(1.j * csi.data.phase)

    ref_antenna = 0
    packet = np.random.randint(csi.data.length)

    diff_csilist = csilist * csilist[:, :, ref_antenna, :][:, :, np.newaxis, :].conj()
    plt.plot(np.unwrap(np.angle(diff_csilist[packet, :, :, 0])))
    plt.xlabel('subcarrier')
    plt.ylabel('difference of phase')
    plt.title(name1 + ' phasediff at ' + str(packet) + ' with ref ' + str(ref_antenna))
    plt.show()


def test_sanitize(name1, path, name0=None):
    """
    Plots phase difference of 30 subcarriers of 3 antennas from 2 random packets before and after sanitization.
    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """

    # standard = name0 if isinstance(name0, pycsi.MyCsi) else npzloader(name0, path)
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

    csi.data.remove_inf_values()
    csi.data.phase -= np.mean(csi.data.phase, axis=1).reshape(-1, 1,3,1)
    # standard.data.remove_inf_values()

    #packet1 = np.random.randint(csi.data.length)
    #packet2 = np.random.randint(csi.data.length)

    packet1 = 1000
    packet2 = 1001

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title("Before sanitization")
    ax[0].plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 0, :])), label='antenna0 #' + str(packet1), color='b')
    ax[0].plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 1, :])), label='antenna1 #' + str(packet1), color='r')
    ax[0].plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 2, :])), label='antenna2 #' + str(packet1), color='y')
    ax[0].plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 0, :])), label='antenna0 #' + str(packet2), color='b', linestyle='--')
    ax[0].plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 1, :])), label='antenna1 #' + str(packet2), color='r', linestyle='--')
    ax[0].plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 2, :])), label='antenna2 #' + str(packet2), color='y', linestyle='--')
    ax[0].set_xlabel('Subcarrier', loc='right')
    ax[0].set_ylabel('Phase Difference')
    ax[0].legend()

    csi.sanitize_phase()
    csi.data.phase -= np.mean(csi.data.phase, axis=1).reshape(-1, 1,3,1)

    ax[1].set_title("After sanitization")
    ax[1].plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 0, :])), label='antenna0 #' + str(packet1), color='b')
    ax[1].plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 1, :])), label='antenna1 #' + str(packet1), color='r')
    ax[1].plot(np.unwrap(np.squeeze(csi.data.phase[packet1, :, 2, :])), label='antenna2 #' + str(packet1), color='y')
    ax[1].plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 0, :])), label='antenna0 #' + str(packet2), color='b', linestyle='--')
    ax[1].plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 1, :])), label='antenna1 #' + str(packet2), color='r', linestyle='--')
    ax[1].plot(np.unwrap(np.squeeze(csi.data.phase[packet2, :, 2, :])), label='antenna2 #' + str(packet2), color='y', linestyle='--')
    ax[1].set_xlabel('Subcarrier', loc='right')
    ax[1].set_ylabel('Phase Difference')
    ax[1].legend()

    plt.suptitle('Sanitization of ' + name1[4:])
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


def test_times(name1, path, name0=None):

    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
    plt.plot(csi.data.timestamps)
    plt.show()


def order(index, mypath=None, batch=False, *args, **kwargs):

    menu = {1: test_calibration,
            2: test_resampling,
            3: test_doppler,
            4: test_aoa,
            5: test_aoatof,
            6: test_phasediff,
            7: test_sanitize,
            8: test_simulation,
            9: test_times}

    func = menu[index]

    print(func.__name__)

    if batch is True:
        print("- Enabling batch processing -")

        filenames = os.listdir(mypath)

        for file in filenames:
            name = file[:-9]
            result = func(name0=kwargs['name0'], name1=name, path=mypath)

        print("- Batch processing complete -")

    else:
        result = func(*args, **kwargs)

    return result


if __name__ == '__main__':

    n0 = "1010A01"
    n1 = "1010A16"

    npzpath = 'npsave/1010/'
    ref = npzloader(n0, npzpath)

    order(index=3, batch=True, name0=ref, name1=n1, mypath=npzpath)
