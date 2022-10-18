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
    Extra methods added for testing.\n
    """


def npzloader(name, path=None):

    if path is None:
        file = "npsave/" + name[:4] + '/' + name + "-csis.npz"
    else:
        file = path + name + "-csis.npz"

    csi = pycsi.MyCsi(name, file)
    csi.load_data()
    csi.data.remove_inf_values()
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


def test_calibration(name1, path, cal_dict):
    """
    Plots phase difference of 30 subcarriers of antenna1-0 and 2-0 from 2 random packets before and after calibration.
    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """

    for key,value in cal_dict.items():
        degref = value if isinstance(value, pycsi.MyCsi) else npzloader(value, path)
        cal_dict[key] = degref

    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

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

    csi.calibrate_phase(cal_dict=cal_dict)

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

    plt.suptitle('Calibration of ' + name1[4:])
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


def test_doppler(name0, name1, path, resample=1000, wl=100):
    """
    Plots doppler spectrum. Walks through dynamic extraction and resampling.
    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """

    standard = name0 if isinstance(name0, pycsi.MyCsi) else npzloader(name0, path)
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

    if resample is True:
        if csi.resample_packets() == 'bad':
            pass

    csi.doppler_by_music(resample=resample, window_length=wl, stride=wl)
    csi.data.view_spectrum(threshold=-4, autosave=True, notion='_resample_1kHz')


def test_aoa(name1, path, cal_dict, name0=None):
    """
    Plots aoa spectrum. Walks through calibration and dynamic extraction.
    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """

    for key, value in cal_dict.items():
        degref = value if isinstance(value, pycsi.MyCsi) else npzloader(value, path)
        cal_dict[key] = degref
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)

    #csi.calibrate_phase(cal_dict=cal_dict)
    #csi.extract_dynamic()
    #csi.resample_packets()

    csi.aoa_by_music()
    csi.data.view_spectrum(threshold=0, autosave=False, notion='_vanilla')


def test_aoatof(name1, path, cal_dict, name0=None):
    """
    Plots aoa-tof spectrum. Walks through calibration and dynamic extraction.
    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """
    for key, value in cal_dict.items():
        degref = value if isinstance(value, pycsi.MyCsi) else npzloader(value, path)
        cal_dict[key] = degref
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
    print(csi.data.length)

    csi.calibrate_phase(cal_dict=cal_dict)
    csi.sanitize_phase()
    csi.extract_dynamic()

    csi.data.length = 10
    csi.data.amp = csi.data.amp[30000:30010]
    csi.data.phase = csi.data.phase[30000:30010]

    csi.aoa_tof_by_music()
    print(csi.data.spectrum.shape)
    csi.data.view_spectrum()


def test_aoadoppler(name1, path, cal_dict, name0=None):
    """
    Plots aoa-tof spectrum. Walks through calibration and dynamic extraction.
    :param name0: reference csi
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """
    for key, value in cal_dict.items():
        degref = value if isinstance(value, pycsi.MyCsi) else npzloader(value, path)
        cal_dict[key] = degref
    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
    print(csi.data.length)

    #csi.calibrate_phase(cal_dict=cal_dict)

    csi.aoa_doppler_by_music()
    print(csi.data.spectrum.shape)
    csi.save_spectrum(notion='_30sub')
    csi.data.view_spectrum()


def test_phasediff(name1, path, name0=None):
    """
    Plots phase difference of 30 subcarriers of antenna1-0 and 2-0 from a random packet.
    :param name1: subject csi
    :param path: folder path that contains batch csi files
    :return:
    """

    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
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


def test_abs(name1, path, name0=None):

    csi = name1 if isinstance(name1, pycsi.MyCsi) else npzloader(name1, path)
    print(csi.data.show_antenna_strength())


def order(index, batch=False, *args, **kwargs):

    menu = {1: test_calibration,
            2: test_resampling,
            3: test_doppler,
            4: test_aoa,
            5: test_aoatof,
            6: test_aoadoppler,
            7: test_phasediff,
            8: test_sanitize,
            9: test_simulation,
            10: test_times,
            11: test_abs}

    func = menu[index]

    print(func.__name__)

    path = kwargs['path']

    if batch is True:
        print("- Enabling batch processing -")

        filenames = os.listdir(path)

        for file in filenames:
            name = file[:-9]
            kwargs['name1'] = name
            func(*args, **kwargs)

        print("- Batch processing complete -")

    else:
        func(*args, **kwargs)


if __name__ == '__main__':

    n0 = "1010A01"
    n1 = "1010A26"

    npzpath = 'npsave/1010/'
    ref = npzloader(n0, npzpath)

    cal = {'0': "1010A01",
            '-30': "1010A02",
            '-60': "1010A03",
            '30': "1010A04",
            '60': "1010A05"}

    order(index=6, batch=False, name0=ref, name1=n1, path=npzpath, cal_dict=cal)
