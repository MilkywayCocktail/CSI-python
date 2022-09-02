   -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from joblib import Parallel, delayed
import datetime
import colorsys
from tqdm import tqdm
import constant_value


class config(object):
    sample_rate = 1000
    window_length = 100  # T
    num_subcarrier = 30  # F
    num_rx = 3  # A
    num_signal = 5  # L
    max_loop = 100  # N
    delta_f = 2 * 312.5 * (10. ** 3)  # FI
    # delta_t = 1. / sample_rate  # TI
    dist_antenna = 0.025  # AS
    center_freq = 5.320 * (10 ** 9)
    taulist = np.arange(-100., 400., 1.) * (10. ** -9)  # TR
    thetalist = np.deg2rad(np.arange(-0., 180., 1.))  # AR
    dopplerlist = np.arange(-5., 5., 0.01)  # DR
    update_ratio = 1.  # UR
    overlapped = 0.0
    centering = False
    n_jobs = -1
    debug = False
    uppe_stop = 50
    lowe_stop = 1

    @staticmethod
    def print_config():
        print("====================================")
        print("sample_rate:{}".format(config.sample_rate))
        print("window_length:{}".format(config.window_length))
        print("overlapped:{}".format(config.overlapped))
        print("num_subcarrier:{}".format(config.num_subcarrier))
        print("num_rx:{}".format(config.num_rx))
        print("num_signal:{}".format(config.num_signal))
        print("max_loop:{}".format(config.max_loop))
        print("delta_f:{}".format(config.delta_f))
        print("dist_antenna:{}".format(config.dist_antenna))
        print("center_freq:{}".format(config.center_freq))
        print("centering:{}".format(config.centering))
        print("====================================")


def steering_vector(delta_f, num_subcarrier, taulist,
                    dist_antenna, center_freq, num_rx, thetalist,
                    delta_t, window_length, dopplerlist):
    a_tau = np.exp(-1.j * 2 * np.pi
                   * delta_f * np.arange(num_subcarrier).reshape(-1, 1) * taulist.reshape(1, -1))
    a_theta = np.exp(-1.j * 2 * np.pi
                     * dist_antenna * (center_freq / constant_value.light_speed)
                     * np.cos(thetalist).reshape(1, -1) * np.arange(num_rx).reshape(-1, 1))

    # a_theta = np.exp(-1.j * thetalist.reshape(1, -1) * np.arange(num_rx).reshape(-1, 1))

    a_doppler = np.exp(1.j * 2 * np.pi * (center_freq / constant_value.light_speed)
                       * delta_t * np.arange(window_length).reshape(-1, 1) * dopplerlist.reshape(1, -1))

    return a_tau, a_theta, a_doppler


def sage_signal(latent_parameter, latent_index, a_tau, a_theta, a_doppler):
    """
    computing each path signal
    :param latent_parameter: tof, aoa, doppler, amplitude
    :param latent_index: tof, aoa, doppler
    :return:
    """
    latent_index = latent_index.astype(int)

    tof_matrix = a_tau[:, latent_index[0]].reshape(1, -1, 1)
    aoa_matrix = a_theta[:, latent_index[1]].reshape(1, 1, -1)
    doppler_matrix = a_doppler[:, latent_index[2]].reshape(-1, 1, 1)
    return latent_parameter[3] * tof_matrix * aoa_matrix * doppler_matrix


def sage_expectation(csi_signal, latent_signal, expect_index, update_ratio):
    """

    :param csi_signal: window_length * num_subcarrier * num_rx
    :param latent_signal: window_length * num_subcarrier * num_rx * num_signal
    :param expect_index: (int) index of path
    :param update_ratio: (int)
    :return:
    """

    noise_signal = csi_signal - np.sum(latent_signal, axis=3)
    expect_signal = latent_signal[:, :, :, expect_index] + update_ratio * noise_signal

    return expect_signal


def sage_maximization(latent_signal, latent_parameter, latent_index,
                      a_tau, a_theta, a_doppler, taulist, thetalist, dopplerlist,
                      window_length, num_subcarrier, num_rx):
    """

    :param latent_signal: window_length*num_subcarrier*num_rx*num_signal
    :param latent_parameter: tof, aoa, doppler, amplitude
    :param latent_index: tof, aoa, doppler
    :return:
    """
    latent_index = latent_index.astype(int)

    _latent_parameter = np.zeros_like(latent_parameter)
    _latent_index = np.zeros_like(latent_index)

    # Estimation of tof
    aoa_matrix = a_theta[:, latent_index[1]].reshape(1, 1, -1)
    doppler_matrix = a_doppler[:, latent_index[2]].reshape(-1, 1, 1)

    coeff_matrix = latent_signal * np.conj(aoa_matrix) * np.conj(doppler_matrix)
    coeff_vector = np.sum(coeff_matrix, axis=(0, 2)).reshape(-1, 1)
    tof_object_vector = np.abs(np.sum(coeff_vector * np.conj(a_tau), axis=0))
    _latent_index[0] = np.argmax(tof_object_vector)
    _latent_parameter[0] = taulist[_latent_index[0]]

    # Estimation of aoa
    tof_matrix = a_tau[:, _latent_index[0]].reshape(1, -1, 1)

    coeff_matrix = latent_signal * np.conj(doppler_matrix) * np.conj(tof_matrix)
    coeff_vector = np.sum(coeff_matrix, axis=(0, 1)).reshape(-1, 1)
    aoa_object_vector = np.abs(np.sum(coeff_vector * np.conj(a_theta), axis=0))
    _latent_index[1] = np.argmax(aoa_object_vector)
    _latent_parameter[1] = thetalist[_latent_index[1]]

    # Estimation of doppler
    aoa_matrix = a_theta[:, _latent_index[1]].reshape(1, 1, -1)

    coeff_matrix = latent_signal * np.conj(aoa_matrix) * np.conj(tof_matrix)
    coeff_vector = np.sum(coeff_matrix, axis=(1, 2)).reshape(-1, 1)
    doppler_object_vector = np.abs(np.sum(coeff_vector * np.conj(a_doppler), axis=0))
    _latent_index[2] = np.argmax(doppler_object_vector)
    _latent_parameter[2] = dopplerlist[_latent_index[2]]

    # Estimation of amplitude
    doppler_matrix = a_doppler[:, _latent_index[2]].reshape(-1, 1, 1)
    coeff_matrix = latent_signal * np.conj(aoa_matrix) * np.conj(tof_matrix) * np.conj(doppler_matrix)
    _latent_parameter[3] = np.sum(coeff_matrix) / (window_length * num_subcarrier * num_rx)

    return _latent_parameter, _latent_index, [tof_object_vector, aoa_object_vector, doppler_object_vector]




def sage_algorithm(csi_signal, id=0, num_sample=1, static_csilist=None):
    """
    Space Alternating Generalized Expectation Maximization (SAGE) algorithm
    :param csi_signal: window_length * num_subcarrier * num_rx
    :return:
    """
    initial_parameter = np.zeros((4, config.num_signal), dtype=complex)
    initial_index = np.zeros((3, config.num_signal), dtype=int)
    initial_index[0, :] = int(np.round((0 - config.taulist[0]) / (config.taulist[1] - config.taulist[0])))
    initial_index[1, :] = int(np.round((0 - config.thetalist[0]) / (config.thetalist[1] - config.thetalist[0])))
    initial_index[2, :] = int(np.round((0 - config.dopplerlist[0]) / (config.dopplerlist[1] - config.dopplerlist[0])))

    if config.centering:
        if static_csilist is None:
            # mean_amp = np.mean(np.abs(csi_signal), axis=0)
            # mean_pha = np.angle(np.mean(np.exp(1.j * np.angle(csi_signal)), axis=0))
            # static_csilist = (mean_amp * np.exp(1.j * mean_pha))
            static_csilist = np.mean(csi_signal, axis=0)
        csi_signal = csi_signal - static_csilist

    st = datetime.datetime.now()

    a_tau, a_theta, a_doppler = steering_vector(config.delta_f, config.num_subcarrier, config.taulist,
                                                config.dist_antenna, config.center_freq, config.num_rx,
                                                config.thetalist,
                                                1. / config.sample_rate, config.window_length, config.dopplerlist)

    # Initialize
    latent_signal = np.zeros((config.window_length, config.num_subcarrier, config.num_rx, config.num_signal),
                             dtype=complex)
    for i in range(config.num_signal):
        if initial_parameter[3, i] != 0:
            latent_signal[:, :, :, i] = sage_signal(initial_parameter[:, i], initial_index[:, i],
                                                    a_tau, a_theta, a_doppler)

    # Iteration
    final_parameter = np.copy(initial_parameter)
    temp_parameter = np.copy(initial_parameter)
    temp_index = np.copy(initial_index)
    final_index = np.copy(initial_index)
    for lp in range(config.max_loop):
        for i in range(config.num_signal):
            temp_signal = sage_expectation(csi_signal, latent_signal, i, config.update_ratio)
            temp_parameter[:, i], temp_index[:, i], obj_vec \
                = sage_maximization(temp_signal, final_parameter[:, i], final_index[:, i],
                                    a_tau, a_theta, a_doppler, config.taulist, config.thetalist, config.dopplerlist,
                                    config.window_length, config.num_subcarrier, config.num_rx)
            latent_signal[:, :, :, i] = sage_signal(temp_parameter[:, i], temp_index[:, i], a_tau, a_theta, a_doppler)

        parameter_diff = np.sqrt(np.sum(np.power(np.abs(temp_parameter - final_parameter), 2), axis=1))
        final_parameter = np.copy(temp_parameter)
        final_index = np.copy(temp_index)

        if parameter_diff[0] < 10. ** -9 and parameter_diff[1] < 1. / 180. * np.pi \
                and parameter_diff[2] < 0.01 and parameter_diff[3] < 10. ** -9:
            break

    residue_error = csi_signal - np.sum(latent_signal, axis=3)
    residue_error = np.mean(np.abs(residue_error)) / np.mean(np.abs(csi_signal))

    if config.debug:
        print("{}/{} {}%, Loop:{}, residual:{}, time:{}s".format(id, num_sample, int(id * 100 / num_sample),
                                                                 lp, residue_error,
                                                                 (datetime.datetime.now() - st).total_seconds()))
    return final_parameter, residue_error, id, obj_vec, lp


def sage_main(csilist, indices, num_sample=1):
    # estimated_parameter = np.zeros((4, config.num_signal, len(indices)), dtype=complex)
    # residue_errors = np.zeros(len(indices))

    initial_parameter = np.zeros((4, config.num_signal), dtype=complex)
    initial_index = np.zeros((3, config.num_signal), dtype=int)
    initial_index[0, :] = int(np.round((0 - config.taulist[0]) / (config.taulist[1] - config.taulist[0])))
    initial_index[1, :] = int(np.round((0 - config.thetalist[0]) / (config.thetalist[1] - config.thetalist[0])))
    initial_index[2, :] = int(np.round((0 - config.dopplerlist[0]) / (config.dopplerlist[1] - config.dopplerlist[0])))

    # for i in range(len(indices)):
    #     sage_algorithm(csilist[indices[i]], initial_parameter, initial_index, i)

    if config.debug:
        verbose = 10
    else:
        verbose = 0

    retlist = Parallel(n_jobs=config.n_jobs, verbose=verbose)(
        [delayed(sage_algorithm)(csilist[indices[i]], indices[i][0], num_sample,
                                 None
                                 # np.mean(csilist[
                                 #         max(indices[i][int(config.window_length / 2)] - int(config.sample_rate / 4),
                                 #             0):
                                 #         min(indices[i][int(config.window_length / 2)] + int(config.sample_rate / 4),
                                 #             len(csilist))],
                                 #         axis=0)
                                 )
         for i in range(len(indices))])

    estimated_parameter = np.array([ret[0] for ret in retlist]).transpose(1, 2, 0)
    residue_errors = np.array([ret[1] for ret in retlist])
    ids = np.array([ret[2] for ret in retlist])
    obj_vec = np.array([ret[3] for ret in retlist])
    num_loop = np.array([ret[4] for ret in retlist])

    return estimated_parameter, obj_vec


def sage_main_divided(csilist, indices, num_batch=100):
    estimated_parameter_list = np.empty((4, config.num_signal, 0), dtype=complex)
    obj_vec_list = np.empty((0, 3), dtype=float)
    with tqdm(total=len(indices), desc="Widar2") as pbar:
        for i in range(0, len(indices), num_batch):
            ids = indices[i:i + num_batch]
            estimated_parameter, obj_vec = sage_main(csilist, ids, num_sample=len(csilist))
            estimated_parameter_list = np.concatenate((estimated_parameter_list, estimated_parameter), axis=-1)
            obj_vec_list = np.concatenate((obj_vec_list, obj_vec), axis=0)
            pbar.update(len(ids))

    return estimated_parameter_list, obj_vec_list


def conj_mult(csilist, param_file_name=None):
    ### Find reference antenna.
    csi_amplitude = np.mean(np.abs(csilist), axis=0)
    csi_variance = np.std(np.abs(csilist), axis=0)
    csi_ratio = csi_amplitude / csi_variance
    ant_ratio = np.mean(csi_ratio, axis=1)
    midx = np.argmax(ant_ratio)
    # midx=0
    csi_ref = csilist[:, midx]
    if config.debug:
        print("Reference antenna:{}".format(midx))

    ### Weight
    alpha = np.min(np.abs(csilist), axis=0)
    csilist = (np.abs(csilist) - alpha) * np.exp(1.j * np.angle(csilist))
    beta = np.sum(alpha) ** 2 / len(csilist[0].reshape(-1)) * 1000
    csi_ref = (np.abs(csi_ref) + beta) * np.exp(1.j * np.angle(csi_ref))

    # alpha = np.max(np.absolute(csilist[:, 0, :]))
    # beta = 1000 * alpha
    # csilist = (np.absolute(csilist) / alpha) * np.exp(1.j * np.angle(csilist))
    # csi_ref = (np.absolute(csi_ref) * (beta ** 2)) * np.exp(1.j * np.angle(csi_ref))

    csi_mult = csilist * np.conj(csi_ref).reshape(csi_ref.shape[0], 1, csi_ref.shape[1])

    if param_file_name is not None:
        with open(param_file_name, "w") as f:
            f.write("Reference antenna,{}\n".format(midx))
            f.write("alpha,{}\n".format(alpha))
            f.write("beta,{}\n".format(beta))

    return csi_mult


def preprocess_Widar2(csilist, sec_timelist, datetimelist=None, use_filter=True, static_indices=None,
                      use_bandstop_filter=False, bandstop=0, param_file_name=None):
    csi_mult = conj_mult(csilist, param_file_name)

    ### Filter
    if use_filter:
        if config.debug:
            print("filtering...")
        if datetimelist is not None:
            hlfrt = float((len(csilist) / (datetimelist[-1] - datetimelist[0]).total_seconds()) / 2)
        else:
            hlfrt = config.sample_rate / 2.

        B33, A33 = signal.butter(1, [max((bandstop - 2), 1) / hlfrt, (bandstop + 2) / hlfrt], 'bandstop')
        # B66, A66 = signal.butter(1, [63. / hlfrt, 69. / hlfrt], 'bandstop')

        # uppe_orde = 6
        uppe_stop = config.uppe_stop
        # lowe_orde = 3
        lowe_stop = config.lowe_stop
        # B, A = signal.butter(5, [lowe_stop / hlfrt, uppe_stop / hlfrt], 'bandpass')
        B, A = signal.butter(4, uppe_stop / hlfrt, 'lowpass')
        B2, A2 = signal.butter(4, lowe_stop / hlfrt, 'highpass')
        # csi_filter = signal.filtfilt(B, A, csi_mult, axis=0)  # python default
        # csi_mult = signal.filtfilt(B, A, csi_mult, axis=0, padlen=3 * (max(len(A), len(B)) - 1))  # matlab

        csi_filter = np.zeros_like(csi_mult)
        for i in range(csi_mult.shape[1]):
            for j in range(csi_mult.shape[2]):
                # csi_filter[:, i, j] = signal.filtfilt(B, A, csi_mult[:, i, j])
                tmp = csi_mult[:, i, j]
                if use_bandstop_filter:
                    tmp = signal.filtfilt(B33, A33, tmp)
                    # tmp = signal.filtfilt(B66, B66, tmp)
                csi_filter[:, i, j] = signal.filtfilt(B2, A2, signal.filtfilt(B, A, tmp))

        csi_mult = csi_filter

        # wl = 100
        # w = np.ones(wl) / float(wl)
        # tmp_csi_mult = csi_mult.reshape(len(csi_mult), -1)

        # moving_avg_amp = np.array(Parallel(n_jobs=-1, verbose=10)(
        #     [delayed(np.convolve)(np.abs(tmp_csi_mult[:, i]), w, mode="same")
        #      for i in range(tmp_csi_mult.shape[1])])).T
        # moving_avg_pha = np.array(Parallel(n_jobs=-1, verbose=10)(
        #     [delayed(np.convolve)(np.angle(tmp_csi_mult[:, i]), w, mode="same")
        #      for i in range(tmp_csi_mult.shape[1])])).T
        #
        # csi_mult = (moving_avg_amp * np.exp(1.j * moving_avg_pha)).reshape(csi_mult.shape)

        # moving_avg_csi_mult = np.array(Parallel(n_jobs=-1, verbose=10)(
        #     [delayed(np.convolve)(tmp_csi_mult[:, i], w, mode="same")
        #      for i in range(tmp_csi_mult.shape[1])])).T
        # csi_mult = moving_avg_csi_mult.reshape(csi_mult.shape)

    if static_indices is not None:
        static_csi = csi_mult[static_indices]
        csi_mult = csi_mult - np.mean(static_csi, axis=0)

    ### Interpolation
    if config.debug:
        print("Interpolating...")
    interp_stamp = np.arange(0, sec_timelist[-1] * config.sample_rate) / config.sample_rate
    csi_interp = interpolate.interp1d(sec_timelist, csi_mult, axis=0)(interp_stamp)

    return csi_interp


def widar_main(csilist, sec_timelist, datetimelist=None, use_filter=True, static_indices=None,
               use_bandstop_filter=False, bandstop=0, param_file_name=None, num_batch=2000):
    config.print_config()

    if csilist.ndim == 4:
        csilist = csilist[:, 0]

    csilist = preprocess_Widar2(csilist, sec_timelist, datetimelist, use_filter, static_indices, use_bandstop_filter,
                                bandstop, param_file_name=param_file_name)

    ### Estimation
    slide = int(config.window_length * (1. - config.overlapped))
    start = np.arange(0, len(csilist) - config.window_length, slide, dtype=int)
    indices = [range(i, i + config.window_length) for i in start]

    estimated_parameter, obj_vec = sage_main_divided(csilist.transpose(0, 2, 1), indices, num_batch=num_batch)

    if datetimelist is not None:
        interp_stamp = np.arange(0, sec_timelist[-1] * config.sample_rate) / config.sample_rate
        float_datetimelist = np.array([(d - datetimelist[0]).total_seconds() for d in datetimelist])
        float_datetime_interp = interpolate.interp1d(sec_timelist, float_datetimelist, axis=0)(interp_stamp)
        datetime_interp = np.array([datetimelist[0] + datetime.timedelta(seconds=d) for d in float_datetime_interp])

        return estimated_parameter, np.array([datetime_interp[ids][int(len(ids) / 2)] for ids in indices]), obj_vec
    else:
        return estimated_parameter, obj_vec


def plot_estimated_parameter(estimated_parameter):
    fig, axs = plt.subplots(2, 2, figsize=(24, 15))
    axs = axs.flatten()

    axs[0].scatter(range(estimated_parameter.shape[2]) * estimated_parameter.shape[1],
                   estimated_parameter[0].real.reshape(-1), c=np.log(np.abs(estimated_parameter[3]).reshape(-1)),
                   linewidths=0)
    axs[1].scatter(range(estimated_parameter.shape[2]) * estimated_parameter.shape[1],
                   np.rad2deg(estimated_parameter[1].real).reshape(-1),
                   c=np.log(np.abs(estimated_parameter[3]).reshape(-1)), linewidths=0)
    axs[2].scatter(range(estimated_parameter.shape[2]) * estimated_parameter.shape[1],
                   estimated_parameter[2].real.reshape(-1), c=np.log(np.abs(estimated_parameter[3]).reshape(-1)),
                   linewidths=0)
    axs[3].scatter(range(estimated_parameter.shape[2]) * estimated_parameter.shape[1],
                   np.abs(estimated_parameter[3]).reshape(-1), c=np.log(np.abs(estimated_parameter[3]).reshape(-1)),
                   linewidths=0)

    axs[0].set_title("tof")
    axs[0].set_ylim(np.min(estimated_parameter[0].real), np.max(estimated_parameter[0].real))
    axs[1].set_title("aoa")
    axs[1].set_ylim(0, 180)
    axs[2].set_title("doppler")
    axs[3].set_title("amplitude")

    axs[0].set_xlim(0, estimated_parameter.shape[2])
    axs[1].set_xlim(0, estimated_parameter.shape[2])
    axs[2].set_xlim(0, estimated_parameter.shape[2])
    axs[3].set_xlim(0, estimated_parameter.shape[2])

    return fig


def plot_obj_vec(estimated_parameter, obj_vec):
    fig, axs = plt.subplots(2, 2, figsize=(24, 15))
    axs = axs.flatten()

    tof_obj_vec = np.array([i for i in obj_vec.T[0]])
    aoa_obj_vec = np.array([i for i in obj_vec.T[1]])
    doppler_obj_vec = np.array([i for i in obj_vec.T[2]])

    maxamp = np.max(np.log(np.abs(estimated_parameter[3]).T.reshape(-1)))
    minamp = np.min(np.log(np.abs(estimated_parameter[3]).T.reshape(-1)))
    scaled_amp = ((np.log(np.abs(estimated_parameter[3]).T.reshape(-1)) - minamp) / (maxamp - minamp)) * 2. / 3.

    maxid = signal.argrelmax(tof_obj_vec, order=1, axis=1)
    scaled_obj = ((tof_obj_vec - np.min(tof_obj_vec, axis=1).reshape(-1, 1))
                  / (np.max(tof_obj_vec, axis=1) - np.min(tof_obj_vec, axis=1)).reshape(-1, 1))
    color = [colorsys.hsv_to_rgb(2. / 3. - a, v, 1.) for (a, v) in
             zip(scaled_amp[maxid[0]], scaled_obj[maxid[0], maxid[1]])]

    axs[0].scatter(maxid[0], config.taulist[maxid[1]], c=color, linewidths=0)

    maxid = signal.argrelmax(aoa_obj_vec, order=1, axis=1)
    scaled_obj = ((aoa_obj_vec - np.min(aoa_obj_vec, axis=1).reshape(-1, 1))
                  / (np.max(aoa_obj_vec, axis=1) - np.min(aoa_obj_vec, axis=1)).reshape(-1, 1))
    color = [colorsys.hsv_to_rgb(2. / 3. - a, v, 1.) for (a, v) in
             zip(scaled_amp[maxid[0]], scaled_obj[maxid[0], maxid[1]])]

    axs[1].scatter(maxid[0], np.arange(180)[maxid[1]], c=color, linewidths=0)

    maxid = signal.argrelmax(doppler_obj_vec, order=1, axis=1)
    scaled_obj = ((doppler_obj_vec - np.min(doppler_obj_vec, axis=1).reshape(-1, 1))
                  / (np.max(doppler_obj_vec, axis=1) - np.min(doppler_obj_vec, axis=1)).reshape(-1, 1))
    color = [colorsys.hsv_to_rgb(2. / 3. - a, v, 1.) for (a, v) in
             zip(scaled_amp[maxid[0]], scaled_obj[maxid[0], maxid[1]])]

    axs[2].scatter(maxid[0], config.dopplerlist[maxid[1]], c=color, linewidths=0)

    color = [colorsys.hsv_to_rgb(2. / 3. - a, 1., 1.) for a in scaled_amp]
    axs[3].scatter(range(estimated_parameter.shape[2]) * estimated_parameter.shape[1],
                   np.abs(estimated_parameter[3]).T.reshape(-1), c=color, linewidths=0)

    # axs[0].scatter(maxid[0],
    #                config.taulist[maxid[1]], c=np.log(np.abs(estimated_parameter[3][:, maxid[0]]).T.reshape(-1)),
    #                linewidths=0)
    # maxid = signal.argrelmax(aoa_obj_vec, order=1, axis=1)
    # axs[1].scatter(maxid[0],
    #                np.arange(180)[maxid[1]], c=np.log(np.abs(estimated_parameter[3][:, maxid[0]]).T.reshape(-1)),
    #                linewidths=0)
    # maxid = signal.argrelmax(doppler_obj_vec, order=1, axis=1)
    # axs[2].scatter(maxid[0],
    #                config.dopplerlist[maxid[1]], c=np.log(np.abs(estimated_parameter[3][:, maxid[0]]).T.reshape(-1)),
    #                linewidths=0)
    # axs[3].scatter(range(estimated_parameter.shape[2]) * estimated_parameter.shape[1],
    #                np.abs(estimated_parameter[3]).T.reshape(-1), c=np.log(np.abs(estimated_parameter[3]).T.reshape(-1)),
    #                linewidths=0)

    axs[0].set_title("tof")
    axs[0].set_ylim(np.min(config.taulist), np.max(config.taulist))
    axs[1].set_title("aoa")
    axs[1].set_ylim(0, 180)
    axs[2].set_title("doppler")
    axs[3].set_title("amplitude")

    return fig


def _test(csilist, timestamp):
    ### Find reference antenna.
    csi_amplitude = np.mean(np.abs(csilist), axis=0)
    csi_variance = np.std(np.abs(csilist), axis=0)
    csi_ratio = csi_amplitude / csi_variance
    ant_ratio = np.mean(csi_ratio, axis=1)
    midx = np.argmax(ant_ratio)
    csi_ref = csilist[:, midx]
    print("Reference antenna:{}".format(midx))

    ### Weight
    alpha = np.min(np.abs(csilist), axis=0)
    csilist = (np.abs(csilist) - alpha) * np.exp(1.j * np.angle(csilist))
    beta = np.sum(alpha) / len(csilist[0].reshape(-1)) * 1000
    csi_ref = (np.abs(csi_ref) + beta) * np.exp(1.j * np.angle(csi_ref))

    csi_mult = csilist * np.conj(csi_ref).reshape(csi_ref.shape[0], 1, csi_ref.shape[1])

    # rx = 0
    # sub = 15
    # fig, axs = plt.subplots(2, 1, figsize=(16, 16))
    # axs = axs.flatten()
    # axs[0].plot(np.abs(csi_mult[:, rx, sub]))
    # axs[1].plot(np.angle(csi_mult[:, rx, sub]))
    # fig.savefig("/home/maekawa/Dropbox/tmp_widar_data.jpg")

    estimated_parameter = widar_main(csilist, timestamp, use_filter=True)
    plot_estimated_parameter(estimated_parameter)

    ### Estimation
    csi_interp = io.loadmat("/mnt/poplin/2018/ohara/csi/Widar2.0Project/data/classroom/interp-T01.mat")["csi_interp"]
    slide = int(config.window_length * (1. - config.overlapped))
    start = np.arange(0, len(csi_interp) - config.window_length, slide, dtype=int)
    indices = [range(i, i + config.window_length) for i in start]

    csi_interp = csi_interp.reshape(-1, config.num_rx, config.num_subcarrier)

    estimated_parameter = sage_main(csi_interp.transpose(0, 2, 1), indices)

    fig = plot_estimated_parameter(estimated_parameter)
    # fig.savefig("/home/maekawa/Dropbox/tmp_widar_est.jpg")

    em = io.loadmat("/mnt/poplin/2018/ohara/csi/Widar2.0Project/data/classroom/PPM-T01.mat")
    estimated_parameter = em["estimated_parameter"]
    fig = plot_estimated_parameter(estimated_parameter)

    estimated_path = io.loadmat("/mnt/poplin/2018/ohara/csi/Widar2.0Project/data/classroom/PMM-T01.mat")[
        "estimated_path"]
    fig.axes[0].plot(estimated_path[0].real.T)
    fig.axes[1].plot(np.rad2deg(estimated_path[1].real.T))
    fig.axes[2].plot(estimated_path[2].real.T)
    fig.axes[3].plot(np.abs(estimated_path[3].T))
    plt.show()


if __name__ == "__main__":
    from scipy import io
    from loader import csi_loader

    dev_conf = io.loadmat("/mnt/poplin/2018/ohara/csi/Widar2.0Project/data/classroom/device_config.mat")
    data = io.loadmat("/mnt/poplin/2018/ohara/csi/Widar2.0Project/data/classroom/T01.mat")
    _test(data["csi_data"].reshape(-1, 3, 30),
          csi_loader.clocktime2second(data["time_stamp"].reshape(-1)))



    # estimated_parameter = widar_main(data["csi_data"].reshape(-1, 3, 30),
    #                                  csi_loader.clocktime2second(data["time_stamp"].reshape(-1)))

    # print(estimated_parameter.shape)
