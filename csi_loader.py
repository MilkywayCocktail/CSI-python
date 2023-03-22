# -*- coding: utf-8 -*-

# Modified dat2npy to accept save_path

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import datetime
import numpy as np
import os
import glob
import warnings
import argparse
import sys

sys.path.append("../..")

try:
    import csi_parser_numba
except:
    csi_parser_numba = None
import csi_parser

# import utils
import constant_value


def _get_total_rss(rssi, agc):
    tmprssi = np.array([r for r in rssi if r != 0])
    rssi_mag = np.sum(np.power(10, tmprssi / 10.0))

    return 10.0 * np.log10(rssi_mag) - 44 - agc


def _get_scaled_csi(csi, rssi, noise, agc, is_agc=False):
    csi_pwr = np.sum(np.power(np.real(csi), 2) + np.power(np.imag(csi), 2))
    rssi_pwr = np.power(10, _get_total_rss(rssi, agc) / 10.0)
    scale = rssi_pwr / (csi_pwr / 30)

    noise_db = -92 if noise == -127 else noise
    total_noise_pwr = np.power(10, noise_db / 10.0) + scale * (csi.shape[0] * csi.shape[1])

    ret = csi * np.sqrt(scale / total_noise_pwr)

    if csi.shape[0] == 2:
        ret = ret * np.sqrt(2)
    elif csi.shape[0] == 3:
        ret = ret * np.sqrt(4.5)

    if is_agc:
        return ret / np.power(10, agc / 10.0)
    else:
        return ret


def _get_scaled_csilist(csilist, rssilist, noiselist, agclist, is_agc=False):
    try:
        csi_pwr = np.sum(np.power(np.absolute(csilist), 2.).reshape(len(csilist), -1), axis=-1)
    except:
        csi_pwr = np.array([np.sum(np.power(np.absolute(csi), 2.)) for csi in csilist])
    try:
        rssi_pwr = np.power(10., rssilist / 10.0)
        rssi_pwr[rssi_pwr == 1] = 0
        rssi_pwr = np.sum(rssi_pwr, axis=-1) / np.power(10., (44. + np.array(agclist)) / 10.0)
    except:
        rssi_pwr = np.array([np.sum(np.power(10., np.array(rssi)[np.array(rssi) != 0] / 10.0)) for rssi in rssilist]) \
                   / np.power(10., (44. + np.array(agclist)) / 10.0)
    scale = rssi_pwr / (csi_pwr / 30)

    ntx = np.array([csi.shape[0] for csi in csilist])
    noiselist = np.array(noiselist)
    noiselist[noiselist == -127] = -92
    total_noise_pwr = np.power(10., noiselist / 10.0) + scale * ntx * np.array([csi.shape[1] for csi in csilist])

    multiply_ntx = np.ones(len(csilist))
    multiply_ntx[ntx == 2] = np.sqrt(2.)
    multiply_ntx[ntx == 3] = np.sqrt(np.power(10., 4.5 / 10.0))
    ret = csilist * (np.sqrt(scale / total_noise_pwr)
                     * multiply_ntx).reshape((-1,) + (1,) * len(np.array(csilist).shape[1:]))

    return ret


def _combine_rssi_agc(rssilist, agclist):
    return rssilist - 44 - agclist.reshape(-1, 1)


def remove_sm(csi, rate):
    """
    https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/remove_sm.m
    changed the order of sub and tx!
    :param csi:
    :param rate:
    :return:
    """
    ntx = csi.shape[2]
    nsub = csi.shape[0]
    ret = np.zeros_like(csi)

    if ntx == 1:
        ret = csi
        return ret
    sm = []
    cond = (np.bitwise_and(rate, 2048) == 2048)

    if cond:
        if ntx == 3:
            sm = constant_value.sm_matrices.sm_3_40
        elif ntx == 2:
            sm = constant_value.sm_matrices.sm_2_40
    else:
        if ntx == 3:
            sm = constant_value.sm_matrices.sm_3_20
        elif ntx == 2:
            sm = constant_value.sm_matrices.sm_2_20

    for i in range(0, nsub):
        t = np.array(csi)[i, :, :]
        ret[i, :, :] = t.dot(np.transpose(sm))
    return ret


def remove_sm_list(csilist, ratelist=None, isHT40=None):
    ret = np.zeros_like(csilist)
    ntx = csilist.shape[1]
    nrx = csilist.shape[2]

    if ntx == 1:
        ret = csilist
        return ret
    sm = []
    if isHT40 is None:
        cond = (np.bitwise_and(ratelist[0], 2048) == 2048)
    else:
        cond = isHT40

    if cond:
        if ntx == 3:
            sm = constant_value.sm_matrices.sm_3_40
        elif ntx == 2:
            sm = constant_value.sm_matrices.sm_2_40
    else:
        if ntx == 3:
            sm = constant_value.sm_matrices.sm_3_20
        elif ntx == 2:
            sm = constant_value.sm_matrices.sm_2_20

    for tx in range(ntx):
        for rx in range(nrx):
            ret[:, tx, rx, :] = np.sum(csilist[:, :, rx, :] * sm[tx, :].reshape(1, -1, 1).conj(), axis=1)
    return ret


def load_datetime(filename):
    lines = np.loadtxt(filename, delimiter='\t', dtype=np.str)
    try:
        timelist = [datetime.datetime.strptime(lines[i], '%Y/%m/%d %H:%M:%S.%f') for i in range(len(lines))]
    except Exception:
        timelist = [datetime.datetime.strptime(lines[i], '%Y-%m-%d %H:%M:%S.%f') for i in range(len(lines))]

    return timelist


def load_csidata(filename, rssi_agc=False, return_all=False, debug=False, return_skip=False, *args, **kwargs):
    now = datetime.datetime.now()
    # if is2:
    #     timelist, csilist, rssilist, permlist, bfee_countlist, noiselist, agclist, fake_rate_n_flagslist = \
    #         read_bf_file.read_bf_file2(filename)
    # else:
    if debug:
        timelist, csilist, rssilist, permlist, bfee_countlist, noiselist, agclist, fake_rate_n_flagslist, skip_list = \
            csi_parser.read_bf_file(filename, debug=debug)
    elif csi_parser_numba is not None:
        timelist, csilist, rssilist, permlist, bfee_countlist, noiselist, agclist, fake_rate_n_flagslist, skip_list = \
            csi_parser_numba.read_bf_file_multi(filename)
    else:
        timelist, csilist, rssilist, permlist, bfee_countlist, noiselist, agclist, fake_rate_n_flagslist, skip_list = \
            csi_parser.read_bf_file_multi(filename)
    try:
        newcsilist = [np.zeros(csilist[t].shape, dtype=np.complex) for t in range(len(csilist))]
        for t in range(len(csilist)):
            # newcsilist[t][:, permlist[t].astype(int)[:csilist[t].shape[1]] - 1, :] = csilist[t][:, :, :]
            newcsilist[t][:, permlist[t].astype(int) - 1, :] = csilist[t][:, :, :]
    except Exception as e:
        print(e.message)
        newcsilist = [np.zeros((1, 1, 30), dtype=np.complex) for t in range(len(csilist))]
        for t in range(len(csilist)):
            # if csilist[t].shape[1] != 1:
            #     print str(t) + ": " + str(csilist[t].shape)

            newcsilist[t][:, :, :] = csilist[t][0, 0, :]

    print('loaded csi')
    # newcsilist = [_get_scaled_csi(newcsilist[i], rssilist[i], noiselist[i], agclist[i], is_agc=rssi_agc) for i in
    #               range(len(newcsilist))]
    newcsilist = _get_scaled_csilist(newcsilist, rssilist, noiselist, agclist, is_agc=rssi_agc)

    print('csi load time: ' + str((datetime.datetime.now() - now).total_seconds()) + '[s]')
    print(str(len(csilist)) + 'packets')

    if rssi_agc:
        rssilist = _combine_rssi_agc(np.array(rssilist), np.array(agclist))

    if return_all:
        return newcsilist, timelist, rssilist, {
            "time": timelist, "csi": csilist, "rssi": rssilist, "perm": permlist, "bfee_count": bfee_countlist,
            "noise": noiselist, "agc": agclist, "fake_rate_n_flags": fake_rate_n_flagslist}
    elif return_skip:
        return newcsilist, timelist, rssilist, skip_list
    else:
        return newcsilist, timelist, rssilist


def load_csidata_with_timestamp(filename, rssi_agc=False, debug=False, *args, **kwargs):
    csilist, timelist, rssilist, skip = load_csidata(filename, rssi_agc=rssi_agc, debug=debug,
                                                     return_skip=True)

    #    los_csilist, tdomaincsilist=csitools.multipath_mitigation(csilist)

    if os.path.exists(filename.replace('.dat', '_time_mod.txt')):
        print('loading from time_mod.txt')
        datetimelist = load_datetime(filename.replace('.dat', '_time_mod.txt'))
    elif os.path.exists(filename.replace('.dat', '_time.txt')):
        print('loading from time.txt')
        datetimelist = load_datetime(filename.replace('.dat', '_time.txt'))
    else:
        print('time file does not exist')
        datetimelist = None

    if datetimelist is not None:
        if len(datetimelist) != len(csilist):
            idx = np.ones(len(datetimelist), dtype=bool)
            idx[skip] = False
            datetimelist = np.array(datetimelist)[idx]

        if len(datetimelist) != len(csilist):
            warnings.warn("different length: datetimelist {} and csilist {}".format(len(datetimelist), len(csilist)))

        print('loaded timestamp')
        print('sampling rate: ' + str(float(len(datetimelist)) / (datetimelist[-1] - datetimelist[0]).total_seconds()))

    return csilist, timelist, rssilist, datetimelist


def clocktime2second(timelist, clock_freq=1. * 10 ** 6, clock_maxval=2. ** 32):
    timelist = np.array(timelist)
    clockdiff = timelist[1:] - timelist[:-1]
    # minus_indices = np.arange(len(clockdiff))[clockdiff < 0] + 1
    minus_indices = np.arange(len(clockdiff))[clockdiff < -clock_maxval / 2] + 1
    for i in minus_indices:
        timelist[i:] = timelist[i:] + clock_maxval

    return (timelist - timelist[0]).astype(float) / clock_freq


def unwrap_clocktime(timelist, clock_maxval=2. ** 32):
    timelist = np.array(timelist)
    clockdiff = timelist[1:] - timelist[:-1]
    # minus_indices = np.arange(len(clockdiff))[clockdiff < 0] + 1
    minus_indices = np.arange(len(clockdiff))[clockdiff < -clock_maxval / 2] + 1
    for i in minus_indices:
        timelist[i:] = timelist[i:] + clock_maxval

    return timelist


def dat2npy(filename, save_path, autosave=True):
    now = datetime.datetime.now()
    if csi_parser_numba is not None:
        timelist, csilist, rssilist, permlist, bfee_countlist, noiselist, agclist, fake_rate_n_flagslist, skip_list = \
            csi_parser_numba.read_bf_file_multi(filename)
    else:
        timelist, csilist, rssilist, permlist, bfee_countlist, noiselist, agclist, fake_rate_n_flagslist, skip_list = \
            csi_parser.read_bf_file_multi(filename)
    try:
        newcsilist = [np.zeros(csilist[t].shape, dtype=np.complex) for t in range(len(csilist))]
        for t in range(len(csilist)):
            # newcsilist[t][:, permlist[t].astype(int)[:csilist[t].shape[1]] - 1, :] = csilist[t][:, :, :]
            newcsilist[t][:, permlist[t].astype(int) - 1, :] = csilist[t][:, :, :]
    except Exception as e:
        print(e.message)
        newcsilist = [np.zeros((1, 1, 30), dtype=np.complex) for t in range(len(csilist))]
        for t in range(len(csilist)):
            newcsilist[t][:, :, :] = csilist[t][0, 0, :]

    print('loaded csi')
    # newcsilist = _get_scaled_csilist(newcsilist, rssilist, noiselist, agclist)

    print('csi load time: ' + str((datetime.datetime.now() - now).total_seconds()) + '[s]')
    print(str(len(csilist)) + 'packets')

    if os.path.exists(filename.replace('.dat', '_time_mod.txt')):
        print('loading from time_mod.txt')
        datetimelist = load_datetime(filename.replace('.dat', '_time_mod.txt'))
    elif os.path.exists(filename.replace('.dat', '_time.txt')):
        print('loading from time.txt')
        datetimelist = load_datetime(filename.replace('.dat', '_time.txt'))
    else:
        print('time file does not exist')
        datetimelist = None

    if datetimelist is not None:
        if len(datetimelist) != len(csilist):
            idx = np.ones(len(datetimelist), dtype=bool)
            skip_list = [s for s in skip_list if s < len(datetimelist)]
            idx[skip_list] = False
            datetimelist = np.array(datetimelist)[idx]

        if len(datetimelist) != len(csilist):
            warnings.warn("different length: datetimelist {} and csilist {}".format(len(datetimelist), len(csilist)))

        print('loaded timestamp')
        print('sampling rate: ' + str(float(len(datetimelist)) / (datetimelist[-1] - datetimelist[0]).total_seconds()))

    real_csilist = np.asarray(np.real(newcsilist), np.int8)
    assert np.all(real_csilist == np.real(newcsilist)), "real part of csi list may overflow or underflow"

    imag_csilist = np.asarray(np.imag(newcsilist), np.int8)
    assert np.all(imag_csilist == np.imag(newcsilist)), "imag part of csi list  may overflow or underflow"

    uint_rssilist = np.array(rssilist).astype(np.uint8)
    assert np.all(uint_rssilist == rssilist), "rssi list may overflow or underflow"

    int_noiselist = np.array(noiselist).astype(np.int8)
    assert np.all(int_noiselist == noiselist), "noise list may overflow or underflow"

    uint_agclist = np.array(agclist).astype(np.uint8)
    assert np.all(uint_agclist == agclist), "agc list may overflow or underflow"

    if autosave is True:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = save_path + os.path.basename(filename)[3:-4] + "-csio"
        np.save(save_name,
                ((real_csilist, imag_csilist), uint_rssilist, int_noiselist, uint_agclist, timelist, datetimelist))
        # utils.save_by_joblib(
    #     (real_csilist, imag_csilist, uint_rssilist, int_noiselist, uint_agclist, timelist, datetimelist),
    #     filename.replace(".dat", ".dump"))

    return real_csilist.swapaxes(1, 3) + 1.j * imag_csilist.swapaxes(1, 3), timelist


def load_npy(filename):
    csilist, rssilist, noiselist, agclist, timelist, datetimelist = np.load(filename, allow_pickle=True)
    print("loaded")
    csilist = csilist[0] + 1.j * csilist[1]
    csilist = _get_scaled_csilist(csilist, rssilist, noiselist, agclist)
    print("scaled")
    if len(datetimelist) != len(csilist):
        warnings.warn("different length: datetimelist {} and csilist {}".format(len(datetimelist), len(csilist)))
        minlen = min((len(datetimelist), len(csilist)))
        csilist = csilist[:minlen]
        datetimelist = datetimelist[:minlen]
        timelist = timelist[:minlen]
        rssilist = rssilist[:minlen]

    return csilist.swapaxes(1, 3), timelist, rssilist, datetimelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, action='store')
    args = parser.parse_args()

    if args.filename is not None:
        dat2npy(args.filename)  # load dat file & save to npy file
        # csilist, timelist, rssilist, datetimelist=load_npy(npyfilename) # load from npy file
