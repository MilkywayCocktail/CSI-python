# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import struct
import numpy as np
from joblib import Parallel, delayed
from numba import jit

DTYPE_LENGTH_TLV = np.dtype([
    ("length", np.uint16),
]).newbyteorder('>')

DTYPE_CSI_HEADER_TLV = np.dtype([
    ("timestamp_low", np.uint32),
    ("bfee_count", np.uint16),
    ("reserved1", np.uint16),
    ("Nrx", np.uint8),
    ("Ntx", np.uint8),
    ("rssiA", np.uint8),
    ("rssiB", np.uint8),
    ("rssiC", np.uint8),
    ("noise", np.int8),
    ("agc", np.uint8),
    ("antenna_sel", np.uint8),
    ("len", np.uint16),
    ("fake_rate_n_flags", np.uint16),
]).newbyteorder('<')

DTYPE_CSI_DATA_TLV = np.dtype(np.ubyte).newbyteorder('<')


@jit
def _read_bfee(byte):
    header = np.frombuffer(byte[0:DTYPE_CSI_HEADER_TLV.itemsize], dtype=DTYPE_CSI_HEADER_TLV)
    csiData = np.frombuffer(byte[DTYPE_CSI_HEADER_TLV.itemsize:], dtype=DTYPE_CSI_DATA_TLV)

    timestamp_low = header["timestamp_low"][0]
    bfee_count = header["bfee_count"][0]
    Nrx = header["Nrx"][0]
    Ntx = header["Ntx"][0]
    rssi = [header["rssiA"][0], header["rssiB"][0], header["rssiC"][0]]
    noise = header["noise"][0]
    agc = header["agc"][0]
    antenna_sel = header["antenna_sel"][0]
    length = header["len"][0]
    fake_rate_n_flags = header["fake_rate_n_flags"][0]
    csi = np.zeros([Ntx, Nrx, 30], dtype=np.complex)
    index = 0
    calc_len = int((30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8)
    flg = True

    # Check that length matches what it should
    if length != calc_len:
        print("MIMOToolbox:read_bfee_new:size", "Wrong beamforming matrix size.")
        flg = False

    for i in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx * Ntx):
            idx = np.int(index / 8)
            ptrR = np.int8((csiData[idx] >> remainder) | (csiData[idx + 1] << (8 - remainder)))

            ptrI = np.int8((csiData[idx + 1] >> remainder) | (csiData[idx + 2] << (8 - remainder)))
            csi[j % Ntx, int(j / Ntx), i] = np.complex(ptrR, ptrI)
            index += 16

    # Compute the permutation array
    perm = np.array([(antenna_sel & 0x3) + 1, ((antenna_sel >> 2) & 0x3) + 1, ((antenna_sel >> 4) & 0x3) + 1])

    #    return timestamp_low, Nrx, Ntx, rssi, np.reshape(csi,[Ntx, Nrx, 30]), perm
    return timestamp_low, np.reshape(csi, [Ntx, Nrx,
                                           30]), rssi, perm, bfee_count, noise, agc, fake_rate_n_flags, Nrx, Ntx, flg


@jit
def read_bf_file(filename, debug=False):
    # filename=r'/mnt/poplin/2016/ohara/data/matsulab_chair/raw/201607111524.dat'

    # Open file
    f = open(filename, 'rb')
    f.seek(0, 2)
    length = f.tell()

    f.seek(0, 0)

    # Initialize variables
    # ret = cell(ceil(len/95),1)     % Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
    timestamp_lowlist = np.array([])
    rssilist = np.empty([0, 3])
    csilist = []
    permlist = []
    bfee_countlist = []
    noiselist = []
    agclist = []
    fake_rate_n_flagslist = []
    skip_list = []
    cur = 0  # Current offset into file
    count = 0  # Number of records output
    percent = 0.0  # Number of records output
    broken_perm = 0  # Flag marking whether we've encountered a broken CSI yet
    triangle = [1, 3, 6]  # What perm should sum to for 1,2,3 antennas

    # Process all entries in file
    # Need 3 bytes -- 2 byte size field and 1 byte code
    while cur < (length - 3):
        count += 1
        if debug:
            print("current: " + str(cur) + " / " + str(length) + ", len(csilist): " + str(len(csilist)))
        # Read size and code
        # field_len = struct.unpack('>H', f.read(2))[0]
        field_len = np.frombuffer(f.read(2), dtype=DTYPE_LENGTH_TLV)["length"][0]
        code = ord(f.read(1))
        cur += 3

        # If unhandled code, skip (seek over) the record and continue
        if code == 187:  # get beamforming or phy data
            byte = f.read(field_len - 1)
            cur = cur + field_len - 1
            if len(byte) != field_len - 1:
                f.close()
                return
        else:  # skip all other info
            if debug:
                print("skip: " + str(count) + " code=: " + str(code))
            skip_list.append(count - 1)
            f.seek(field_len - 1, 1)
            cur = cur + field_len - 1
            continue

        if code == 187:  # hex2dec('bb')) Beamforming matrix -- output a record

            timestamp_low, csi, rssi, perm, bfee_count, noise, agc, fake_rate_n_flags, nrx, ntx, flg = _read_bfee(
                byte)  # for python
            if debug:
                print("csi.shape: " + str(csi.shape))
            if not flg:
                continue
            if nrx != csi.shape[1]:
                print('Nrx is not match')
                continue
            if ntx != csi.shape[0]:
                print('Ntx is not match')
                continue
            rssi_a = rssi[0]
            rssi_b = rssi[1]
            rssi_c = rssi[2]
            # timestamp_low, Nrx, Ntx, rssi_a, rssi_b, rssi_c, perm, csi =csitool.read_bfee(byte) # for C++

            # if nrx == 1:  # No permuting needed for only 1 antenna
            #     continue

            if sum(perm) != triangle[nrx - 1]:  # matrix does not contain default values
                if broken_perm == 0:
                    broken_perm = 1
                    print('WARN ONCE: Found CSI (' + filename + ') with Nrx=' + str(
                        nrx) + ' and invalid perm=' + str(perm))
                    #
                    #                else:
                    #                   csi[:,perm[0:Nrx]-1,:] = csi[:,0:Nrx,:]

            timestamp_lowlist = np.append(timestamp_lowlist, timestamp_low)
            rssilist = np.concatenate((rssilist, np.reshape(np.array([rssi_a, rssi_b, rssi_c]), [-1, 3])))
            csilist.append(csi)

            permlist.append(perm)
            bfee_countlist.append(bfee_count)
            noiselist.append(noise)
            agclist.append(agc)
            fake_rate_n_flagslist.append(fake_rate_n_flags)

            if cur > (length - 3) * percent:
                print(str(cur) + 'bytes / ' + str(length) + 'bytes')
                print(str(len(csilist)) + 'packets')
                percent += 0.1

    # Close file
    f.close()

    return timestamp_lowlist, csilist, rssilist, permlist, bfee_countlist, noiselist, agclist, fake_rate_n_flagslist, skip_list

@jit
def _read_bfee(byte):
    header = np.frombuffer(byte[0:DTYPE_CSI_HEADER_TLV.itemsize], dtype=DTYPE_CSI_HEADER_TLV)
    csiData = np.frombuffer(byte[DTYPE_CSI_HEADER_TLV.itemsize:], dtype=DTYPE_CSI_DATA_TLV)

    timestamp_low = header["timestamp_low"][0]
    bfee_count = header["bfee_count"][0]
    Nrx = header["Nrx"][0]
    Ntx = header["Ntx"][0]
    rssi = [header["rssiA"][0], header["rssiB"][0], header["rssiC"][0]]
    noise = header["noise"][0]
    agc = header["agc"][0]
    antenna_sel = header["antenna_sel"][0]
    length = header["len"][0]
    fake_rate_n_flags = header["fake_rate_n_flags"][0]
    csi = np.zeros([Ntx, Nrx, 30], dtype=np.complex)
    index = 0
    calc_len = int((30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8)
    flg = True

    # Check that length matches what it should
    if length != calc_len:
        print("MIMOToolbox:read_bfee_new:size", "Wrong beamforming matrix size.")
        flg = False

    for i in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx * Ntx):
            idx = np.int(index / 8)
            ptrR = np.int8((csiData[idx] >> remainder) | (csiData[idx + 1] << (8 - remainder)))

            ptrI = np.int8((csiData[idx + 1] >> remainder) | (csiData[idx + 2] << (8 - remainder)))
            csi[j % Ntx, int(j / Ntx), i] = np.complex(ptrR, ptrI)
            index += 16

    # Compute the permutation array
    perm = np.array([(antenna_sel & 0x3) + 1, ((antenna_sel >> 2) & 0x3) + 1, ((antenna_sel >> 4) & 0x3) + 1])

    #    return timestamp_low, Nrx, Ntx, rssi, np.reshape(csi,[Ntx, Nrx, 30]), perm
    return timestamp_low, np.reshape(csi, [Ntx, Nrx,
                                           30]), rssi, perm, bfee_count, noise, agc, fake_rate_n_flags, Nrx, Ntx, flg

def _read_bfee_multi(byte_list):
    return [_read_bfee(byte) for byte in byte_list]


def _read_bytes(filename):
    # Open file
    f = open(filename, 'rb')
    f.seek(0, 2)
    length = f.tell()

    f.seek(0, 0)

    # Initialize variables
    # ret = cell(ceil(len/95),1)     % Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
    skip_list = []
    cur = 0  # Current offset into file
    count = 0  # Number of records output
    percent = 0.0  # Number of records output
    broken_perm = 0  # Flag marking whether we've encountered a broken CSI yet
    triangle = [1, 3, 6]  # What perm should sum to for 1,2,3 antennas

    byte_list = []

    # Process all entries in file
    # Need 3 bytes -- 2 byte size field and 1 byte code
    while cur < (length - 3):
        count += 1
        # Read size and code
        field_len = np.frombuffer(f.read(2), dtype=DTYPE_LENGTH_TLV)["length"][0]
        code = ord(f.read(1))
        cur += 3

        # If unhandled code, skip (seek over) the record and continue
        if code == 187:  # get beamforming or phy data
            byte = f.read(field_len - 1)
            byte_list.append(byte)
            cur = cur + field_len - 1
            if len(byte) != field_len - 1:
                f.close()
                return
        else:  # skip all other info
            skip_list.append(count - 1)
            f.seek(field_len - 1, 1)
            cur = cur + field_len - 1
            continue

        if cur > (length - 3) * percent:
            print(str(cur) + 'bytes / ' + str(length) + 'bytes')
            print(str(len(byte_list)) + 'packets')
            percent += 0.1

    # Close file
    f.close()

    return byte_list, skip_list


def read_bf_file_multi(filename):
    byte_list, skip_list = _read_bytes(filename)

    per_one_task = 100
    ret = Parallel(n_jobs=-1, verbose=10)(
        [delayed(_read_bfee_multi)(byte_list[i:i + per_one_task]) for i in range(0, len(byte_list), per_one_task)])

    timestamp_lowlist, csilist, rssilist, permlist, bfee_countlist, noiselist, agclist, fake_rate_n_flagslist, Nrx, Ntx, flg = np.concatenate(
        np.array([np.array(r, dtype=object) for r in ret])).T

    return list(timestamp_lowlist), list(csilist), list(rssilist), list(permlist), list(bfee_countlist), list(
        noiselist), list(agclist), list(fake_rate_n_flagslist), skip_list
