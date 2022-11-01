# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import numpy as np

light_speed = 299792458.0  # [m/s]

subcarrier_index_20MHz_56 = list(range(-28, 0)) + list(range(1, 29))
subcarrier_index_20MHz_30 = [-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                             1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28]
subcarrier_index_20MHz_16 = [-28, -24, -20, -16, -12, -8, -4, -1, 1, 5, 9, 13, 17, 21, 25, 28]
subcarrier_index_40MHz_114 = list(range(-58, 0)) + list(range(1, 59))
subcarrier_index_40MHz_58 = [-58, -56, -54, -52, -50, -48, -46, -44, -42, -40, -38, -36, -34, -32, -30, -28, -26, -24,
                             -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
                             24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]
subcarrier_index_40MHz_30 = [-58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2,
                             2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]

# Subcarrier frequency spacing [Hz]
delta_f = 312.5 * np.power(10, 3)


def channel2freqency(channel, band_width=None):
    """
    convert channel to center freqency
    :param channel: int
    :param band_width: HT20|HT40-|HT40+, HT40+:center_freq+=10MHz, HT40-:center_freq+=-10MHz
    :return: center frequency [Hz]
    """
    channel = int(channel)

    if 1 <= channel and channel <= 13:
        freq = (2.412 * np.power(10.0, 9)) + ((channel - 1) * 5 * np.power(10.0, 6))
    elif ((36 <= channel and channel <= 64) or (100 <= channel and channel <= 140)) and channel % 4 == 0:
        freq = (5.180 * np.power(10.0, 9)) + ((channel - 36) * 5 * np.power(10.0, 6))
    else:
        print("channel " + str(channel) + " is invalid")
        return None

    if band_width == "HT40+":
        freq += 10. * (10 ** 6)
    elif band_width == "HT40-":
        freq -= 10. * (10 ** 6)

    return freq


def get_subcarrier_freq(center_freq, band_width=None, num_subcarrier=30):
    """
    get freqency list of subcarriers
    :param center_freq:
    :param band_width:
    :param num_subcarrier:
    :return:freqency list of subcarriers [Hz]
    """
    if band_width is None:
        if center_freq < 5.0 * np.power(10.0, 9):
            band_width = 20
        else:
            band_width = 40

    str_subcarrier_index = "subcarrier_index_" + str(band_width) + "MHz_" + str(num_subcarrier)
    if str_subcarrier_index in globals().keys():
        subcarrier_index = globals()[str_subcarrier_index]
    else:
        print(str_subcarrier_index + " is invalid")
        subcarrier_index = [0]

    return np.array([center_freq + i * delta_f for i in subcarrier_index])


class sm_matrices():
    """
    https://github.com/dhalperi/linux-80211n-csitool-supplementary/blob/master/matlab/sm_matrices.m
    """
    sm_1 = 1

    sm_2_20 = np.array([[1., 1.], [1., - 1.]]) / np.sqrt(2)
    sm_2_40 = np.array([[1., 1.j], [1.j, 1.]]) / np.sqrt(2)

    sm_3_20 = np.array([[-2 * np.pi / 16, - 2 * np.pi / (80 / 33), 2 * np.pi / (80 / 3)],
                        [2 * np.pi / (80 / 23), 2 * np.pi / (48 / 13), 2 * np.pi / (240 / 13)],
                        [- 2 * np.pi / (80 / 13), 2 * np.pi / (240 / 37), 2 * np.pi / (48 / 13)]], dtype=float)
    sm_3_20 = np.exp(1.j * sm_3_20) / np.sqrt(3)

    sm_3_40 = np.array([[-2 * np.pi / 16, - 2 * np.pi / (80 / 13), 2 * np.pi / (80 / 23)],
                        [- 2 * np.pi / (80 / 37), - 2 * np.pi / (48 / 11), - 2 * np.pi / (240 / 107)],
                        [2 * np.pi / (80 / 7), - 2 * np.pi / (240 / 83), - 2 * np.pi / (48 / 11)]], dtype=float)
    sm_3_40 = np.exp(1.j * sm_3_40) / np.sqrt(3)


if __name__ == "__main__":
    channel = range(1, 14) + range(36, 140, 4)
    for c in channel:
        freq = channel2freqency(c)
        sub_freq = get_subcarrier_freq(freq)
        print(str(c) + ": " + str(freq) + "Hz " + str(sub_freq))
