import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import csi_loader
import pycsi


class MyCsiW2(pycsi.MyCsi):
    def self_calibrate(self, ref_antenna=None):
        """
        Overrode MyCsi.self_calibrate.\n
        Calculates weighted multiplication of csi and its conjugate.\n
        :param ref_antenna: No use.
        """
        recon = self.commonfunc.reconstruct_csi
        csi_ratio = np.mean(self.amp, axis=0) / np.std(self.amp, axis=0)
        ant_ratio = np.mean(csi_ratio, axis=2)
        ref = np.argmax(ant_ratio)

        alpha = np.min(self.amp, axis=0)
        beta = np.sum(alpha) ** 2 / self.length * 1000

        csi_ref = recon(self.amp[:, :, ref] + beta, self.phase[:, :, ref])[:, :, np.newaxis]
        csi_cal = recon(self.amp - alpha, self.phase) * np.repeat(csi_ref.conj(), 3, axis=2)

        self.amp = np.abs(csi_cal)
        self.phase = np.angle(csi_cal)


class MyWidar2:
    def __init__(self, configs: pycsi.MyConfigs, csi: MyCsiW2):
        self.configs = configs
        self.csi = csi
        self.antenna_list = np.arange(0, self.nrx, 1.).reshape(-1, 1)
        self.taulist = np.arange(-100., 400., 1.) * (10. ** -9)
        self.thetalist = np.deg2rad(np.arange(-0., 180., 1.))
        self.dopplerlist = np.arange(-5., 5., 0.01)
        self.window_length = 100
        self.stride = 100
        self.num_paths = 5
        self.steer_tof, self.steer_aoa, self.steer_doppler = self.__gen_steering_vector__()

    def __gen_steering_vector__(self):
        sampling_rate = self.configs.sampling_rate
        subfreqs = self.configs.subfreq_list
        dist_antenna = self.configs.dist_antenna
        antennas = self.antenna_list
        center_freq = self.configs.center_freq
        lightspeed = self.configs.lightspeed

        dt_list = self.taulist[::-1].reshape(-1, 1)
        theta_list = self.thetalist[::-1].reshape(-1, 1)
        velocity_list = self.dopplerlist[::-1].reshape(-1, 1)
        delays = np.arange(0, self.window_length, 1.).reshape(-1, 1) / sampling_rate

        tof_vector = np.exp(-1.j * 2 * np.pi * dt_list.dot(subfreqs.T))
        aoa_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * np.sin(theta_list).dot(
            antennas.T) * center_freq / lightspeed)
        doppler_vector = np.exp(-1.j * 2 * np.pi * center_freq * velocity_list.dot(
            delays.T) / lightspeed)

        return tof_vector, aoa_vector, doppler_vector

    def sage_algorithm(self):
        estimates = np.empty((4, self.num_paths), dtype=complex)

    def sage(self):
        total_steps = (self.csi.length - self.window_length) // self.stride
        estimates = np.empty((4, self.num_paths, 0))

        for i in range(total_steps):
            self.sage_algorithm()

    def run(self):
        if self.configs.ntx > 1:
            self.csi.amp = self.csi.amp[..., 0]
            self.csi.phase = self.csi.phase[..., 0]

        self.csi.self_calibrate()

        self.sage()