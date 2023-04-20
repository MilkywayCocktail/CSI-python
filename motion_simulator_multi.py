import pycsi
import numpy as np
import random
import matplotlib.pyplot as plt
import pyWidar2

class MyConfigsSimuMulti(pycsi.MyConfigs):
    """
    Configurations used in motion_simulator scripts.
    """
    def __init__(self,
                 center_freq=5.32, bandwidth=20, sampling_rate=1000,
                 length=100, xlim=(-2, 2), ylim=(1, 3), vlim=(-3, 3)):

        super(MyConfigsSimuMulti, self).__init__(center_freq=center_freq, bandwidth=bandwidth, sampling_rate=sampling_rate)
        self.length = length    # in seconds
        self.sampling_rate = sampling_rate  # in Hertz
        self.render_ticks = self.length * self.sampling_rate
        self.render_interval = 1. / self.sampling_rate
        self.render_indices = np.arange(self.render_ticks)
        self.xlim = xlim
        self.ylim = ylim
        self.vlim = vlim
        self.vrange = np.arange(vlim[0], vlim[1] + 0.01, 0.01)

        self.tx_list = np.arange(self.ntx)
        self.rx_list = np.arange(self.nrx)

class GroundTruthMulti:
    """
    Ground Truth of movements of a subject considering multiple tx/rx antennas.
    """

    def __init__(self, configs: MyConfigsSimuMulti, name, num_rx, num_tx):
        self.name = name
        self.configs = configs
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_links = num_tx * num_rx
        self.TS = np.zeros((self.configs.render_ticks, num_tx, 2))
        self.RS = np.zeros((self.configs.render_ticks, num_rx, 2))
        self.AoA = np.zeros((self.configs.render_ticks, num_rx))
        self.AoD = np.zeros((self.configs.render_ticks, num_tx))
        self.ToF = np.zeros((self.configs.render_ticks, num_rx, num_tx))
        self.DFS = np.zeros((self.configs.render_ticks, num_rx, num_tx))
        self.AMP = np.ones((self.configs.render_ticks, num_rx, num_tx))
        self.temp_csi_dfs = np.ones((num_rx, num_tx)) * np.exp(0.j)

    def __gen_phase__(self, tick, sub_pos, tx_pos, rx_pos):

        tx_pos = np.array(tx_pos)
        rx_pos = np.array(rx_pos)
        TS = np.zeros((self.num_tx, 2))
        RS = np.zeros((self.num_rx, 2))
        DFS = np.zeros((self.num_rx, self.num_tx))
        AoA = np.zeros(self.num_rx)
        AoD = np.zeros(self.num_tx)
        ToF1 = np.zeros(self.num_tx)
        ToF2 = np.zeros(self.num_rx)

        csi_aod = np.zeros(self.num_tx)
        csi_aoa = np.zeros(self.num_rx)
        csi_tof1 = np.zeros(self.num_tx)
        csi_tof2 = np.zeros(self.num_rx)
        csi_multi = np.zeros((self.configs.nsub, self.configs.nrx, self.configs.nrx))

        for tx in range(self.num_tx):
            TS[tx, :] = sub_pos - tx_pos[tx, :]
            AoD[tx] = TS[tx, 0] / np.linalg.norm(TS[tx])  # in sine
            ToF1[tx] = np.linalg.norm(TS[tx]) / self.configs.lightspeed  # in seconds
            csi_aod[tx] = np.squeeze(
                np.exp((-2.j * np.pi * self.configs.dist_antenna * AoD[tx] * self.configs.subfreq_list /
                        self.configs.lightspeed).dot(self.configs.tx_list.reshape(1, -1))))
            csi_tof1[tx] = np.squeeze(np.exp(-2.j * np.pi * self.configs.subfreq_list * ToF1[tx]))

        for rx in range(self.num_rx):
            RS[rx, :] = sub_pos - rx_pos[rx, :]
            AoA[rx] = RS[rx, 0] / np.linalg.norm(RS[rx])  # in sine
            ToF2[tx] = np.linalg.norm(TS[rx]) / self.configs.lightspeed  # in seconds
            csi_aoa[rx] = np.squeeze(
                np.exp((-2.j * np.pi * self.configs.dist_antenna * AoA[rx] * self.configs.subfreq_list /
                        self.configs.lightspeed).dot(self.configs.rx_list.reshape(1, -1))))
            csi_tof2[tx] = np.squeeze(np.exp(-2.j * np.pi * self.configs.subfreq_list * ToF2[tx]))

        for rx in range(self.num_rx):
            for tx in range(self.num_tx):
                self.ToF[tick, rx, tx] = ToF1[tx] + ToF2[rx]

                csi_base = 0.3 * np.exp(1.j * np.zeros((self.configs.nsub, self.configs.nrx, self.configs.ntx)))
                csi_base *= (csi_tof1[tx] + csi_tof2[rx])[:, np.newaxis, np.newaxis].repeat(
                    self.configs.nrx, axis=1).repeat(self.configs.ntx, axis=2)
                csi_multi[:, :, tx] = csi_base[:, :, tx] * csi_aod[tx]
                if tick == 0:
                    DFS = np.zeros((self.num_rx, self.num_tx))
                else:
                    DFS[rx, tx] = (np.linalg.norm(TS[tx]) + np.linalg.norm(RS[rx]) -
                                   np.linalg.norm(self.TS[tick - 1]) - np.linalg.norm(
                                self.RS[tick - 1])) / self.configs.render_interval  # in m/s

        self.TS[tick] = TS
        self.RS[tick] = RS
        self.AoA[tick] = np.rad2deg(np.arcsin(AoA))
        self.AoD[tick] = np.rad2deg(np.arcsin(AoD))
        self.DFS[tick] = DFS

        return

    def plot_groundtruth(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        plt.suptitle(self.name + '_GroundTruth')
        axs = axs.flatten()

        axs[0].plot(self.ToF)
        axs[1].plot(self.AoA)
        axs[2].plot(self.DFS)
        axs[3].plot(self.AMP)

        axs[0].set_title("ToF")
        axs[0].set_ylim(np.min(self.ToF), np.max(self.ToF))
        axs[1].set_title("AoA")
        axs[1].set_ylim(-90, 90)
        axs[2].set_title("Doppler")
        axs[3].set_title("Amplitude")

        for axi in axs:
            axi.set_xlim(0, self.configs.render_ticks)
            axi.grid()

        plt.tight_layout()
        plt.show()
