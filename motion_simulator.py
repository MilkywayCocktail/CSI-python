import pycsi
import numpy as np
import random
import matplotlib.pyplot as plt
import pyWidar2


class MyConfigsSimu(pycsi.MyConfigs):
    """
    Configurations used in motion_simulator scripts.
    """
    def __init__(self,
                 center_freq=5.32, bandwidth=20, sampling_rate=1000,
                 length=100, xlim=(-2, 2), ylim=(1, 3), vlim=(-3, 3)):

        super(MyConfigsSimu, self).__init__(center_freq=center_freq, bandwidth=bandwidth, sampling_rate=sampling_rate)
        self.length = length    # in seconds
        self.sampling_rate = sampling_rate  # in Hertz
        self.render_ticks = self.length * self.sampling_rate
        self.render_interval = 1. / self.sampling_rate
        self.render_indices = np.arange(self.render_ticks)
        self.xlim = xlim
        self.ylim = ylim
        self.vlim = vlim
        self.vrange = np.arange(vlim[0], vlim[1] + 0.01, 0.01)


class GroundTruth:
    """
    Ground Truth of movements of a subject.
    """
    def __init__(self, configs: MyConfigsSimu, name):
        self.name = name
        self.configs = configs
        self.TS = np.zeros((self.configs.render_ticks, 2))
        self.RS = np.zeros((self.configs.render_ticks, 2))
        self.AoA = np.zeros(self.configs.render_ticks)
        self.ToF = np.zeros(self.configs.render_ticks)
        self.DFS = np.zeros(self.configs.render_ticks)
        self.AMP = np.ones(self.configs.render_ticks)
        self.temp_csi_dfs = np.exp(0.j)

    def plot_groundtruth(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        plt.suptitle(self.name + '_GroundTruth')
        axs = axs.flatten()

        axs[0].plot(self.ToF)
        axs[1].plot(self.AoA)
        axs[2].plot(self.DFS)
        axs[3].plot(self.AMP)

        axs[0].set_title("ToF")
        axs[0].set_ylim(-1.e-7, 4.e-7)
        axs[1].set_title("AoA")
        axs[1].set_ylim(-90, 90)
        axs[2].set_title("Doppler")
        axs[3].set_title("Amplitude")

        for axi in axs:
            axi.set_xlim(0, self.configs.render_ticks)
            axi.grid()

        plt.tight_layout()
        plt.show()


class Subject:
    """
    Subject that moves in the sensing zone.
    """

    def __init__(self, configs: MyConfigsSimu, name=''):
        self.name = name
        self.state = None
        self.configs = configs
        self.state = self.__gen_state__()
        self.pre_rendered = False
        self.inzone = False
        self.inbound = True
        self.groundtruth = None

    def __gen_state__(self):
        state = {'t': np.arange(0, self.configs.length, self.configs.render_interval),
                 'x': np.zeros(self.configs.render_ticks),
                 'y': np.zeros(self.configs.render_ticks),
                 'vx': np.array([np.nan] * self.configs.render_ticks),
                 'vy': np.array([np.nan] * self.configs.render_ticks)
                 }
        return state

    def __gen_phase__(self, tick, tx_pos, rx_pos):
        TS = [self.state['x'][tick] - tx_pos[0], self.state['y'][tick] - tx_pos[1]]
        RS = [self.state['x'][tick] - rx_pos[0], self.state['y'][tick] - rx_pos[1]]

        AoA = RS[0] / np.linalg.norm(RS)  # in sine
        csi_aoa = np.squeeze(np.exp((-2.j * np.pi * self.configs.dist_antenna * AoA * self.configs.subfreq_list /
                                     self.configs.lightspeed).dot(self.configs.antenna_list.reshape(1, -1))))
        csi_aoa = csi_aoa[:, :, np.newaxis].repeat(self.configs.ntx, axis=2)

        ToF = (np.linalg.norm(TS) + np.linalg.norm(RS)) / self.configs.lightspeed  # in seconds
        csi_tof = np.squeeze(np.exp(-1.j * np.pi * self.configs.subfreq_list * ToF))
        csi_tof = csi_tof[:, np.newaxis, np.newaxis].repeat(self.configs.nrx, axis=1).repeat(self.configs.ntx, axis=2)

        if tick == 0:
            DFS = 0
        else:
            DFS = (np.linalg.norm(TS) + np.linalg.norm(RS) -
                   np.linalg.norm(self.groundtruth.TS[tick - 1]) - np.linalg.norm(
                        self.groundtruth.RS[tick - 1])) / self.configs.render_interval  # in m/s

        csi_dfs = np.squeeze(np.exp(-1.j * np.pi * self.configs.subfreq_list * DFS / self.configs.lightspeed *
                                    self.configs.render_interval))
        csi_dfs = self.groundtruth.temp_csi_dfs * csi_dfs[:, np.newaxis, np.newaxis].repeat(
            self.configs.nrx, axis=1).repeat(self.configs.ntx, axis=2)

        csi = np.exp(1.j * np.zeros((self.configs.nsub, self.configs.nrx, self.configs.ntx))) * \
            csi_aoa * csi_tof * csi_dfs

        self.groundtruth.TS[tick] = TS
        self.groundtruth.RS[tick] = RS
        self.groundtruth.AoA[tick] = np.rad2deg(np.arcsin(AoA))
        self.groundtruth.ToF[tick] = ToF
        self.groundtruth.DFS[tick] = DFS
        self.groundtruth.temp_csi_dfs = csi_dfs

        return csi

    def set_init_location(self, x, y):
        self.state['x'][0] = x
        self.state['y'][0] = y

    def random_velocity(self, velocity='vx', num_points=3, order=3, vrange=None):
        print(self.name, "Generating", num_points, "points for", velocity, '...', end='')
        vr = self.configs.vrange if vrange is None else vrange

        x = random.sample(self.configs.render_indices.tolist(), num_points)
        y = random.choices(vr, k=num_points)
        for (index, value) in zip(x, y):
            self.state[velocity][index:index + self.configs.sampling_rate] = (value,)
        print('Done')

        _x = np.argwhere(np.isnan(self.state[velocity]))
        self.state[velocity][_x[:, 0]] = (0,)

        '''
        print("Interpolating", velocity, '...', end='')
        x = self.configs.render_indices[~np.isnan(self.state[velocity])]
        y = np.ma.masked_array(self.state[velocity], mask=np.isnan(self.state[velocity]))
        curve = np.polyfit(x, y[x], order)
        self.state[velocity] = np.polyval(curve, self.configs.render_indices)

        self.state[velocity][self.state[velocity] > vr[-1]] = vr[-1]
        self.state[velocity][self.state[velocity] < vr[0]] = vr[0]
        print('Done')
        '''

    def sine_velocity(self, velocity='vx', period=5):
        print(self.name, "Setting sine velocity for", velocity, '...', end='')
        period_ratio = 2 * np.pi / (self.configs.sampling_rate * period)
        x = self.configs.render_indices * period_ratio
        y = np.sin(x)
        self.state[velocity] = y
        print('Done')

    def constant_velocity(self, velocity='vx', value=0):
        print(self.name, "Setting constant velocity for", velocity, '...', end='')
        self.state[velocity][:] = (value,)
        print('Done')

    def bound_check(self, ind):
        self.state['x'][ind] = self.configs.xlim[0] if self.state['x'][ind] < self.configs.xlim[0] else self.state['x'][
            ind]
        self.state['x'][ind] = self.configs.xlim[1] if self.state['x'][ind] > self.configs.xlim[1] else self.state['x'][
            ind]
        self.state['y'][ind] = self.configs.ylim[0] if self.state['y'][ind] < self.configs.ylim[0] else self.state['y'][
            ind]
        self.state['y'][ind] = self.configs.ylim[1] if self.state['y'][ind] > self.configs.ylim[1] else self.state['y'][
            ind]

    def generate_trajectory(self):
        print(self.name, "Updating trajectory...", end='')
        for step in range(self.configs.render_ticks):
            if step == 0:
                if self.inbound is True:
                    self.bound_check(step)
                continue
            self.state['x'][step] = self.state['x'][step - 1] + self.state['vx'][
                step - 1] * self.configs.render_interval
            self.state['y'][step] = self.state['y'][step - 1] + self.state['vy'][
                step - 1] * self.configs.render_interval
            if self.inbound is True:
                self.bound_check(step)
        self.pre_rendered = True
        print('Done')

    def circle_trajectory(self, direction='clk', period=5, center=(0, 1)):
        print(self.name, "Setting circle velocity...", end='')
        r = np.linalg.norm([self.state['x'][0] - center[0], self.state['y'][0] - center[1]])
        v_abs = v_abs = 2 * np.pi * r / period
        for step in range(self.configs.render_ticks):

            SO = [self.state['x'][step] - center[0], self.state['y'][step] - center[1]]

            if direction == 'clk':
                v = [SO[1], -SO[0]] / r
            elif direction == 'aclk':
                v = [-SO[1], SO[0]] / r
            v = v * v_abs

            self.state['vx'][step] = v[0]
            self.state['vy'][step] = v[1]

            if step > 0:
                self.state['x'][step] = self.state['x'][step - 1] + self.state['vx'][
                    step - 1] * self.configs.render_interval
                self.state['y'][step] = self.state['y'][step - 1] + self.state['vy'][
                    step - 1] * self.configs.render_interval

            if self.inbound is True:
                self.bound_check(step)

        print('Done')

    def plot_velocity(self):
        print(self.name, "Plotting velocities...")
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.configs.render_indices, self.state['vx'])
        ax1.set_ylim(self.configs.vlim)
        ax1.set_ylabel('X Velocity / $m/s$')
        ax1.set_xlabel("#tick")
        ax1.grid()
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(self.configs.render_indices, self.state['vy'])
        ax2.set_ylim(self.configs.vlim)
        ax2.set_ylabel('Y Velocity / $m/s$')
        ax2.set_xlabel("#tick")
        ax2.grid()
        plt.suptitle("Velocity of " + self.name)
        plt.tight_layout()
        plt.show()

    def plot_trajectory(self):
        if self.pre_rendered is False:
            print(self.name, "Please generate the trajectory first!")
            return
        else:
            print(self.name, "Printing trajectory...")
            plt.figure()
            if self.inbound is True:
                plt.xlim((self.configs.xlim[0] - 0.5, self.configs.xlim[1] + 0.5))
                plt.ylim((self.configs.ylim[0] - 0.5, self.configs.ylim[1] + 0.5))
            else:
                plt.xlim((np.min(self.state['x']) - 0.5, np.max(self.state['x'] + 0.5)))
                plt.ylim((np.min(self.state['y']) - 0.5, np.max(self.state['y'] + 0.5)))
            for tick in range(self.configs.render_ticks):
                if tick % 100 == 0:
                    _color = (self.state['vx'][tick] / 6 + 0.5, self.state['vy'][tick] / 6 + 0.5, 0.2)
                    if tick == 0:
                        plt.scatter(self.state['x'][tick], self.state['y'][tick], marker='+', color=_color)
                    elif tick == self.configs.render_ticks - 1:
                        plt.scatter(self.state['x'][tick], self.state['y'][tick], marker='X', color=_color)
                    else:
                        plt.scatter(self.state['x'][tick], self.state['y'][tick], color=_color)
            plt.title("Trajectory of " + self.name)
            plt.grid()
            plt.tight_layout()
            plt.show()


class SensingZone:
    """
    Sensing zone that contains subjects and signals.
    """

    def __init__(self,  configs=MyConfigsSimu(), tx_pos=(-0.15, 0), rx_pos=(0.15, 0), inbound=True):
        self.tx_pos = tx_pos
        self.rx_pos = rx_pos
        self.inbound = inbound
        self.subjects = []
        self.configs = configs
        self.csi = self.__gen_baseband__()

    def __gen_baseband__(self):
        return np.zeros((self.configs.render_ticks, self.configs.nsub, self.configs.nrx, self.configs.ntx), dtype=complex)

    def add_subject(self, subject: Subject):
        subject.groundtruth = GroundTruth(self.configs, name=subject.name)
        subject.inzone = True
        subject.inbound = self.inbound
        self.subjects.append(subject)

    def collect(self):
        print("Collecting simulation data...")

        for tick in range(self.configs.render_ticks):
            if (tick + 1) % 100 == 0:
                print("\r\033[32mCollecting ticks {} / {}\033[0m".format(tick, self.configs.render_ticks), end='')
            for subject in self.subjects:
                if subject.pre_rendered is True:
                    csi = subject.__gen_phase__(tick, self.tx_pos, self.rx_pos)
                    self.csi[tick] += csi
        print('\n')

    def show_groundtruth(self):
        for subject in self.subjects:
            subject.groundtruth.plot_groundtruth()

    def derive_MyCsi(self, configs, name):

        _csi = pycsi.MyCsi(configs, name)
        _csi.load_lists(amp=np.abs(self.csi),
                        phase=np.angle(self.csi),
                        timelist=self.configs.render_indices / self.configs.sampling_rate)
        return _csi


if __name__ == '__main__':
    config = MyConfigsSimu(length=100)
    sub1 = Subject(config, 'sub1')
    #sub1.random_velocity(velocity='vx', num_points=15, vrange=(-1, 1, 0.01))
    #sub1.constant_velocity('vx')
    #sub1.sine_velocity(velocity='vy', period=50)
    sub1.set_init_location(1, 2)
    sub1.circle_trajectory(period=5, center=(1, 3))
    sub1.plot_velocity()

    zone = SensingZone(config, inbound=False)
    zone.add_subject(sub1)

    sub1.generate_trajectory()
    sub1.plot_trajectory()

    zone.collect()
    zone.show_groundtruth()

    simu = zone.derive_MyCsi(config, '0314GT4')
    #simu.save_csi()

    #simu = pycsi.MyCsi(config, '0310GT0')
    #simu.load_lists(path='../npsave/0310/0310GT0-csis.npy')
    #simu.aoa_by_music()
    #simu.viewer.view(autosave=True)
    #simu.tof_by_music()
    #simu.viewer.view(autosave=True)
    #simu.extract_dynamic(mode='running', subtract_mean=False)
    #simu.doppler_by_music(window_length=100, stride=10, raw_window=True)
    #simu.viewer.view(threshold=0.1, autosave=True)

    ##conf = pyWidar2.MyConfigsW2(num_paths=1)
    #widar = pyWidar2.MyWidar2(conf, simu)
    #widar.run(dynamic_durations=False)
    #widar.plot_results()
