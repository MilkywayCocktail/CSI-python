import pycsi
import numpy as np
import random
import matplotlib.pyplot as plt
import pyWidar2


class MyConfigsSimu(pycsi.MyConfigs):
    def __init__(self,
                 center_freq=5.32, bandwidth=20, length=100, sampling_rate=1000,
                 xlim=(-2, 2), ylim=(1, 3), vlim=(-3, 3), vrange=np.arange(-3.01, 3.01, 0.01)):

        super(MyConfigsSimu, self).__init__(center_freq=center_freq, bandwidth=bandwidth, sampling_rate=sampling_rate)
        self.length = length    # in seconds
        self.sampling_rate = sampling_rate  # in Hertz
        self.render_ticks = self.length * self.sampling_rate
        self.render_interval = 1. / self.sampling_rate
        self.render_indices = np.arange(self.render_ticks)
        self.xlim = xlim
        self.ylim = ylim
        self.vlim = vlim
        self.vrange = vrange


class Subject:

    def __init__(self, configs: MyConfigsSimu):
        self.state = None
        self.configs = configs
        self.state = self.__gen_state__()
        self.pre_rendered = False

    def __gen_state__(self):
        state = {'t': np.arange(0, self.configs.length, self.configs.render_interval),
                 'x': np.zeros(self.configs.render_ticks),
                 'y': np.zeros(self.configs.render_ticks),
                 'vx': np.array([np.nan] * self.configs.render_ticks),
                 'vy': np.array([np.nan] * self.configs.render_ticks)
                 }
        return state

    def set_init_location(self, x, y):
        self.state['x'][0] = x
        self.state['y'][0] = y

    def random_velocity(self, velocity='vx', num_points=3, order=3, vrange=None):
        print("Generating", num_points, "points for", velocity, '...', end='')
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
        print("Setting sine velocity for", velocity, '...', end='')
        period_ratio = 2 * np.pi / (self.configs.sampling_rate * period)
        x = self.configs.render_indices * period_ratio
        y = np.sin(x) * 1.5
        self.state[velocity] = y
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

    def generate_trajectory(self, bound_limit=True):
        print("Updating trajectory...", end='')
        for step in range(self.configs.render_ticks):

            if step == 0:
                if bound_limit is True:
                    self.bound_check(step)
                continue
            self.state['x'][step] = self.state['x'][step - 1] + self.state['vx'][
                step - 1] * self.configs.render_interval
            self.state['y'][step] = self.state['y'][step - 1] + self.state['vy'][
                step - 1] * self.configs.render_interval
            if bound_limit is True:
                self.bound_check(step)
        self.pre_rendered = True
        print('Done')

    def plot_velocity(self):
        print("Plotting velocities...")
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
        plt.suptitle("Ground Truth of Velocity")
        plt.tight_layout()
        plt.show()

    def plot_trajectory(self):
        if self.pre_rendered is False:
            print("Please generate the trajectory first!")
            return
        else:
            print("Printing trajectory...")
            plt.figure()
            plt.xlim((self.configs.xlim[0] - 0.5, self.configs.xlim[1] + 0.5))
            plt.ylim((self.configs.ylim[0] - 0.5, self.configs.ylim[1] + 0.5))
            for tick in range(self.configs.render_ticks):
                if tick % 100 == 0:
                    _color = (self.state['vx'][tick] / 6 + 0.5, self.state['vy'][tick] / 6 + 0.5, 0.2)
                    if tick == 0:
                        plt.scatter(self.state['x'][tick], self.state['y'][tick], marker='+', color=_color)
                    elif tick == self.configs.render_ticks - 1:
                        plt.scatter(self.state['x'][tick], self.state['y'][tick], marker='X', color=_color)
                    else:
                        plt.scatter(self.state['x'][tick], self.state['y'][tick], color=_color)
            plt.title("Trajectory of the subject")
            plt.tight_layout()
            plt.show()


class SensingZone:

    def __init__(self,  configs=MyConfigsSimu(), tx_pos=(-0.15, 0), rx_pos=(0.15, 0),):
        self.tx_pos = tx_pos
        self.rx_pos = rx_pos
        self.inbound = True
        self.subjects = []
        self.configs = configs
        self.csi = self.__gen_baseband__()
        self.temp_TS = 0
        self.temp_RS = 0
        self.temp_csi_dfs = np.exp(0.j)

    def add_subject(self, subject: Subject):
        self.subjects.append(subject)

    def __gen_baseband__(self):
        return np.zeros((self.configs.render_ticks, self.configs.nsub, self.configs.nrx, self.configs.ntx), dtype=complex)

    def __gen_phase__(self, subject, tick):
        TS = [subject.state['x'][tick] - self.tx_pos[0], subject.state['y'][tick] - self.tx_pos[1]]
        RS = [subject.state['x'][tick] - self.rx_pos[0], subject.state['y'][tick] - self.rx_pos[1]]

        AOA = RS[0] / np.linalg.norm(RS)    # in sine
        csi_aoa = np.squeeze(np.exp((-2.j * np.pi * self.configs.dist_antenna * AOA * self.configs.subfreq_list /
                                       self.configs.lightspeed).dot(self.configs.antenna_list.reshape(1, -1))))
        csi_aoa = csi_aoa[:, :, np.newaxis].repeat(self.configs.ntx, axis=2)

        TOF = (np.linalg.norm(TS) + np.linalg.norm(RS)) / self.configs.lightspeed   # in seconds
        csi_tof = np.squeeze(np.exp(-2.j * np.pi * self.configs.subfreq_list * TOF))
        csi_tof = csi_tof[:, np.newaxis, np.newaxis].repeat(self.configs.nrx, axis=1).repeat(self.configs.ntx, axis=2)

        DFS = (np.linalg.norm(TS) + np.linalg.norm(RS) -
               np.linalg.norm(self.temp_TS) - np.linalg.norm(self.temp_RS)) / self.configs.render_interval  # in m/s
        csi_dfs = np.squeeze(np.exp(-2.j * np.pi * self.configs.subfreq_list * DFS / self.configs.lightspeed *
                             self.configs.render_interval))
        csi_dfs = self.temp_csi_dfs * csi_dfs[:, np.newaxis, np.newaxis].repeat(
                    self.configs.nrx, axis=1).repeat(self.configs.ntx, axis=2)

        csi = np.exp(1.j * np.zeros((self.configs.nsub, self.configs.nrx, self.configs.ntx))) * \
            csi_aoa * csi_tof * csi_dfs

        self.temp_phase = csi_dfs
        self.temp_TS = TS
        self.temp_RS = RS
        return csi

    def collect(self):
        print("Collecting simulation data...")

        for tick in range(self.configs.render_ticks):
            if (tick + 1) % 100 == 0:
                print("\r\033[32mCollecting ticks {} / {}\033[0m".format(tick, self.configs.render_ticks), end='')
            for subject in self.subjects:
                csi = self.__gen_phase__(subject, tick)
                self.csi[tick] += csi
        print('\n')

    def derive_MyCsi(self, configs, name):

        _csi = pycsi.MyCsi(configs, name)
        _csi.load_lists(amp=np.abs(self.csi),
                        phase=np.angle(self.csi),
                        timelist=self.configs.render_indices / self.configs.sampling_rate)
        return _csi


if __name__ == '__main__':
    config = MyConfigsSimu(length=100)
    sub1 = Subject(config)
    sub1.random_velocity(velocity='vx', num_points=15, vrange=(-1, 1, 0.01))
    sub1.sine_velocity(velocity='vy', period=10)

    sub1.plot_velocity()
    sub1.generate_trajectory()
    sub1.plot_trajectory()

    zone = SensingZone(config)
    zone.add_subject(sub1)
    zone.collect()

    simu = zone.derive_MyCsi(config, '0310GT2')
    #simu.save_csi()

    #simu = pycsi.MyCsi(config, '0310GT0')
    #simu.load_lists(path='../npsave/0310/0310GT0-csis.npy')
    #simu.aoa_by_music()
    #simu.viewer.view(autosave=True)
    #simu.tof_by_music()
    #simu.viewer.view(autosave=True)
    #simu.extract_dynamic(mode='running', subtract_mean=False)
    simu.doppler_by_music(window_length=100, stride=10, raw_window=True)
    simu.viewer.view(threshold=0.1, autosave=True)

    #conf = pyWidar2.MyConfigsW2(num_paths=1)
    #widar = pyWidar2.MyWidar2(conf, simu)
    #widar.run(dynamic_durations=False)
    #widar.plot_results()
