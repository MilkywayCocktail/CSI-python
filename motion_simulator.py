import pycsi
import numpy as np
import random
import matplotlib.pyplot as plt


class GroundTruth:

    def __init__(self, length=100, sampling_rate=1000):
        self.length = length    # in seconds
        self.sampling_rate = sampling_rate  # in Hertz
        self.render_ticks = self.length * self.sampling_rate
        self.render_interval = 1. / self.sampling_rate
        self.state = self.__gen_state__()
        self.trajectory = np.zeros((self.render_ticks, 2), dtype=float)
        self.render_indices = np.arange(self.render_ticks)
        self.xlim = (-2, 2)
        self.ylim = (1, 3)
        self.vlim = (-3.01, 3.01)
        self.vrange = np.arange(-3.01, 3.01, 0.01)

    def __gen_state__(self):
        state = {'t': np.arange(0, self.length, self.render_interval),
                 'x': np.zeros(self.render_ticks),
                 'y': np.zeros(self.render_ticks),
                 'vx': [np.nan] * self.render_ticks,
                 'vy': [np.nan] * self.render_ticks
                 }
        return state

    def set_init_location(self, x, y):
        self.state['x'][0] = x
        self.state['y'][0] = y

    def random_velocity(self, velocity='vx', num_points=3):
        print("Generating", num_points, "points for", velocity, '...', end='')
        x = random.sample(self.render_indices.tolist(), num_points)
        y = random.choices(self.vrange, k=num_points)
        for (index, value) in zip(x, y):
            self.state[velocity][index] = value
        print('Done')

    def interpolate_velocity(self, velocity='vx',  order=3):
        print("Interpolating", velocity, '...', end='')
        x = self.render_indices[~np.isnan(self.state[velocity])]
        y = np.ma.masked_array(self.state[velocity], mask=np.isnan(self.state[velocity]))
        curve = np.polyfit(x, y[x], order)
        self.state[velocity] = np.polyval(curve, self.render_indices)

        self.state[velocity][self.state[velocity] > self.vlim[-1]] = self.vlim[-1]
        self.state[velocity][self.state[velocity] < self.vlim[0]] = self.vlim[0]
        print('Done')

    def bound_check(self, ind):
        self.state['x'][ind] = self.xlim[0] if self.state['x'][ind] < self.xlim[0] else self.state['x'][ind]
        self.state['x'][ind] = self.xlim[1] if self.state['x'][ind] > self.xlim[1] else self.state['x'][ind]
        self.state['y'][ind] = self.ylim[0] if self.state['y'][ind] < self.ylim[0] else self.state['y'][ind]
        self.state['y'][ind] = self.ylim[1] if self.state['y'][ind] > self.ylim[1] else self.state['y'][ind]

    def update_trajectory(self, bound_limit=True):

        for step in range(self.render_ticks):
            print("\r\033[32mRendering ticks {} / {}\033[0m".format(step, self.render_ticks), end='')
            if step == 0:
                if bound_limit is True:
                    self.bound_check(step)
                continue
            self.state['x'][step] = self.state['x'][step - 1] + self.state['vx'][step - 1] * self.render_interval
            self.state['y'][step] = self.state['y'][step - 1] + self.state['vy'][step - 1] * self.render_interval
            if bound_limit is True:
                self.bound_check(step)

    def plot_velocity(self):
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.render_indices, self.state['vx'])
        ax1.set_ylim(self.vlim)
        ax1.set_ylabel('X Velocity / $m/s$')
        ax1.set_xlabel("#tick")
        ax1.grid()
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(self.render_indices, self.state['vy'])
        ax2.set_ylim(self.vlim)
        ax2.set_ylabel('Y Velocity / $m/s$')
        ax2.set_xlabel("#tick")
        ax2.grid()
        plt.suptitle("Ground Truth of Velocity")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    gt = GroundTruth()
    gt.random_velocity(velocity='vx', num_points=10)
    gt.random_velocity(velocity='vy', num_points=15)
    gt.interpolate_velocity(velocity='vx')
    gt.interpolate_velocity(velocity='vy')
    gt.plot_velocity()
