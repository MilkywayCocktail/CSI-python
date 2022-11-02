import numpy as np
import random
import matplotlib.pyplot as plt


class GroundTruth:

    def __init__(self, category, length=10000):
        self.length = length
        self.category = category
        self.x = np.arange(self.length)
        self.y = [np.nan] * self.length

    def random_points(self, num_points=3):

        try:
            x = random.sample(self.x.tolist(), num_points)
            if self.category == 'aoa':
                y = random.choices(np.arange(-180, 180, 1), k=num_points)
            elif self.category == 'tof':
                y = random.choices(np.arange(0, 1.e-7, 5.e-10), k=num_points)
            elif self.category == 'doppler':
                y = random.choices(np.arange(-5, 5, 0.05), k=num_points)

            for (index, value) in zip(x, y):
                self.y[index] = value

            print("Generated", self.category, "with", num_points, "points!")

        except:
            print("Ground truth generation failed!")

    def set_points(self, pointlist):
        try:
            count = 0
            for x, y in pointlist:
                if self.category == 'aoa':
                    y = y % 360
                self.y[x] = y
                count += 1
            print("Set", count, "points!")
        except:
            print("Point set failed!")

    def set_constant(self, value, rd=True):
        try:
            if rd is True:
                if self.category == 'aoa':
                    y = random.choice(np.arange(-180, 180, 1))
                elif self.category == 'tof':
                    y = random.choice(np.arange(0, 1.e-7, 5.e-10))
                elif self.category == 'doppler':
                    y = random.choice(np.arange(-5, 5, 0.05))

                self.y = np.ones(self.length) * y

            else:
                self.y = np.ones(self.length) * value

            print("Generated", self.category, "with constant value!")
        except:
            print("Constant value set failed!")

    def interpolate(self):
        try:
            x = self.x[~np.isnan(self.y)]
            y = np.ma.masked_array(self.y, mask=np.isnan(self.y))
            curve = np.polyfit(x, y[x], 3)
            self.y = np.polyval(curve, self.x)

            print("Interpolation completed!")
        except:
            print("Interpolation failed!")

    def show(self):
        try:
            if self.category == 'aoa':
                plt.ylim((-181, 181))
                plt.ylabel("AoA / $deg$")
            elif self.category == 'tof':
                plt.ylim((-5.e-8, 15.e-8))
                plt.ylabel("ToF / $s$")
            elif self.category == 'doppler':
                plt.ylim((-5.05, 5.05))
                plt.ylabel("Doppler Velocity / $m/s$")

            plt.plot(self.x, self.y)
            plt.title(str(self.category))
            plt.xlabel("#packet")
            plt.show()
        except:
            print("Plot failed!")

class DataSimulator:

    def __init__(self):
        self.nrx = 3
        self.ntx = 1
        self.nsub = 30
        self.center_freq = 5.67e+09
        self.lightspeed = 299792458
        self.dist_antenna = 0.0264
        self.bandwidth = 40e+06
        self.delta_subfreq = 3.125e+05
        self.sampling_rate = 3000
        self.length = 10000
        self.amp = None
        self.phase = None
        self.timestamps = None

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def add_baseband(self):
        try:
            self.amp = np.ones((self.length, self.nsub, self.nrx, self.ntx))
            self.phase = np.zeros((self.length, self.nsub, self.nrx, self.ntx))
            self.timestamps = np.arange(0, self.length) / self.sampling_rate
            print("Baseband established!")
        except:
            print("Failed to establish baseband.")

    def add_noise(self, snr=80):
        try:
            snr = 10 ** (snr/10.0)
            signal_p = np.sum(self.amp ** 2) / self.length
            noise_p = signal_p / snr
            noise = np.random.randn(self.length, self.nsub, self.nrx, self.ntx) * np.sqrt(noise_p)
            self.amp += np.abs(noise)
            print("White noise added!")
        except:
            print("Failed to add noise.")


    def apply_aoa(self, ground_truth):
        try:
            antenna_list = np.arange(0, self.nrx, 1.).reshape(-1, 1)
            frame = np.exp(-2.j * np.pi * antenna_list * self.dist_antenna *
                           np.sin(ground_truth * np.pi / 180) * self.center_freq / self.lightspeed)
            csi = frame[np.newaxis, :, np.newaxis].repeat(
                self.length, axis=0).repeat(self.nsub, axis=1).repeat(self.ntx, axis=3)
            self.amp += np.abs(csi)
            self.phase += np.angle(csi)
            print("AoA added! GT=", ground_truth)
        except:
            print("Failed to add AoA.")

    def apply_tof(self, ground_truth):
        try:
            subcarrier_list = np.arange(-58, 62, 4)
            frame = np.exp([-2.j * self.delta_subfreq * subcarrier_list * ground_truth]).reshape(1, -1)
            csi = frame[np.newaxis, :, np.newaxis].repeat(
                self.length, axis=0).repeat(self.nrx, axis=2).repeat(self.ntx, axis=3)
            self.amp += np.abs(csi)
            self.phase += np.angle(csi)
            print("ToF added! GT=", ground_truth)
        except:
            print("Failed to add ToF.")


if __name__ == '__main__':

    gt = GroundTruth('doppler')
    gt.random_points(30)
    gt.interpolate()
    gt.show()