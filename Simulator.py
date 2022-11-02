import numpy as np
import matplotlib.pyplot as plt

class GroundTruth:

    def __init__(self, length, category):
        self.length = length
        self.category = category
        self.ground_truth = np.zeros(self.length)

    def random_points(self, num_points=3, constant=False):
        try:
            x = np.random.choice(self.length, num_points, replace=False)
            if self.category == 'aoa':
                y = np.random.choice(360, num_points)
            elif self.category == 'tof':
                y = np.random.choice(1, num_points) * 1.e-7
            elif self.category == 'doppler':
                y = np.random.choice(10., num_points) - 5

            if constant is True:
                self.ground_truth = np.ones(self.length) * y[0]
                print("Generated", self.category, "with constant value!")
            else:
                self.ground_truth[x] = y
                print("Generated", self.category, "with", num_points, "points!")
        except:
            print("Ground truth generation failed!")

    def set_points(self, *args):
        try:
            count = 0
            for x, y  in args:
                self.ground_truth[x] = y
                count += 1
            print("Set", count, "points!")
        except:
            print("Point set failed!")

    def interpolate(self):
        try:
            f = interpolate.interp1d(np.arange(self.length), self.ground_truth, 'linear')
            self.ground_truth = f()

    def show(self):
        try:
            plt.plot(np.arange(self.length), self.ground_truth)
            plt.title(str(self.category))
            plt.xlabel("#packet")
            plt.ylabel("Value")
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

