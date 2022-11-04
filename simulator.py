import pycsi
import numpy as np
import random
import matplotlib.pyplot as plt


class GroundTruth:

    def __init__(self, length=10000):
        self.length = length
        self.x = np.arange(self.length).tolist()
        self.y = [np.nan] * self.length
        self.category = None
        self.span = None
        self.ylim = None
        self.ylabel = None

    @property
    def aoa(self):
        return _GTAoA(length=self.length)

    @property
    def tof(self):
        return _GTToF(length=self.length)

    @property
    def doppler(self):
        return _GTDoppler(length=self.length)

    def random_points(self, num_points=3):

        try:
            x = random.sample(self.x, num_points)
            y = random.choices(self.span, k=num_points)
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
                    y = (y + 180) % 360 - 180
                self.y[x] = y
                count += 1
            print("Set", count, "points!")
        except:
            print("Point set failed!")

    def set_constant(self, value, rd=True):
        try:
            if rd is True:
                y = random.choice(self.span)
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

            self.y[self.y > self.span[-1]] = self.span[-1]
            self.y[self.y < self.span[0]] = self.span[0]

            print("Interpolation completed!")
        except:
            print("Interpolation failed!")

    def show(self):
        try:

            plt.plot(self.x, self.y)
            plt.ylim(self.ylim)
            plt.ylabel(self.ylabel)
            plt.xlabel("#packet")
            plt.title("Ground Truth of " + str(self.category))
            plt.show()
        except:
            print("Plot failed!")


class _GTDoppler(GroundTruth):
    def __init__(self, *args, **kwargs):
        GroundTruth.__init__(self, *args, **kwargs)
        self.category = 'Doppler'
        self.span = np.arange(-5, 5.05, 0.05)
        self.ylim = (-5.05, 5.05)
        self.ylabel = "Doppler Velocity / $m/s$"


class _GTAoA(GroundTruth):

    def __init__(self, *args, **kwargs):
        GroundTruth.__init__(self, *args, **kwargs)
        self.category = 'AoA'
        self.span = np.arange(-180, 181, 1)
        self.ylim = (-181, 181)
        self.ylabel = "AoA / $deg$"


class _GTToF(GroundTruth):

    def __init__(self, *args, **kwargs):
        GroundTruth.__init__(self, *args, **kwargs)
        self.category = 'ToF'
        self.span = np.arange(0, 1.e-7, 5.e-10)
        self.ylim = (-5.e-8, 15.e-8)
        self.ylabel = ("ToF / $s$")


class DataSimulator:

    def __init__(self, length=10000, sampling_rate=3000):
        self.nrx = 3
        self.ntx = 1
        self.nsub = 30
        self.center_freq = 5.67e+09
        self.lightspeed = 299792458
        self.dist_antenna = 0.0264
        self.bandwidth = 40e+06
        self.delta_subfreq = 3.125e+05
        self.length = length
        self.sampling_rate = sampling_rate
        self.timestamps = np.arange(0, self.length, self.sampling_rate).tolist()
        self.amp = None
        self.phase = None
        self.timestamps = None

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def phase_add(self, phases):
        self.phase = np.rad2deg(np.angle(np.exp(1.j * phases) * np.exp(1.j * self.phase)))

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

        antenna_list = np.arange(0, self.nrx, 1.).reshape(-1, 1)

        if ground_truth.length != self.length:
            print("Length of ground truth", ground_truth.length, "does not match length", self.length, "!")

        else:
            if isinstance(ground_truth, GroundTruth):
                csi = np.ones((self.length, self.nsub, self.nrx, self.ntx)) * (0 + 0.j)
                for i, gt in enumerate(ground_truth.y):
                    frame = np.exp(-2.j * np.pi * antenna_list * self.dist_antenna *
                                   np.sin(gt * np.pi / 180) * self.center_freq / self.lightspeed)
                    csi[i] = frame[np.newaxis, :].repeat(self.nsub, axis=0).repeat(self.ntx, axis=2)

            else:
                frame = np.exp(-2.j * np.pi * antenna_list * self.dist_antenna *
                               np.sin(ground_truth * np.pi / 180) * self.center_freq / self.lightspeed)
                csi = frame[np.newaxis, :, np.newaxis].repeat(
                    self.length, axis=0).repeat(self.nsub, axis=1).repeat(self.ntx, axis=3)

            self.amp += np.abs(csi)
            self.phase_add(csi)

        print("AoA added!")
        #print("Failed to add AoA.")

    def apply_tof(self, ground_truth):
        try:
            subcarrier_list = np.arange(-58, 62, 4)
            frame = np.exp([-2.j * self.delta_subfreq * subcarrier_list * ground_truth]).reshape(1, -1)
            csi = frame[np.newaxis, :, np.newaxis].repeat(
                self.length, axis=0).repeat(self.nrx, axis=2).repeat(self.ntx, axis=3)
            self.amp += np.abs(csi)
            self.phase_add(csi)
            print("ToF added! GT=", ground_truth)
        except:
            print("Failed to add ToF.")

    def apply_doppler(self, ground_truth):
        try:
            subcarrier_list = np.arange(-58, 62, 4)
            frame = np.exp([-2.j * self.delta_subfreq * subcarrier_list * ground_truth]).reshape(1, -1)
            csi = frame[np.newaxis, :, np.newaxis].repeat(
                self.length, axis=0).repeat(self.nrx, axis=2).repeat(self.ntx, axis=3)
            self.amp += np.abs(csi)
            self.phase_add(csi)
            print("ToF added! GT=", ground_truth)
        except:
            print("Failed to add ToF.")

    def derive_MyCsi(self, name):

        _csi = pycsi.MyCsi(name)
        _csi.load_lists(amp=self.amp, phase=self.phase, timelist=self.timestamps)
        return _csi


if __name__ == '__main__':

    gt = GroundTruth().aoa
    gt.set_constant(-45, rd=False)
    gt.show()

    gt2 = GroundTruth().aoa
    gt2.random_points(10)
    gt2.interpolate()
    gt2.show()

    data = DataSimulator()
    data.add_baseband()

    data.apply_aoa(gt2)

    simu = data.derive_MyCsi('MySimu')
    simu.aoa_by_music()
    simu.data.view_spectrum()