import pycsi
import numpy as np
import random
import matplotlib.pyplot as plt


class GroundTruth:

    def __init__(self, length=10000):
        self.length = length
        self.x = np.arange(self.length)
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
            x = random.sample(self.x.tolist(), num_points)
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

    def set_constant(self, value=0, rd=True):
        try:
            if rd is True:
                y = random.choice(self.span)
                self.y = np.ones(self.length) * y

            else:
                self.y = np.ones(self.length) * value

            print("Generated", self.category, "with constant value!")
        except:
            print("Constant value set failed!")

    def interpolate(self, order=3):

        x = self.x[~np.isnan(self.y)]
        y = np.ma.masked_array(self.y, mask=np.isnan(self.y))
        curve = np.polyfit(x, y[x], order)
        self.y = np.polyval(curve, self.x)

        self.y[self.y > self.span[-1]] = self.span[-1]
        self.y[self.y < self.span[0]] = self.span[0]

        print("Interpolation completed!")

        # print("Interpolation failed!")

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
        self.ylim = (0, 1.e-7)
        self.ylabel = "ToF / $s$"


class _GTDoppler(GroundTruth):
    def __init__(self, *args, **kwargs):
        GroundTruth.__init__(self, *args, **kwargs)
        self.category = 'Doppler'
        self.span = np.arange(-5, 5.05, 0.05)
        self.ylim = (-5.05, 5.05)
        self.ylabel = "Doppler Velocity / $m/s$"


class DataSimulator:

    def __init__(self, length=10000, sampling_rate=3000):
        self.nrx = 3
        self.ntx = 1
        self.nsub = 30
        self.center_freq = 5.67e+09
        self.lightspeed = 299792458
        self.dist_antenna = self.lightspeed / self.center_freq / 2.
        self.bandwidth = 40e+06
        self.delta_subfreq = 3.125e+05
        self.length = length
        self.sampling_rate = sampling_rate
        self.amp = None
        self.phase = None
        self.csi = None
        self.timestamps = np.arange(0, self.length, 1.) / self.sampling_rate
        self.subfreq_list = np.arange(self.center_freq - 58 * self.delta_subfreq,
                                      self.center_freq + 62 * self.delta_subfreq,
                                      4 * self.delta_subfreq).reshape(-1, 1)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def add_baseband(self):
        try:
            self.amp = np.ones((self.length, self.nsub, self.nrx, self.ntx))
            self.timestamps = np.arange(0, self.length) / self.sampling_rate
            self.phase = np.zeros((self.length, self.nsub, self.nrx, self.ntx))
            self.csi = self.amp * np.exp(1.j * self.phase)
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

    def apply_gt(self, ground_truth):

        def apply_AoA():

            antenna_list = np.arange(0, self.nrx, 1.).reshape(1, -1)
            for i, gt in enumerate(ground_truth.y):
                frame = np.squeeze(np.exp((-2.j * np.pi * self.dist_antenna * np.sin(
                    gt * np.pi / 180) * self.subfreq_list / self.lightspeed).dot(antenna_list)))
                csi[i] = frame[:, :, np.newaxis].repeat(self.ntx, axis=2)

            return csi

        def apply_ToF():
            for i, gt in enumerate(ground_truth.y):
                frame = np.squeeze(np.exp(-2.j * np.pi * self.subfreq_list * gt))
                csi[i] = frame[:, np.newaxis, np.newaxis].repeat(self.nrx, axis=1).repeat(self.ntx, axis=2)

            return csi

        def apply_Doppler():

            for i, gt in enumerate(ground_truth.y):

                frame = np.squeeze(np.exp(-2.j * np.pi * self.subfreq_list *
                                          gt / self.lightspeed / self.sampling_rate))

                if i > 0:
                    csi[i] = csi[i - 1] * frame[:, np.newaxis, np.newaxis].repeat(
                        self.nrx, axis=1).repeat(self.ntx, axis=2)
                else:
                    csi[i] = frame[:, np.newaxis, np.newaxis].repeat(
                        self.nrx, axis=1).repeat(self.ntx, axis=2)

            return csi

        csi = np.ones((self.length, self.nsub, self.nrx, self.ntx)) * (0 + 0.j)

        if not isinstance(ground_truth, GroundTruth):
            print("Please first generate the ground truth!")

        elif isinstance(ground_truth, GroundTruth) and ground_truth.length != self.length:
            print("Length of ground truth", ground_truth.length, "does not match length", self.length, "!")

        else:
            csi = eval('apply_' + ground_truth.category + '()')
            self.amp += np.abs(csi)
            self.phase += np.angle(csi)

    def derive_MyCsi(self, name):

        _csi = pycsi.MyCsi(name)
        _csi.load_lists(amp=self.amp, phase=self.phase, timelist=self.timestamps)
        return _csi


if __name__ == '__main__':

    gt1 = GroundTruth(length=10000).doppler
    gt1.random_points(10)
    gt1.interpolate(5)
    gt1.show()

#    gt2 = GroundTruth(length=1000).doppler
#    gt2.random_points(3)
#    gt2.interpolate()
#    gt2.show()

    data = DataSimulator(length=10000)
    data.add_baseband()
    #data.add_noise()
    data.apply_gt(gt1)
    #    data.apply_gt(gt2)

    simu = data.derive_MyCsi('GT11')
    plt.plot(np.unwrap(simu.data.phase[:,0,:,0], axis=0))
    plt.show()
    #simu.data.view_phase_diff()
    simu.doppler_by_music(window_length=100, stride=100, raw_timestamps=False, raw_window=False)
    simu.data.view_spectrum(threshold=10)

#    for i, spectrum in enumerate(simu.data.spectrum):
#        simu.data.view_spectrum(sid=i, autosave=True, notion='_' + str(i))

