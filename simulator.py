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

    def set_constant(self, value=None):
        try:
            if value is None:
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
        self.span = np.arange(-90, 91, 1)
        self.ylim = (-91, 91)
        self.ylabel = "AoA / $deg$"


class _GTToF(GroundTruth):

    def __init__(self, *args, **kwargs):
        GroundTruth.__init__(self, *args, **kwargs)
        self.category = 'ToF'
        self.span = np.arange(-1.e-7, 4.e-7, 1.e-9)
        self.ylim = (-1.e-7, 4.e-7)
        self.ylabel = "ToF / $s$"


class _GTDoppler(GroundTruth):
    def __init__(self, *args, **kwargs):
        GroundTruth.__init__(self, *args, **kwargs)
        self.category = 'Doppler'
        self.span = np.arange(-5, 5.05, 0.05)
        self.ylim = (-5.05, 5.05)
        self.ylabel = "Doppler Velocity / $m/s$"


class DataSimulator:

    def __init__(self, configs:pycsi.MyConfigs, length=10000):
        self.configs = configs
        self.length = length
        self.amp = None
        self.phase = None
        self.csi = None
        self.timestamps = np.arange(0, self.length, 1.) / self.configs.sampling_rate

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def add_baseband(self):
        try:
            self.amp = np.ones((self.length, self.configs.nsub, self.configs.nrx, self.configs.ntx))
            self.timestamps = np.arange(0, self.length) / self.configs.sampling_rate
            self.phase = np.zeros((self.length, self.configs.nsub, self.configs.nrx, self.configs.ntx))
            self.csi = self.amp * np.exp(1.j * self.phase)
            print("Baseband established!")
        except:
            print("Failed to establish baseband.")

    def add_noise(self, snr=80):
        try:
            snr = 10 ** (snr/10.0)
            signal_p = np.sum(self.amp ** 2) / self.length
            noise_p = signal_p / snr
            noise = np.random.randn(self.length, self.configs.nsub, self.configs.nrx, self.configs.ntx) * np.sqrt(noise_p)
            self.amp += np.abs(noise)
            print("White noise added!")
        except:
            print("Failed to add noise.")

    def add_ipo(self, p1=None, p2=None):
        try:
            if p1 is None:
                p1 = 2 * np.pi * (np.random.randn() - 0.5)
            if p2 is None:
                p2 = 2 * np.pi * (np.random.randn() - 0.5)
            self.phase[:, :, 1, :] += p1
            self.phase[:, :, 2, :] += p2
            print("IPO added!")
        except:
            print("Failed to add IPO.")

    def apply_gt(self, *args):

        def apply_AoA():

            antenna_list = np.arange(0, self.configs.nrx, 1.).reshape(1, -1)
            for i, gt in enumerate(ground_truth.y):
                frame = np.squeeze(np.exp((-2.j * np.pi * self.configs.dist_antenna * np.sin(
                    gt * np.pi / 180) * self.configs.subfreq_list / self.configs.lightspeed).dot(antenna_list)))
                csi[i] = frame[:, :, np.newaxis].repeat(self.configs.ntx, axis=2)

            return csi

        def apply_ToF():
            for i, gt in enumerate(ground_truth.y):
                frame = np.squeeze(np.exp(-2.j * np.pi * self.configs.subfreq_list * gt))
                csi[i] = frame[:, np.newaxis, np.newaxis].repeat(self.configs.nrx, axis=1).repeat(self.configs.ntx, axis=2)

            return csi

        def apply_Doppler():

            for i, gt in enumerate(ground_truth.y):

                frame = np.squeeze(np.exp(-2.j * np.pi * self.configs.subfreq_list *
                                          gt / self.configs.lightspeed / self.configs.sampling_rate))

                if i > 0:
                    csi[i] = csi[i - 1] * frame[:, np.newaxis, np.newaxis].repeat(
                        self.configs.nrx, axis=1).repeat(self.configs.ntx, axis=2)
                else:
                    csi[i] = frame[:, np.newaxis, np.newaxis].repeat(
                        self.configs.nrx, axis=1).repeat(self.configs.ntx, axis=2)

            return csi

        csi = np.ones((self.length, self.configs.nsub, self.configs.nrx, self.configs.ntx)) * (0 + 0.j)

        for ground_truth in args:
            if not isinstance(ground_truth, GroundTruth):
                print("Please first generate the GroundTruth object!")

            elif isinstance(ground_truth, GroundTruth) and ground_truth.length != self.length:
                print("Length of ground truth", ground_truth.length, "does not match length", self.length, "!")

            else:
                csi = eval('apply_' + ground_truth.category + '()')
                self.amp += np.abs(csi)
                self.phase += np.angle(csi)

    def derive_MyCsi(self, configs, name):

        _csi = pycsi.MyCsi(configs, name)
        _csi.load_lists(amp=self.amp, phase=self.phase, timelist=self.timestamps)
        return _csi


if __name__ == '__main__':

    ipo1 = -2.33
    ipo2 = 3.10

    gt1 = GroundTruth(length=10000).tof
    gt1.random_points(7)
    gt1.interpolate()
    gt1.show()

    #gt2 = GroundTruth(length=10000).aoa
    #gt2.set_constant()
    #gt2.interpolate()
    # gt2.show()

    configs = pycsi.MyConfigs(center_freq=5.32, bandwidth=20)
    data = DataSimulator(configs, length=10000)
    data.add_baseband()
    print(data.amp.shape)
    #data.add_noise()
    data.apply_gt(gt1)
    #data.add_ipo(ipo1, ipo2)

    simu = data.derive_MyCsi(configs, '0314GT3')
    #plt.plot(np.unwrap(simu.data.phase[:,0,:,0], axis=0))
    #plt.title("Phase with IPO")
    #plt.show()
    #simu.view_phase_diff()
    #simu.extract_dynamic()
    simu.tof_by_music()
    simu.viewer.view()
    #simu.save_csi('0126G00')
    #simu.doppler_by_music(raw_window=False)
    #simu.viewer.view(threshold=-4.4)

#    for i, spectrum in enumerate(simu.data.spectrum):
#        simu.data.view_spectrum(sid=i, autosave=True, folder_name='GT3', notion='_' + str(i))

