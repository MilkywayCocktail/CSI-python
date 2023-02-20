import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import csi_loader
import pycsi


class MyConfigsW2(pycsi.MyConfigs):
    def __init__(self,
                 center_freq=5.32, bandwidth=20, sampling_rate=1000,
                 window_length=100, stride=100, num_paths=5):

        super(MyConfigsW2, self).__init__(center_freq=center_freq, bandwidth=bandwidth, sampling_rate=sampling_rate)
        self.antenna_list = np.arange(0, self.nrx, 1.).reshape(-1, 1)
        self.toflist = np.arange(-1.e-7, 4.e-7, 1.e-9).reshape(-1, 1)
        self.aoalist = np.deg2rad(np.arange(-0, 180, 1.)).reshape(-1, 1)
        self.dopplerlist = np.arange(-5, 5, 0.01).reshape(-1, 1)
        self.window_length = window_length
        self.stride = stride
        self.num_paths = num_paths
        self.max_loop = 100
        self.update_ratio = 1.


class MyCsiW2(pycsi.MyCsi):
    def __init__(self, configs: MyConfigsW2, input_name='', path=None):
        """
        Changed configs type to MyConfigsW2.\n
        :param configs: MyConfigsW2 object
        :param input_name: name of CSI sample
        :param path: path of CSI
        """
        super(MyCsiW2, self).__init__(configs=configs, input_name=input_name, path=path)

    def self_calibrate(self, ref_antenna=None):
        """
        Overrode MyCsi.self_calibrate.\n
        Preprocessing of Widar2.\n
        :param ref_antenna: No use.
        """
        print(self.name, "self calibrating...", end='')
        recon = self.commonfunc.reconstruct_csi
        csi_ratio = np.mean(self.amp, axis=0) / np.std(self.amp, axis=0)
        ant_ratio = np.mean(csi_ratio, axis=0)
        ref = np.argmax(ant_ratio)

        alpha = np.min(self.amp, axis=0)
        beta = np.sum(alpha) ** 2 / self.length * 1000

        csi_ref = recon(self.amp[:, :, ref] + beta, self.phase[:, :, ref], squeeze=False)[:, :, np.newaxis]
        csi_cal = recon(self.amp - alpha, self.phase, squeeze=False) * np.repeat(csi_ref.conj(), 3, axis=2)

        self.amp = np.abs(csi_cal)
        self.phase = np.angle(csi_cal)

        print("Done")
        return csi_cal


class MyWidar2:
    def __init__(self, configs: MyConfigsW2, csi: MyCsiW2):
        self.configs = configs
        self.csi = csi
        self.total_steps = (self.csi.length - self.configs.window_length) // self.configs.stride
        self.steer_tof, self.steer_aoa, self.steer_doppler = self.__gen_steering_vector__()
        self.estimates = self.__gen_estimates__()
        self.arg_i = self.__gen_arg_indices__()
        self.temp_estimates, self.temp_arg_i = self.__gen_temp_parameters__()

    def __gen_steering_vector__(self):

        sampling_rate = self.configs.sampling_rate
        dist_antenna = self.configs.dist_antenna
        center_freq = self.configs.center_freq
        lightspeed = self.configs.lightspeed

        subfreqs = self.configs.subfreq_list
        antennas = self.configs.antenna_list
        delays = np.arange(0, self.configs.window_length, 1.).reshape(-1, 1) / sampling_rate
        toflist = self.configs.toflist.reshape(-1, 1)
        aoalist = self.configs.aoalist.reshape(-1, 1)
        dopplerlist = self.configs.dopplerlist.reshape(-1, 1)

        tof_vector = np.exp(-1.j * 2 * np.pi * subfreqs.dot(toflist.T))
        aoa_vector = np.exp(-1.j * 2 * np.pi * dist_antenna * antennas.dot(
            np.sin(aoalist.T)) * center_freq / lightspeed)
        doppler_vector = np.exp(-1.j * 2 * np.pi * center_freq * delays.dot(
            dopplerlist.T) / lightspeed)

        return tof_vector, aoa_vector, doppler_vector

    def __gen_estimates__(self):
        est = {'tof': np.zeros((self.total_steps, self.configs.num_paths), dtype=complex),
               'aoa': np.zeros((self.total_steps, self.configs.num_paths), dtype=complex),
               'doppler': np.zeros((self.total_steps, self.configs.num_paths), dtype=complex),
               'amplitude': np.zeros((self.total_steps, self.configs.num_paths), dtype=complex)
               }
        return est

    def __gen_arg_indices__(self):
        toflist = self.configs.toflist
        aoalist = self.configs.aoalist
        dopplerlist = self.configs.dopplerlist

        ini = {'tof': np.zeros((self.total_steps, self.configs.num_paths), dtype=int),
               'aoa': np.zeros((self.total_steps, self.configs.num_paths), dtype=int),
               'doppler': np.zeros((self.total_steps, self.configs.num_paths), dtype=int),
               'amplitude': np.zeros((self.total_steps, self.configs.num_paths), dtype=int)
               }
        ini['tof'][:] = int(np.round((0 - toflist[0]) / (toflist[1] - toflist[0])))
        ini['aoa'][:] = int(np.round((0 - aoalist[0]) / (aoalist[1] - aoalist[0])))
        ini['doppler'][:] = int(np.round((0 - dopplerlist[0]) / (dopplerlist[1] - dopplerlist[0])))
        return ini

    def __gen_temp_parameters__(self):
        _estimates = {'tof': np.zeros(self.configs.num_paths, dtype=complex),
                      'aoa': np.zeros(self.configs.num_paths, dtype=complex),
                      'doppler': np.zeros(self.configs.num_paths, dtype=complex),
                      'amplitude': np.zeros(self.configs.num_paths, dtype=complex),
                      }
        _arg_index = {'tof': np.zeros(self.configs.num_paths, dtype=int),
                      'aoa': np.zeros(self.configs.num_paths, dtype=int),
                      'doppler': np.zeros(self.configs.num_paths, dtype=int),
                      'amplitude': np.zeros(self.configs.num_paths, dtype=int)
                      }
        return _estimates, _arg_index

    def sage(self):
        recon = self.csi.commonfunc.reconstruct_csi
        r = self.configs.update_ratio
        stride = self.configs.stride
        window_length = self.configs.window_length
        nsub = self.csi.configs.nsub
        nrx = self.csi.configs.nrx

        csi_signal = recon(self.csi.amp, self.csi.phase)
        print("total steps=", self.total_steps)

        for step in range(self.total_steps):

            actual_csi = csi_signal[step * stride: step * stride + window_length]
            latent_signal = np.zeros((self.configs.window_length, self.configs.nsub, self.configs.nrx,
                                      self.configs.num_paths), dtype=complex)

            self.temp_estimates, self.temp_arg_i = self.__gen_temp_parameters__()
            for loop in range(self.configs.max_loop):
                print("\r\033[32mstep{} / loop{}\033[0m".format(step, loop), end='')
                for path in range(self.configs.num_paths):
                    noise_signal = actual_csi - np.sum(latent_signal, axis=3)
                    expect_signal = latent_signal[..., path] + r * noise_signal

                    # Estimation of tof
                    aoa_matrix = self.steer_aoa[:, self.arg_i['aoa'][0, path]].reshape(1, 1, -1)
                    doppler_matrix = self.steer_doppler[:, self.arg_i['doppler'][0, path]].reshape(-1, 1, 1)

                    coeff_matrix = expect_signal * np.conj(aoa_matrix) * np.conj(doppler_matrix)
                    coeff_vector = np.sum(coeff_matrix, axis=(0, 2)).reshape(-1, 1)
                    tof_object_vector = np.abs(np.sum(coeff_vector * np.conj(self.steer_tof), axis=0))
                    self.temp_arg_i['tof'][path] = np.argmax(tof_object_vector)
                    self.temp_estimates['tof'][path] = self.configs.toflist[self.temp_arg_i['tof'][path]]

                    # Estimation of aoa
                    tof_matrix = self.steer_tof[:, self.temp_arg_i['tof'][path]].reshape(1, -1, 1)

                    coeff_matrix = expect_signal * np.conj(doppler_matrix) * np.conj(tof_matrix)
                    coeff_vector = np.sum(coeff_matrix, axis=(0, 1)).reshape(-1, 1)
                    aoa_object_vector = np.abs(np.sum(coeff_vector * np.conj(self.steer_aoa), axis=0))
                    self.temp_arg_i['aoa'][path] = np.argmax(aoa_object_vector)
                    self.temp_estimates['aoa'][path] = self.configs.aoalist[self.temp_arg_i['aoa'][path]]

                    # Estimation of doppler
                    aoa_matrix = self.steer_aoa[:, self.temp_arg_i['aoa'][path]].reshape(1, 1, -1)

                    coeff_matrix = expect_signal * np.conj(aoa_matrix) * np.conj(tof_matrix)
                    coeff_vector = np.sum(coeff_matrix, axis=(1, 2)).reshape(-1, 1)
                    doppler_object_vector = np.abs(np.sum(coeff_vector * np.conj(self.steer_doppler), axis=0))
                    self.temp_arg_i['doppler'][path] = np.argmax(doppler_object_vector)
                    self.temp_estimates['doppler'][path] = self.configs.dopplerlist[self.temp_arg_i['doppler'][path]]

                    # Estimation of amplitude
                    doppler_matrix = self.steer_doppler[:, self.temp_arg_i['doppler'][path]].reshape(-1, 1, 1)
                    coeff_matrix = expect_signal * np.conj(aoa_matrix) * np.conj(tof_matrix) * np.conj(doppler_matrix)
                    self.temp_estimates['amplitude'][path] = np.sum(coeff_matrix) / (window_length * nsub * nrx)

                    # Update latent signal
                    tof_matrix = self.steer_tof[:, self.temp_arg_i['tof'][path]].reshape(1, -1, 1)
                    aoa_matrix = self.steer_aoa[:, self.temp_arg_i['aoa'][path]].reshape(1, 1, -1)
                    doppler_matrix = self.steer_doppler[:, self.temp_arg_i['doppler'][path]].reshape(-1, 1, 1)
                    latent_signal[..., path] = self.temp_estimates['amplitude'][path] * tof_matrix * aoa_matrix * doppler_matrix

                delta = [np.linalg.norm(param) for param in self.temp_estimates.values()]
                for key in self.estimates.keys():
                    self.estimates[key][step] = self.temp_estimates[key]
                    self.arg_i[key][step] = self.temp_arg_i[key]

                if delta[0] < 1.e-9 and delta[1] < np.pi / 180. and delta[2] < 0.01 and delta[3] < 1.e-9:
                    break

            residue_error = actual_csi - np.sum(latent_signal, axis=3)
            residue_error_ratio = np.mean(np.abs(residue_error)) / np.mean(np.abs(actual_csi))

    def run(self):
        start = time.time()
        if self.configs.ntx > 1:
            self.csi.amp = self.csi.amp[..., 0][..., np.newaxis]
            self.csi.phase = self.csi.phase[..., 0][..., np.newaxis]

        self.csi.self_calibrate()
        self.sage()

        end = time.time()
        print("\nTotal time:", end-start)

    def plot_results(self):
        fig, axs = plt.subplots(2, 2, figsize=(24, 15))
        axs = axs.flatten()

        axs[0].scatter(list(range(self.total_steps)) * self.configs.num_paths,
                       self.estimates['tof'].real.reshape(-1),
                       c=np.log(np.abs(self.estimates['amplitude'])).reshape(-1),
                       linewidths=0)
        axs[1].scatter(list(range(self.total_steps)) * self.configs.num_paths,
                       np.rad2deg(self.estimates['aoa'].real).reshape(-1),
                       c=np.log(self.estimates['amplitude']).reshape(-1), linewidths=0)
        axs[2].scatter(list(range(self.total_steps)) * self.configs.num_paths,
                       self.estimates['doppler'].real.reshape(-1),
                       c=np.log(np.abs(self.estimates['amplitude'])).reshape(-1),
                       linewidths=0)
        axs[3].scatter(list(range(self.total_steps)) * self.configs.num_paths,
                       np.abs(self.estimates['amplitude']).reshape(-1),
                       c=np.log(np.abs(self.estimates['amplitude'])).reshape(-1),
                       linewidths=0)

        axs[0].set_title("tof")
        axs[0].set_ylim(np.min(self.estimates['tof'].real), np.max(self.estimates['tof'].real))
        axs[1].set_title("aoa")
        axs[1].set_ylim(0, 180)
        axs[2].set_title("doppler")
        axs[3].set_title("amplitude")

        axs[0].set_xlim(0, self.total_steps)
        axs[1].set_xlim(0, self.total_steps)
        axs[2].set_xlim(0, self.total_steps)
        axs[3].set_xlim(0, self.total_steps)

        plt.show()


if __name__ == "__main__":
    conf = MyConfigsW2()
    csi = MyCsiW2(conf, 'test', '../npsave/0208/0208A03-csio.npy')
    csi.load_data()
    widar = MyWidar2(conf, csi)
    widar.run()
    widar.plot_results()
