import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt


def separate(in_path, out_path, scope: tuple):
    result = {'csi': np.zeros((1, 2, 90, 100)),
             'img': np.zeros((1, 1, 128, 128)),
              'loc': np.zeros((1, 4))}
    inner = []

    filenames = os.listdir(in_path)
    for file in filenames:
        if file[:2] in scope:
            kind = file[-7:-4]
            if kind in list(result.keys()):
                print(file)
                tmp = np.load(in_path + file)
                print(tmp.shape)
                if kind == 'img':
                    tmp = tmp.reshape((-1, 1, 128, 128))

                if kind == 'loc':
                    for i in range(len(tmp)):
                        if tmp[i][0] != 0:
                            inner.append(i)

                result[kind] = np.concatenate((result[kind], tmp), axis=0)

    inner = np.array(inner)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for key in list(result.keys()):
        result[key] = np.delete(result[key], 0, axis=0)

        if len(result[key]) != 0:
            result[key] = result[key][inner]
            print(key, len(result[key]))

            np.save(out_path + key + '.npy', result[key])
    print("All saved!")
    
    
class Regrouper:
    def __init__(self, in_path, out_path, scope: tuple, wanted_types: dict):
        self.in_path = in_path
        self.out_path = out_path
        self.scope = scope
        self.result = {}
        for name, shape in wanted_types.items():
            self.result[name] = np.zeros((1, *shape))
            
    def load(self):
        print(f"Loading {self.scope}...")
        filenames = os.listdir(self.in_path)
        
        for file in filenames:
            if file[:2] in self.scope:
                modality = file.split('_')[-1].split('.')[0]
                try:
                    if modality in list(self.result.keys()):
                        data = np.load(self.in_path + file, mmap_mode='r')
                        self.result[modality] = np.concatenate((self.result[modality], data), axis=0)
                        print(f" Loaded {file} of {data.shape}")
                except Exception as e:
                    print(f"Error at {modality}:", e)

        print("All loaded!")
        for key in list(self.result.keys()):
            self.result[key] = np.delete(self.result[key], 0, axis=0)

    def regroup(self, number=0, img='', csi=''):
        print("Saving...")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        for key in list(self.result.keys()):
            filename = f"{self.out_path}{key}.npy"
            if len(self.result[key]) != 0:
                if key == 'sid':
                    self.result[key] = self.result[key] - min(self.result[key])
                elif key == 'img':
                    filename = f"{self.out_path}{img}{key}.npy"
                elif key == 'csi':
                    filename = f"{self.out_path}{csi}{key}.npy"
                if number == 0:
                    print(f" Saved {key} of len {len(self.result[key])}")
                    np.save(filename, self.result[key])
                else:
                    print(f" Saved {key} of len {number}")
                    np.save(filename, self.result[key][:number])
        print("All saved!")


class DataViewer:
    def __init__(self, path):
        self.path = path
        self.data = np.load(self.path, mmap_mode='r')
        print(f"Loaded file of {self.data.shape} as {self.data.dtype}")

    def view_csi(self):
        np.set_printoptions(threshold=np.inf)
        plt.subplot(1, 3, 1)
        plt.plot(self.data[:, 0, 0, 0])
        plt.title("Overall")
        plt.subplot(1, 3, 2)
        plt.imshow(self.data[0, 0])
        plt.title("Packet - amp")
        plt.subplot(1, 3, 3)
        plt.imshow(self.data[0, 1])
        plt.title("Packet - phase")
        plt.show()

    def view_image(self, shape=(640, 480)):
        self.data = (np.squeeze(self.data) * 255).astype(np.uint8)

        for i in range(len(self.data)):
            print(f"min = {np.min(self.data[i]) * 3000}, max = {np.max(self.data[i]) * 3000}")
            #img = cv2.convertScaleAbs(imgs[i], alpha=0.03)
            img = self.data[i]

            img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
            cv2.imshow('Image', img)

            #cv2.imwrite('../dataset/view/' + str(i).zfill(4) + '.jpg', img)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                break


class DataSplitter:
    def __init__(self, paths: dict, save_path):
        self.data = {}
        self.length = 0
        self.save_path = save_path

        print("Loading...")
        for key, value in paths:
            self.data[key] = np.load(value)
            self.length = len(self.data[key])
            print(f" Loaded {key} of len {self.length} as {self.data[key].dtype}")

        self.indices = np.arange(self.length)
        self.train_mask = None
        self.valid_mask = None
        self.train_ind = None
        self.valid_ind = None

    def split(self, train_ratio, valid_ratio):
        train_size = int(self.length * train_ratio)
        valid_size = int(self.length * valid_ratio)
        test_size = int(self.length - train_size - valid_size)
        print(f"Splitting. Train size = {train_size}, Valid size = {valid_size}, Test size = {test_size}...", end='')

        self.train_ind = np.random.choice(np.arange(self.length).astype(int), train_size, replace=False)
        self.train_mask = np.ones(self.length, np.bool)
        self.train_mask[self.train_ind] = 0

        self.valid_ind = np.random.choice(np.arange(self.length - train_size).astype(int), valid_size, replace=False)
        self.valid_mask = np.ones(self.length - train_size, np.bool)
        self.valid_mask[self.valid_ind] = 0
        print('Done')

    def save(self):
        print("Saving...")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for key, value in self.data:
            print(f" Saving {key}...")
            np.save(f"{self.save_path}{key}_train.npy", value[self.train_ind])
            np.save(f"{self.save_path}{key}_valid.npy", value[self.train_mask][self.valid_ind])
            np.save(f"{self.save_path}{key}_test.npy", value[self.train_mask][self.valid_mask])
        print("All saved!")


class PhaseDiff:
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path
        data = np.load(self.in_path, mmap_mode='r')
        print(f"Loaded file of {data.shape} as {data.dtype}")
        self.csi = (np.squeeze(data[:, 0, :, :]) * np.squeeze(np.exp(1.j * data[:, 1, :, :]))).reshape(-1, 30, 3, 100)
        self.result = {'AoA': np.zeros(self.csi.shape[0]),
                       'ToF': np.zeros(self.csi.shape[0])}

    def svd(self, mode='aoa'):
        print(f"Calculating {mode}...", end='')
        if mode == 'aoa':
            u, s, v = np.linalg.svd(self.csi.transpose(0, 2, 1, 3).reshape(-1, 3, 30 * 100), full_matrices=False)
            self.result['AoA'] = np.angle(np.squeeze(u[:, 0, 0]).conj() * np.squeeze(u[:, 1, 0]))
        elif mode == 'tof':
            u, s, v = np.linalg.svd(self.csi.transpose(0, 1, 2, 3).reshape(-1, 30, 3 * 100), full_matrices=False)
            self.result['ToF'] = np.average(np.angle(np.squeeze(u[:, :-1, 0])).conj() * np.squeeze(u[:, 1:, 0]), axis=-1)
        else:
            raise Exception('Please specify mode = \'aoa\' or \'tof\'.')
        print("Done!")

    def view(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.result['AoA'])
        plt.title("Estimated AoA")
        plt.subplot(1, 2, 2)
        plt.plot(self.result['ToF'])
        plt.title("Estimated ToF")
        plt.show()

    def save(self):
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        np.save(f"{self.out_path}pd.npy", np.concatenate(
            (self.result['AoA'][np.newaxis, ...], self.result['ToF'][np.newaxis, ...]), axis=0))
        print("All saved!")


def pseudo_dataset(out_path):
    csi = np.ones((1000, 1, 100, 90), dtype=complex) * (-1)
    csi_1 = np.ones((1000, 1, 100, 90), dtype=complex) * 0.5j
    csi_2 = np.ones((1000, 1, 100, 90), dtype=complex)

    sid = np.zeros(1000)
    sid1 = np.ones(1000)
    sid2 = np.ones(1000) * 2

    csi = np.concatenate((csi, csi_1, csi_2), axis=0)
    sid = np.concatenate((sid, sid1, sid2), axis=0)
    out_sid = np.zeros((3000, 3))
    for i in range(len(sid)):
        if sid[i] == -1:
            print("-1")
            out_sid[i] = [1, 0, 0]
        elif sid[i] == 0:
            print("0")
            out_sid[i] = [0, 1, 0]
        elif sid[i] == 1:
            print("1")
            out_sid[i] = [0, 0, 1]

    csi_abs = np.abs(csi)
    csi_phase = np.angle(csi)
    out_csi = np.concatenate((csi_abs, csi_phase), axis=1)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    np.save(out_path + 'csi.npy', out_csi)
    np.save(out_path + 'sid_.npy', sid)
    print("All saved!")


def pseudo_dataset_frq(out_path):
    x = np.arange(2000)
    y1 = np.sin(x)
    y2 = np.sin(2 * x)
    y3 = np.sin(3 * x)

    csi1 = np.zeros((1000, 1, 100, 90), dtype=complex)
    csi2 = np.zeros((1000, 1, 100, 90), dtype=complex)
    csi3 = np.zeros((1000, 1, 100, 90), dtype=complex)

    ind = np.zeros((3, 1000), dtype=int)
    for i in range(3):
        ind[i] = [random.randint(0, 1900) for _ in range(1000)]

    for i in range(1000):
        csi1[i] = 2 * np.exp(1j * np.arcsin(y1[ind[0,i]:ind[0,i] + 100]))[..., np.newaxis].repeat(90, axis=1).reshape(1, 1, 100, 90)
        csi2[i] = np.exp(1j * np.arcsin(y2[ind[1,i]:ind[1,i] + 100]))[..., np.newaxis].repeat(90, axis=1).reshape(1, 1, 100, 90)
        csi3[i] = 0.5 * np.exp(1j * np.arcsin(y3[ind[2,i]:ind[2,i] + 100]))[..., np.newaxis].repeat(90, axis=1).reshape(1, 1, 100, 90)

    sid = np.zeros(1000)
    sid1 = np.ones(1000)
    sid2 = np.ones(1000) * 2

    csi = np.concatenate((csi1, csi2, csi3), axis=0)
    sid = np.concatenate((sid, sid1, sid2), axis=0)

    csi_abs = np.abs(csi)
    csi_phase = np.angle(csi)
    out_csi = np.concatenate((csi_abs, csi_phase), axis=1)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    np.save(out_path + 'csi.npy', out_csi)
    np.save(out_path + 'sid.npy', sid)
    print("All saved!")


def simu_dataset(paths, out_path):
    out = []
    sid = []
    for path in paths:
        csi = np.load(path, allow_pickle=True)
        s = eval(path[-10])
        for i in range(500):
            amp = csi.item()['amp'][i * 100: (i + 1) * 100].reshape(100, 90).T
            phs = csi.item()['phs'][i * 100: (i + 1) * 100].reshape(100, 90).T
            data = np.concatenate((amp[np.newaxis, ...], phs[np.newaxis, ...]), axis=0)
            out.append(data)
            sid.append(s)
    out = np.array(out)
    sid = np.array(sid)
    print(out.shape)
    print(sid.shape)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #np.save(out_path + 'csi.npy', out)
    np.save(out_path + 'sid.npy', sid)


if __name__ == '__main__':
    viewer = DataViewer('../dataset/0509/make01-finished/csi.npy')
    viewer.view_csi()
    #pseudo_dataset('../dataset/0221/make01_finished/')

    #to_onehot('../dataset/0208/make00_finished/sid.npy', '../dataset/0208/make00_finished/sid2.npy')
    #from_onehot('../dataset/0208/make00_finished/sid_oh.npy', '../dataset/0208/make00_finished/sid.npy')
    #pseudo_dataset_frq('../dataset/0302/make00_finished/')

    #regroup('../dataset/0509/make05/', '../dataset/0509/make05-finished/', ('01', '02', '03', '04'))
    # separate('../dataset/0509/make01/', '../dataset/0509/make02-train/', ('01'))
    # wi2vi_channels('../dataset/0307/make07-finished/csi.npy', '../dataset/0307/make07-finished/csi-wi2vi2.npy')