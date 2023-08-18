import numpy as np
import os
from tqdm import tqdm

from make_dataset import MyDataMaker
from make_dataset import MyConfigsDM


class MyDataMaker_v01c(MyDataMaker):
    # Generates images, CSI (in 3D), side labels

    def __init__(self, *args, **kwargs):
        MyDataMaker.__init__(self, *args, **kwargs)

    def __init_data__(self):
        # img_size = (width, height)
        csi = np.zeros((self.total_frames, 2, self.configs.sample_length, 30, 3))
        images = np.zeros((self.total_frames, self.configs.img_size[1], self.configs.img_size[0]))
        timestamps = np.zeros(self.total_frames)
        indices = np.zeros(self.total_frames)
        side_labels = np.zeros((self.total_frames, 3))
        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices, 'sid': side_labels}

    def export_sidelabel(self, label='x'):
        """
        Export labels: x-axis class or y-axis class
        :return: result['sid']
        """
        tqdm.write('Starting exporting sidelabel...')
        rel_timestamps = self.result['tim'] - self.result['tim'][0]
        sidelabels = []

        with open(self.paths[3]) as f:
            for i, line in enumerate(f):
                if i > 0:
                    if label == 'x':
                        sidelabels.append([eval(line.split(',')[0]),
                                           eval(line.split(',')[1]),
                                           eval(line.split('(')[1][:-4])[0]])
                    elif label == 'y':
                        sidelabels.append([eval(line.split(',')[0]),
                                           eval(line.split(',')[1]),
                                           eval(line.split('(')[1][:-4])[1]])

        for i in tqdm(range(len(rel_timestamps))):
            for rows in sidelabels:
                if rows[0] <= rel_timestamps[i] <= rows[1]:
                    # print(rows[0], rel_timestamps[i], rows[1])
                    self.result['sid'][i] = rows[2]

    def export_csi(self, dynamic_csi=True):
        """
        Requires export_image first
        """
        tqdm.write('Starting exporting CSI...')

        for i in tqdm(range(self.total_frames)):

            csi_index = np.searchsorted(self.csi_stream.timestamps, self.result['tim'][i])
            self.result['ind'][i] = csi_index
            csi_chunk = self.csi_stream.csi[csi_index: csi_index + self.configs.sample_length, :, :, 0]

            if dynamic_csi is True:
                csi_chunk = self.windowed_dynamic(csi_chunk)
            else:
                csi_chunk = csi_chunk

            # Store in two channels
            self.result['csi'][i, 0, ...] = np.abs(csi_chunk)
            self.result['csi'][i, 1, ...] = np.angle(csi_chunk)


if __name__ == '__main__':

    sub = '04'
    length = 3000

    path = [os.path.join('../sense/0307', sub + '.bag'),
            os.path.join('../sense/0307', sub + '_timestamps.txt'),
            os.path.join('../npsave/0307', '0307A' + sub + '-csio.npy'),
            os.path.join('../sense/0307', sub + '_labels.csv')]

    configs = MyConfigsDM()

    mkdata = MyDataMaker_v01c(configs=configs, paths=path, total_frames=length)
    mkdata.csi_stream.extract_dynamic(mode='overall-divide', ref='tx', reference_antenna=1)
    mkdata.csi_stream.extract_dynamic(mode='highpass')
    mkdata.export_image(show_img=False)
    # mkdata.export_sidelabel(label='y')
    mkdata.export_csi()
    mkdata.slice_by_label()
    print(mkdata.result['sid'])
    #mkdata.playback_image()
    mkdata.save_dataset('../dataset/0307/make07', sub + '_dyn', 'csi', 'img')
