import numpy as np
import os
from tqdm import tqdm

from make_dataset import MyDataMaker


class MyDataMaker_v01c(MyDataMaker):

    def __init__(self, *args, **kwargs):
        MyDataMaker.__init__(self, *args, **kwargs)

    def __init_data__(self):
        # img_size = (width, height)
        csi = np.zeros((self.total_frames, 2, self.sample_length, 30, 3))
        images = np.zeros((self.total_frames, self.img_size[1], self.img_size[0]))
        timestamps = np.zeros(self.total_frames)
        indices = np.zeros(self.total_frames)
        side_labels = np.zeros((self.total_frames, 3))
        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices, 'sid': side_labels}

    def export_sidelabel(self):
        tqdm.write('Starting exporting sidelabel...')
        rel_timestamps = self.result['tim'] - self.result['tim'][0]
        sidelabels = []
        onehot = {-1: (1, 0, 0),
                  0: (0, 1, 0),
                  1: (0, 0, 1)}

        with open(self.paths[4]) as f:
            for i, line in enumerate(f):
                if i > 0:
                    sidelabels.append([eval(line.split(',')[0]), eval(line.split(',')[1]), eval(line.split('(')[1][:-4])[0]])

        for i in tqdm(range(len(rel_timestamps))):
            for rows in sidelabels:
                if rows[0] <= rel_timestamps[i] <= rows[1]:
                    # print(rows[0], rel_timestamps[i], rows[1])
                    self.result['sid'][i] = onehot[rows[2]]

    def export_csi(self, dynamic_csi=True):
        """
        Requires export_image
        """
        tqdm.write('Starting exporting CSI...')

        for i in tqdm(range(self.total_frames)):

            csi_index = np.searchsorted(self.csi_stream['time'], self.result['tim'][i])
            self.result['ind'][i] = csi_index
            csi_chunk = self.csi_stream['csi'][csi_index: csi_index + self.sample_length, :, :, 0]

            if dynamic_csi is True:
                csi_chunk = self.windowed_dynamic(csi_chunk)
            else:
                csi_chunk = csi_chunk

            # Store in two channels
            self.result['csi'][i, 0, ...] = np.abs(csi_chunk)
            self.result['csi'][i, 1, ...] = np.angle(csi_chunk)


if __name__ == '__main__':

    sub = '02'
    length = 3000

    path = [os.path.join('../sense/0208', sub + '.bag'),
            os.path.join('../sense/0208', sub + '_timestamps.txt'),
            os.path.join('../npsave/0208', '0208A' + sub + '-csio.npy'),
            os.path.join('../data/0208', 'csi0208A' + sub + '_time_mod.txt'),
            os.path.join('../sense/0208', sub + '_labels.csv')]

    mkdata = MyDataMaker_v01c(paths=path, total_frames=length, img_size=(128, 128), sample_length=100)
    mkdata.export_image(show_img=False)
    mkdata.export_sidelabel()
    mkdata.export_csi()
    mkdata.slice_by_label()
    print(mkdata.result['sid'])
    #mkdata.playback_image()
    mkdata.save_dataset('../dataset/0208/make01', sub + '_dyn', 'csi', 'sid')
