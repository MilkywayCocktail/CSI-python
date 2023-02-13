import numpy as np
import os
from tqdm import tqdm

from make_dataset import MyDataMaker


class MyDataMaker_v01b(MyDataMaker):

    def __init__(self, *args, **kwargs):
        MyDataMaker.__init__(self, *args, **kwargs)

    def __init_data__(self):
        # img_size = (width, height)
        csi = np.zeros((self.total_frames, 2, 90, self.sample_length))
        images = np.zeros((self.total_frames, self.img_size[1], self.img_size[0]))
        timestamps = np.zeros(self.total_frames)
        indices = np.zeros(self.total_frames)
        side_labels = np.zeros(self.total_frames)
        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices, 'sid': side_labels}

    def export_sidelabel(self):
        tqdm.write('Starting exporting sidelabel...')
        rel_timestamps = self.result['tim'] - self.result['tim'][0]
        sidelabels = []
        with open(self.paths[4]) as f:
            for i, line in enumerate(f):
                if i > 0:
                    sidelabels.append([eval(line.split(',')[0]), eval(line.split(',')[1]), eval(line.split('(')[1][:-4])[0]])

        for i in tqdm(range(len(rel_timestamps))):
            for rows in sidelabels:
                if rows[0] <= rel_timestamps[i] <= rows[1]:
                    # print(rows[0], rel_timestamps[i], rows[1])
                    self.result['sid'][i] = rows[2]


if __name__ == '__main__':

    sub = '03'
    length = 3000

    path = [os.path.join('../sense/0208', sub + '.bag'),
            os.path.join('../sense/0208', sub + '_timestamps.txt'),
            os.path.join('../npsave/0208', '0208A' + sub + '-csio.npy'),
            os.path.join('../data/0208', 'csi0208A' + sub + '_time_mod.txt'),
            os.path.join('../sense/0208', sub + '_labels.csv')]

    mkdata = MyDataMaker_v01b(paths=path, total_frames=length, img_size=(128, 128), sample_length=100)
    mkdata.export_image(show_img=False)
    mkdata.export_sidelabel()
    mkdata.export_csi()
    mkdata.slice_by_label()
    print(mkdata.result['sid'])
    #mkdata.playback_image()
    mkdata.save_dataset('../dataset/0208/make00', sub + '_dyn', 'csi', 'sid')
