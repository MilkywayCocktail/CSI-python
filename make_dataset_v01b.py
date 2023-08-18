import numpy as np
import os
from tqdm import tqdm

from make_dataset import MyDataMaker
from make_dataset import MyConfigsDM


class MyDataMaker_v01b(MyDataMaker):
    # Generates images, CSI, side labels

    def __init__(self, *args, **kwargs):
        MyDataMaker.__init__(self, *args, **kwargs)

    def __init_data__(self):
        # img_size = (width, height)
        csi = np.zeros((self.total_frames, 2, 90, self.configs.sample_length))
        images = np.zeros((self.total_frames, self.configs.img_size[1], self.configs.img_size[0]))
        timestamps = np.zeros(self.total_frames)
        indices = np.zeros(self.total_frames, dtype=int)
        side_labels = np.zeros(self.total_frames, dtype=int)
        return {'csi': csi, 'img': images, 'tim': timestamps, 'ind': indices, 'sid': side_labels}

    def export_sidelabel(self, label='x'):
        """
        Export labels: x-axis class or y-axis class
        :return: result['sid']
        """
        tqdm.write('Starting exporting sidelabel...')
        rel_timestamps = self.result['tim'] - self.result['tim'][0]
        sidelabels = []

        with open(self.paths[4]) as f:
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


if __name__ == '__main__':

    date = '0307'
    sub = '06'
    length = 3000

    path = ['../sense/' + date + '/' + sub + '.bag',
            '../sense/' + date + '/' + sub + '_timestamps.txt',
            '../npsave/' + date + '/' + date + 'A' + sub + '-csio.npy',
            '../data/' + date + '/csi' + date + 'A' + sub + '_time_mod.txt',
            '../sense/' + date + '/' + sub + '_labels.csv']

    configs = MyConfigsDM()

    mkdata = MyDataMaker_v01b(configs=configs, paths=path, total_frames=length)
    mkdata.csi_stream.extract_dynamic(mode='overall-divide', ref='tx', reference_antenna=1)
    mkdata.csi_stream.extract_dynamic(mode='highpass')
    mkdata.export_image(show_img=False)
    #print(mkdata.csi_stream.abs_timestamps)
    #print(mkdata.local_timestamps)
    #print(mkdata.result['tim'])
    mkdata.export_sidelabel(label='y')
    mkdata.export_csi(dynamic_csi=False, pick_tx=0)
    mkdata.slice_by_label()
    #mkdata.playback_image()
    mkdata.save_dataset('../dataset/0307/make07', sub + '_div', 'csi', 'sid')
