import os
import numpy as np


def label_convert(in_path, out_path=None, autosave=False):
    labels = []
    with open(in_path) as f:
        for i, line in enumerate(f):
            if i > 0:
                labels.append([eval(line.split(',')[0]), eval(line.split(',')[1])])

    labels = np.array(labels)

    if autosave is True and out_path is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        np.save(out_path, labels)

    else:
        return labels


#label_convert('../sense/0208/02_labels.csv', None)
