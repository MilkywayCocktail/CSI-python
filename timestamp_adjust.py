import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
np.set_printoptions(suppress=True)


def cal_diff(in_path, out_path):
    f = open(in_path, mode='r', encoding='utf-8')
    r = open(out_path, mode='w+', encoding='utf-8')

    out = np.array(f.readlines())
    start = None
    for i in range(len(out)):
        if start is None:
            start = datetime.datetime.strptime(out[i].strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp()
        out[i] = datetime.datetime.strptime(out[i].strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp() - start

    print(out)
    np.savetxt(out_path, out, fmt='%s')
    f.close()
    r.close()


def compensate(in_path, out_path, offset):
    f = open(in_path, mode='r', encoding='utf-8')
    r = open(out_path, mode='w+', encoding='utf-8')

    out = np.array(f.readlines())
    offset = datetime.timedelta(seconds=eval(offset))
    for i in range(len(out)):
        out[i] = datetime.datetime.strptime(out[i].strip(), "%Y/%m/%d %H:%M:%S.%f") - offset

    print(out)
    np.savetxt(out_path, out, fmt='%s')
    f.close()
    r.close()


file = '../sense/0124/03_timestamps.txt'
result = '../sense/0124/03_timediff.txt'

file2 = '../data/0124/csi0124A01_time.txt'
result2 = '../data/0124/csi0124A01_time_mod.txt'

compensate(file2, result2, '19.54')

cal_diff(file, result)
