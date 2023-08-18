import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
np.set_printoptions(suppress=True)


def calculate_timediff(in_path, out_path):
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
    offset = datetime.timedelta(microseconds=eval(offset) * 1e6)
    for i in range(len(out)):
        out[i] = datetime.datetime.strptime(out[i].strip(), "%Y/%m/%d %H:%M:%S.%f") - offset
        try:
            a = datetime.datetime.timestamp(datetime.datetime.strptime(out[i].strip(), "%Y-%m-%d %H:%M:%S.%f"))
        except ValueError:
            out[i] = datetime.datetime.strptime(
                out[i].strip(), "%Y-%m-%d %H:%M:%S") + datetime.timedelta(microseconds=1)

    print(out)
    np.savetxt(out_path, out, fmt='%s')
    f.close()
    r.close()


def calibrate_lag(in_path1, in_path2, out_path):
    """
    Calibrates timestamps of in_path1 against in_path2.
    Warning: -1day not solved!
    """
    f1 = open(in_path1, mode='r', encoding='utf-8')
    f2 = open(in_path2, mode='r', encoding='utf-8')
    r = open(out_path, mode='w+', encoding='utf-8')

    l1 = np.array(f1.readlines())
    l2 = np.array(f2.readlines())

    length = np.min([len(l1), len(l2)])
    temp = np.zeros(length)

    for i in range(length):
        t1 = datetime.datetime.strptime(l1[i].strip(), "%Y-%m-%d %H:%M:%S.%f")
        t2 = datetime.datetime.strptime(l2[i].strip(), "%Y-%m-%d %H:%M:%S.%f")
        temp[i] = (t1-t2).seconds

    lag = datetime.timedelta(seconds=np.mean(temp))

    for i in range(len(l1)):
        l1[i] = datetime.datetime.strptime(l1[i].strip(), "%Y-%m-%d %H:%M:%S.%f") - lag

    print(lag)
    np.savetxt(out_path, l1, fmt='%s')

    r.close()
    f1.close()
    f2.close()


# file = '../sense/0725/01_timestamps.txt'
# result = '../sense/0124/03_timediff.txt'

# file2 = '../data/0725/csi0725A01_time.txt'
# result2 = file2[:-4] + '_mod.txt'

file3 = '../sense/0124/00_cameratime.txt'
file4 = '../sense/0124/00_timestamps.txt'
result3 = '../sense/0124/00_camtime_mod.txt'

target = ('00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13')
for tar in target:
    file = f"../data/0726/csi0726A{tar}_time.txt"
    result = file[:-4] + '_mod.txt'
    compensate(file, result, '106.1387')

#calculate_timediff(file, result)

#calibrate_lag(file3, file4, result3)