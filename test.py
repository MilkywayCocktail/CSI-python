import csi_loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time


def foo():
    for i in range(100000):
        print('\r\r{}\n{}'.format(i, i+10), end='')
        time.sleep(1)

foo()
