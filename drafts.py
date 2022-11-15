import pycsi
import csitest
import numpy as np
import random
import matplotlib.pyplot as plt

cal = {'0': '1110A00',
       '30': '1110A01',
       '60': '1110A02',
       '-60': '1110A10',
       '-30': '1110A11'}

print(str(cal.keys())[10:-1])

s = '1110A11'

print(s[-3:])