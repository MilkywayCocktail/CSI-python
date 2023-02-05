import csi_loader
import numpy as np
import datetime

t1 = '2023-01-24 17:38:04.621438'
t2 = '2023-01-24 17:38:04.720080'

time1 = datetime.datetime.strptime(t1.strip(), "%Y-%m-%d %H:%M:%S.%f")
time2 = datetime.datetime.strptime(t2.strip(), "%Y-%m-%d %H:%M:%S.%f")

print((time1 - time2).days)
print((time2 - time1).microseconds / 1000)