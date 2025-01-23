import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def timer(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} total time: {end - start} sec")
        return result

    return wrapper


def plot_settings(figsize=(20, 10)):
    """
    Prepares plot configurations.
    :return: plt args
    """
    # Seems that the first figure has to be discarded
    _ = plt.figure()

    mpl.rcParams['figure.figsize'] = figsize
    mpl.rcParams["figure.titlesize"] = 35
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['axes.titlesize'] = 26
    mpl.rcParams['axes.labelsize'] = 26
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    fig = plt.figure(constrained_layout=True)
    return fig


def colors(arrays):
    """
    Color solution for plotting loss
    :param arrays: array of learning rates
    :return: variation of colors
    """
    arr = -np.log(arrays)
    norm = plt.Normalize(arr.min(), arr.max())
    map_vir = cm.get_cmap(name='viridis')
    c = map_vir(norm(arr))
    return c

def file_finder(path, func, process_name=None, *args, **kwargs):
    process_name = f"{process_name}: " if process_name else ''
    print(f"\033[32m{process_name}Loading {path}\033[0m")
    for p, _, file_lst in os.walk(path):
        for file_name in file_lst:
            file_name_, ext = os.path.splitext(file_name)
            func(os.path.join(p, file_name), file_name_, ext, *args, **kwargs)
            

def file_finder_multi(path, process_name=None):
    process_name = f"{process_name}: " if process_name else ''
    print(f"\033[32m{process_name}Loading {path}\033[0m")
    
    file_tasks = []  # To collect file processing tasks
    
    for p, _, file_lst in os.walk(path):
        for file_name in file_lst:
            file_name_, ext = os.path.splitext(file_name)
            file_tasks.append((os.path.join(p, file_name), file_name_, ext))
    
    return file_tasks

