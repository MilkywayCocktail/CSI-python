# Python-based CSI analyze

### pycsi.py
main functionality codes

### tests.py
collection of all tests. You can run parallel tests and get batch results. (To be moved into myfunc.py)

### myfunc.py
A collection of all testing methods.

### csitests.py
The work panel of getting csi results.

### simulator.py
Generates virtual CSI data.

### Widar2.py
raw version

# How to use this code
The folder structure:<br>
----------<br>
|\_\_data (_Put your raw csi data here_)<br>
&ensp;&ensp;|\_\_ (folder named by date)<br>
|\_\_logs (_Logs of tests_)<br>
&ensp;&ensp;|\_\_ (folder named by date)<br>
|\_\_npsave (_Stores csi and spectrum as .npz_)<br>
&ensp;&ensp;|\_\_csi<br>
&ensp;&ensp;&ensp;&ensp;|\_\_ (folder named by date)<br>
&ensp;&ensp;|\_\_spectrum<br>
&ensp;&ensp;&ensp;&ensp;|\_\_ (folder named by date)<br>
|\_\_visualization (_stores figures_)<br>
&ensp;&ensp;|\_\_ (folder named by date)<br>
|  
|\_\_constant_value.py (_Ohara's codes_)<br>
|\_\_csi_loader.py (_Ohara's codes, converts CSI files to .npy_)<br>
|\_\_csi_parser.py (_Ohara's codes, supporting csi_loader_)<br>
|\_\_csi_parser_numba.py (_Ohara's codes, supporting csi_loader, faster_)<br>
|\_\_csitest.py (_Execute tests here, you can also import this file_)<br>
|\_\_myfunc.py (_Stores testing methods. Executed by csitest.py_)<br>
|\_\_pycsi.py (_Defines csi data strucutres and algorithms_)<br>
|\_\_save_npy.py (_Save CSI files as .npy with csi_loader_)<br>
|\_\_simulator.py (_Perform simulation here_)<br>

## MANUAL
### First things first
- Name your raw csi file as <csiMMDDNXX.dat>
- MMDD: month and day
- N: computer index (A, B, ...)
- XX: number of take

### Preload the data
- Use CSIKit to convert .dat files into .npz, which saves a bunch of time for later use
- Put your raw csi file in data/MMDD/ as illustrated in the folder structure
- Sample code:

```python
    filepath = "data/0919/"
    filenames = os.listdir(filepath)
    for file in filenames:
        name = file[3:-4]
        mypath = filepath + file
        _csi = MyCsi(name, mypath)
        _csi.load_data()
        _csi.save_csi(name)
```
- The saved file is named as <MMDDNXX-csis.npz>
- MyCsi is defined in pycsi.py, make sure to import it
- You may find '-inf' values in CSI data with CSIKit. Using Ohara's csi_loader can avoid this.

### Try out algorithms
- At present, we have AoA, Doppler velocity, AoA-ToF and AoA-Doppler algorithms, all based on MUSIC.
- Sample code:

```python
    name = "1010A30"
    file = "npsave/" + name[:4] + '/' + name + "-csis.npz"
    csi = pycsi.MyCsi(name, file)
    csi.load_data()
    csi.aoa_by_music()
    csi.data.view_spectrum(threshold=0, autosave=False, notion='_vanilla')
```
- The csi data will be stored in MyCsi object after being instantialized. You can perform multiple actions by calling methods.

### Try out simulation
- Using simulator.py, you can generate virtual CSI data.
- First generate ground truth, then apply it into simulated CSI.
- Make sure the lengths of ground truth and vitual data are the same.
- Sample code:

```python
    gt1 = GroundTruth(length=10000).aoa
    gt1.random_points(10)
    gt1.interpolate(5)
    data = DataSimulator(length=10000)
    data.add_baseband()
    data.apply_gt(gt1)
    data.add_noise()
    simu = data.derive_MyCsi('GT01')
    simu.aoa_by_music()
    simu.data.view_spectrum()
```

# How to modify this code
Please write a new class to inherit the class you want to modify and override.<br>
