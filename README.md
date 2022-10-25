# Python-based CSI analyze

### pycsi.py
main functionality codes

### tests.py
collection of all tests. You can run parallel tests and get batch results. (To be moved into myfunc.py)

### myfunc.py
A collection of all testing methods.

### csitests.py
The work panel of getting csi results.

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
|\_\_csitest.py (_Execute tests here, you can also import this file_)<br>
|\_\_myfunc.py (_Stores testing methods. Executed by csitest.py_)<br>
|\_\_pycsi.py (_Defines csi data strucutres and algorithms_)<br>

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

```
    filepath = "data/0919/"
    filenames = os.listdir(filepath)
    for file in filenames:
        name = file[3:-4]
        mypath = filepath + file
        _csi = MyCsi(name, mypath)
        _csi.load_data()
        _csi.save_csi(name)
```
- MyCsi is defined in pycsi.py, make sure to import it

### Try out algorithms
- At present, we have AoA, Doppler velocity, AoA-ToF and AoA-Doppler algorithms, all based on MUSIC.
- Sample code:

```
    name = "1010A30"
    npzpath = 'npsave/1010/csi'
    file = "npsave/" + name[:4] + '/' + name + "-csis.npz"
    csi = pycsi.MyCsi(name, file)
    csi.load_data()
    csi.aoa_by_music()
    csi.data.view_spectrum(threshold=0, autosave=False, notion='_vanilla')
```
- The csi data will be stored in MyCsi object after being instantialized. You can perform multiple actions by calling methods.
- 

# How to modify this code
Please write a new class to inherit the class you want to modify and override.<br>
