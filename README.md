# CSI

## Python-based CSI analyze

### pycsi.py
main functionality codes

### test-xxx.py
test codes for testing defferent parts of pycsi. (To be moved into tests.py)

### tests.py
collection of all tests. You can run parallel tests and get batch results. (To be moved into myfunc.py)

### myfunc.py
A collection of all testing methods.

### csitests.py
The work panel of getting csi results.

### WIdar2.py
raw version

## How to use this code
<br>
<br>
The folder structure:<br>
----------<br>
|__data (Put your raw csi data here)<br>
____|__(folder named by date)<br>
|__logs (Logs of tests)<br>
____|__(folder named by date)<br>
|__npsave (stores csi and spectrum as .npz)<br>
____|__csi<br>
________|__(folder named by date)<br>
____|__spectrum<br>
________|__(folder names by date)<br>
|__visualization (stores figures)<br>
____|__(folder named by date)<br>
|<br>    
|__csitest.py (Execute tests here, you can also import this file)<br>
|__myfunc.py (Stores testing methods. Executed by csitest.py)<br>
|__pycsi.py (Defines csi data strucutres and algorithms)<br>

## How to modify this code
Please write a new class to inherit the class you want to modify and override.<br>
