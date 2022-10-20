# Python-based CSI analyze

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

# How to use this code
The folder structure:<br>
----------<br>
|\_\_data (_Put your raw csi data here_)<br>
&ensp;&ensp;|\_\_(folder named by date)<br>
|\_\_logs (_Logs of tests_)<br>
&ensp;&ensp;|\_\_(folder named by date)<br>
|\_\_npsave (_Stores csi and spectrum as .npz_)<br>
&ensp;&ensp;|\_\_csi<br>
&ensp;&ensp;&ensp;&ensp;|\_\_(folder named by date)<br>
&ensp;&ensp;|\_\_spectrum<br>
&ensp;&ensp;&ensp;&ensp;|\_\_(folder named by date)<br>
|\_\_visualization (_stores figures_)<br>
&ensp;&ensp;|\_\_(folder named by date)<br>
|  
|\_\_csitest.py (_Execute tests here, you can also import this file_)<br>
|\_\_myfunc.py (_Stores testing methods. Executed by csitest.py_)<br>
|\_\_pycsi.py (_Defines csi data strucutres and algorithms_)<br>

# How to modify this code
Please write a new class to inherit the class you want to modify and override.<br>
