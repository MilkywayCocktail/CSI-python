# CSI

## Python-based CSI analyze

### pycsi.py
main functionality codes

### test-xxx.py
test codes for testing defferent parts of pycsi. (To be moved into tests.py)

### tests.py
collection of all tests. You can run parallel tests and get batch results. (To be moved into pycsitest.py)

### pycsitest.py
Provides an overall view of all testing methods.

### WIdar2.py
raw version

## How to use this code

Mainly you need pycsi.py which defines data structures and methodologies.<br>
CSIKit is used for acquiring CSI data.<br>
Test functions are integrated in tests.py.<br>
In tests.py, all you need to run is function order().<br>
You can refer to the "menu" defined in the order(), and run the test fuction by its corresponding index.
