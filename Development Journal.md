# Development Journal
by CAO

## 0929
(Former basis skipped)

pycsi.py
- Added parameter: reference_antenna into calibrate_phase() and extract_dynamic()
- Added parameter: pick_antenna into doppler_by_music()
- Modification: all functions, changed naming rule for npz data
- Bugfix: view_spwctrum(), labeling related bug

tests.py
- Added function: batch_tool()
- Bugfix: test_resampling(), labeling related bug

## 0930

pycsi.py
- Bugfix: resample_packets(), added response to nagative timestamping bug (originated in csitool)

tests.py
- Added function: npzloader()
- Added function: test_times()
- Modification: all functions, inserted npzloader()

## 1003

pycsi.py
- Bugfix: extract_dynamic(), conjugate multiplication related bug
- Bugfix: aoa_by_music(), double definition of smooth
- Added function: aoa_tof_by_music()

tests.py
- Modification: test_doppler(), removed calibrate_phase()

## 1004

tests.py
- Added parameter: ref_antenna, packet into test_phasediff()
- Added function: test_aoatof()
- Modification: batch() merged into order() so that the if branches are eliminated

## 1005

pycsi.py
- Added function: sanitize_phase()

tests.py
- Added function: test_sanitize()

## 1011

pycsi.py
- Modification: exception design to handle I/O related situations

tests.py
- BUgfix: douplex parameter in order()

## 1012

pycsi.py
- Added parameter: self.xlabels in class \_Data, for plotting
- Added function: show_antenna_strength()
- Modification: doppler_by_music(), adjusted for strides
- Modification: doppler_by_music(), built in self-calibration and dynamic extraction
- Modification: doppler_by_music(), collaborate with resampling()

tests.py
- Added function: test_abs()

## 1013

pycsi.py
- Modification: calibration(), added strength-based reference selection
- Modification: calibration(), added support for multiple references

tests.py
- Modification: test_calibration(), added support for multiple references

## 1014

pycsi.py
- Modification: calibration(), added antenna-specific phase delay
- Added parameter: self.sampling_rate in class \MyCsi, approximated as 3965(Hz)
- Modification: extract_dynamic(), added strength-based reference selection
- Modification: extract_dynamic(), added high-pass filter option

## 1017

pycsi.py
- Bugfix: extract_dynamic(), highpass filter related bug
- Added function: noise_space(), as static method
- Added function: aoa_doppler_by_music()

## 1018

pycsi.py
- Modification: reconstruct_csi(), changed returning shape
- Modification: vis_spectrum(), added aoa-doppler option
- Modification: save_spectrum(), added notion support
- Added functoin: rearrange_antenna()

tests.py
- Added function: test_aoadoppler()

pycsitest.py
- Started drafting

## 1019

pycsitest.py
- Added function: logger()
- Added function: show_all_methods()
- Added function: test()
- Added function: \_test_phase_diff()

## 1020

pycsi.py
- Added parameter: \_\_str__
- Added parameter: \_\_repr__

pycsitest.py:
- Split into myfunc.py and csitest.py

myfunc.py
- Added class: MyFunc
- Added class: \_TestPhaseDiff

csitest.py
- Added class: MyTest

# 1021

pycsi.py
- Modification: added return for vis_spectrum()
- Added parameter: self_cal in doppler_by_music()

myfunc.py
- Added class: CoultClass
- Added class: \_TestResampling
- Added class: \_TestSanitize
- Added class: \_TestAoA
- Added class: \_TestDoppler
- Added class: \_TestAoAToF
- Added class: \_TestAoADoppler
- Added function: preprocess()

# 1024

pycsi.py
- Added Parameter: folder_name in save_spectrum()

myfunc.py
- Bugfix: iteration for visualizing spectrum in \_TestAoAToF and \_TestAoADoppler

# 1025

pycsi.py
- Bugfix: save_data() and save_spectrum(), replaced os.mkdir() with os.makedirs() to support multi-level folder creation

# 1026

pycsi.py
- Bugfix: noise_space(), squeezing csi before operation
- Bugfix: noise_space(), position of transpose (correlatively in aoa_tof_by_music())
- Modification: remove_inf_valuse(), added denial operation to data with more than 20% -inf packages

myfunc.py
- Added parameter: self.smoothing in \_TestDoppler() and \_TestAoADoppler()

csitest.py
- Bugfix: run(), failed to load references

# 1027

pycsi.py
- Modification: calibrate_phase(), corrected mechanism

myfunc.py
- Added class: \_TestPhase()

csitest.py
- Bugfix: run(), failed to iterate among subjects
