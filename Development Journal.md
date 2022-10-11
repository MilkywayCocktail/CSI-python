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
- Modification: exception mechanism to handle I/O related situations

tests.py
- BUgfix: douplex parameter in order()
