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
- Bugfix: resample_packets(), added reaction to nagative timestamping bug (originated in csitool)

tests.py
- Added function: npzloader()
- Added function: test_times()
- Modification: all functions, inserted npzloader()
