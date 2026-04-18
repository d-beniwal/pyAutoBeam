# pyAutoBeam

A modular Python toolkit for automated synchrotron beamline data processing and analysis.

pyAutoBeam provides building blocks for working with detector data from synchrotron X-ray experiments: reading HDF5 files, preprocessing frames (dark subtraction, pixel masking), and running experiment-specific analysis workflows.

## Installation

```bash
cd pyAutoBeam
pip install -e .
```

## Package Structure

```
src/pyautobeam/
    io/                  General-purpose data I/O
        hdf5_reader.py       Read HDF5 detector data (APS convention)

    processing/          General-purpose data processing
        dark.py              Dark current subtraction
        mask.py              Pixel mask loading, creation, and application

    attenuation/         X-ray attenuation analysis workflow
        analysis.py          Main analysis tool (single + multi-file, CLI)
        beer_lambert.py      Beer-Lambert law fitting
        nist_data.py         NIST Cu attenuation coefficient interpolation
        data/                Tabulated NIST XCOM data
        README.md            Detailed workflow documentation
```

## Modules

### io -- Data I/O

Read HDF5 detector data following APS beamline conventions:

```python
from pyautobeam.io import read_hdf5

result = read_hdf5("path/to/data.h5")
frames = result["data"]       # (N, Y, X) detector frames
dark   = result["dark"]       # dark frames (or None)
bright = result["bright"]     # flat-field frames (or None)
meta   = result["metadata"]   # instrument metadata
```

### processing -- Data Processing

Dark subtraction and pixel masking utilities:

```python
from pyautobeam.processing import subtract_dark, load_mask, apply_mask, create_percentile_mask

# Dark subtraction (multi-frame darks averaged automatically)
corrected = subtract_dark(frames, dark)

# Apply a mask (1=good, 0=bad)
mask = load_mask("bad_pixels.npy")
masked = apply_mask(corrected, mask)

# Or create a percentile mask to remove hot pixels
pct_mask = create_percentile_mask(corrected, percentile=99.99)
masked = apply_mask(corrected, pct_mask)
```

A dark-file based mask can also be created to identify dead and hot pixels:

```python
from pyautobeam.processing import create_dark_mask

mask, info = create_dark_mask("dark.h5")
# Uses local 101x101 window for hot pixel detection
# Handles spatial non-uniformity in detector dark pedestal
```

### attenuation -- X-ray Attenuation Analysis

Analyze X-ray attenuation through Cu attenuators using the scattering model. See [attenuation/README.md](src/pyautobeam/attenuation/README.md) for the full workflow description.

```python
from pyautobeam.attenuation import analyze

# Multi-file: fits mu and S*I0 from all files in a directory
result = analyze(
    path="test_data/Ceria",
    filestem="Ceria",
    target_counts=50000,
    darkfile="test_data/Ceria/dark.h5",
)

# Single-file: uses NIST mu, computes S*I0
result = analyze(
    path="test_data/Ceria/Ceria_63keV_att0_1p0s.h5",
    target_counts=50000,
)
```

Command line:

```bash
python -m pyautobeam.attenuation.analysis --datapath test_data/Ceria/ \
    --filestem Ceria --target_intensity 50000 --darkfile test_data/Ceria/dark.h5
```

## HDF5 Data Format

Expected structure (APS beamline convention):

```
exchange/data    (N, Y, X)  uint16  -- detector frames
exchange/dark    (M, Y, X)  uint16  -- dark current frames (optional)
exchange/bright  (M, Y, X)  uint16  -- flat-field frames (optional)
```

Dark calibration files store dark frames in `exchange/data`.

## Dependencies

- numpy
- h5py
- scipy
- tifffile
- matplotlib

## Test Data

The `test_data/` directory contains example HDF5 files:
- `test_data/Ceria/` -- CeO2 calibration at 63 keV, attenuator positions 0-6
- `test_data/LaB6/` -- LaB6 calibration at 63 keV, various attenuator positions and acquisition times
