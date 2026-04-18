# Attenuation Analysis Module

This module provides automated X-ray attenuation analysis for synchrotron beamline experiments using Cu attenuators.

## Physical Model

An incident X-ray beam of intensity I0 passes through a Cu attenuator disk of known thickness, then scatters off the sample. The measured detector intensity follows:

```
I_measured = S * I0 * exp(-mu * thickness) * acq_time
```


| Symbol        | Description                               | Source                                 |
| ------------- | ----------------------------------------- | -------------------------------------- |
| **mu**        | Cu linear attenuation coefficient (mm^-1) | NIST XCOM data or fitted               |
| **I0**        | Incident beam intensity (counts/s)        | Determined from data (as part of S*I0) |
| **S**         | Sample scattering factor                  | Determined from data (as part of S*I0) |
| **thickness** | Cu attenuator thickness (mm)              | Looked up from attenuator position     |
| **acq_time**  | Per-frame acquisition time (s)            | Parsed from filename                   |


The code determines the product **S*I0** automatically from the data. No user input for I0 is needed.

## How the Fitting Works

Taking the log of both sides:

```
log(I / t) = log(S * I0) - mu * thickness
           = C           - mu * thickness
```

### Single-file mode

With only one data point, mu cannot be fitted. The code uses **mu fixed from NIST** data and computes S*I0 directly:

```
C = log(I_max / acq_time) + mu_nist * thickness
S*I0 = exp(C)
```

### Multi-file mode

With multiple data points at different attenuation levels, both mu and S*I0 are **fitted via linear regression** in log-space:

```
log(I/t) = C - mu * thickness
```

The slope gives mu (fitted) and the intercept gives C = log(S*I0). The NIST mu value is reported alongside for comparison.

Model quality is assessed via:

- **R^2**: goodness of fit
- Per-point residuals, with outliers flagged at > 2*std

## Preprocessing Pipeline

Each data file goes through these steps before intensity extraction:

1. **Load frames** from `exchange/data` in the HDF5 file
2. **Skip frames** (default: frame 0). Configurable as dash-separated indices, e.g., `"0-1-3"`
3. **Dark subtraction**: if a dark file is provided, its frames (from `exchange/data`) are averaged into a mean dark frame, which is subtracted from each data frame. Negative values are clipped to zero.
4. **Dark mask**: dead pixels (zero variance across dark frames) and hot pixels (above local mean + 5*std in a 101x101 neighborhood) are automatically identified and masked
5. **User mask**: optional external binary mask file (.tif or .npy), convention 1=good, 0=bad
6. **Percentile mask**: optional removal of the top percentile of pixels
7. **Max intensity** is extracted from the fully processed frames

## File Filtering

Files are excluded from analysis if their max intensity (after all preprocessing) falls below a threshold:

- `ZERO`: max intensity is 0 (no signal survived processing)
- `LOW`: max intensity is below `min_intensity` (default 1000 counts)

This prevents noisy low-signal files from degrading the fit.

## Usage

### Single-File

```python
from pyautobeam.attenuation import analyze

result = analyze(
    path="data/Ceria_63keV_att0_1p0s.h5",
    target_counts=50000,
    darkfile="data/dark.h5",
)
# result["mu"]  = NIST mu (fixed)
# result["SI0"] = S*I0
```

### Multi-File

```python
result = analyze(
    path="data/Ceria/",
    filestem="Ceria",
    target_counts=50000,
    darkfile="data/Ceria/dark.h5",
)
# result["mu"]      = fitted mu
# result["mu_nist"] = NIST mu (for comparison)
# result["SI0"]     = fitted S*I0
# result["r2"]      = goodness of fit
```

### Command Line

```bash
# Multi-file (fits mu and S*I0)
python -m pyautobeam.attenuation.analysis --datapath data/Ceria/ --filestem Ceria \
    --target_intensity 50000 --darkfile data/dark.h5

# Single file (mu fixed from NIST)
python -m pyautobeam.attenuation.analysis --datapath data/scan_att0_1p0s.h5 --target_intensity 50000

# With all options
python -m pyautobeam.attenuation.analysis --datapath data/ --filestem LaB6 \
    --target_intensity 50000 --darkfile dark.h5 --percentile_mask 99.99 \
    --min_intensity 500 --skip_frames 0-1 --energy 63 --output_plot fit.png
```

## CLI Arguments


| Argument             | Required | Default               | Description                                   |
| -------------------- | -------- | --------------------- | --------------------------------------------- |
| `--datapath`         | yes      | --                    | HDF5 file or directory                        |
| `--target_intensity` | yes      | --                    | Target intensity (counts)                     |
| `--filestem`         | no       | `""`                  | Only process files starting with this string  |
| `--energy`           | no       | from filename         | X-ray energy in keV                           |
| `--darkfile`         | no       | `None`                | Dark HDF5 file path                           |
| `--dark_mask`        | no       | `1`                   | Create dead/hot pixel mask from dark (0 or 1) |
| `--maskfile`         | no       | `None`                | External binary mask file (.tif, .npy)        |
| `--percentile_mask`  | no       | `100.0`               | Percentile of pixels to keep                  |
| `--min_intensity`    | no       | `1000`                | Skip files below this max intensity           |
| `--skip_frames`      | no       | `"0"`                 | Frame indices to skip, dash-separated         |
| `--output_plot`      | no       | `attenuation_fit.png` | Path for output plot                          |


## NIST Data Interpolation

The Cu mass attenuation coefficient (mu/rho) is loaded from `data/Cu_att_data.txt`, which contains tabulated values from the NIST XCOM database across energies from 1 keV to 20 MeV.

Log-log interpolation is used for accuracy across the wide dynamic range. The Cu K-edge at 8.98 keV (a discontinuity in the table) is handled correctly.

Conversion to linear attenuation coefficient:

```
mu_linear [mm^-1] = (mu/rho [cm^2/g]) * density_Cu [g/cm^3] / 10
```

where density_Cu = 8.96 g/cm^3 and the factor of 10 converts cm^-1 to mm^-1.

## Attenuator Position Mapping


| Position | Cu Thickness (mm) |
| -------- | ----------------- |
| 0        | 0.00              |
| 1        | 0.50              |
| 2        | 1.00              |
| 3        | 1.50              |
| 4        | 2.00              |
| 5        | 2.39              |
| 6        | 4.78              |
| 8        | 7.14              |
| 9        | 9.53              |
| 10       | 11.91             |
| 11       | 14.30             |
| 12       | 16.66             |


Position 7 is not defined.

## Filename Convention

Data filenames must contain the pattern `att<N>_<T>p<D>s` for the code to parse the attenuator position and acquisition time. Energy is parsed from `<N>keV` if present.

Examples:

- `Ceria_63keV_900mm_100x100_att0_1p0s_012217.h5` -> att=0, acq=1.0s, energy=63 keV
- `LaB6_63keV_900mm_100x100_att3_0p5s_012345.h5` -> att=3, acq=0.5s, energy=63 keV

## Output

1. **Per-file processing table**: filename, att position, thickness, acq time, max intensity, log(I/t), status
2. **Fit results**: mu (fitted or NIST), S*I0, R^2 (multi-file), NIST comparison
3. **Per-point residuals** (multi-file): flagged if > 2*std
4. **Plot**: two-panel figure -- data vs model line (with NIST reference if fitted) + residual bar chart
5. **Acquisition time recommendations**: predicted time for each attenuator position to reach 90% of target intensity

## Frame Statistics

The `stats.py` module provides per-frame pixel intensity statistics for a single data file. It applies the same preprocessing pipeline as the main analysis and reports pixel counts in user-defined intensity bins.

```bash
python -m pyautobeam.attenuation.stats data/scan_att0_1p0s.h5 \
    --lowI 500 --highI 10000 --targetI 40000 --darkfile dark.h5
```

```python
from pyautobeam.attenuation import frame_stats

result = frame_stats(
    path="data/scan_att0_1p0s.h5",
    lowI=1000, highI=40000, targetI=50000,
    darkfile="dark.h5",
)
```

Output columns per frame:

- Pixels below `lowI`
- Pixels between `lowI` and `highI`
- Pixels above `highI`
- Pixels above `targetI`

## Automatic Attenuation Tuning (Bluesky Plan)

The `auto_attenuate.py` module provides a bluesky plan that automatically finds the optimal attenuator position and acquisition time by iteratively acquiring data and analyzing the results.

### Algorithm

1. Take an initial exposure at a conservative attenuator setting
2. Read the saved file, extract max intensity after preprocessing
3. Estimate S*I0 from the measurement using NIST mu
4. Predict the optimal (att_pos, acq_time) to reach the target window center
5. Acquire at the predicted settings
6. Repeat until measured intensity falls within the target window (default 70-90% of target)

### Usage

```python
from pyautobeam.attenuation import auto_attenuate_plan

# In a bluesky session with RunEngine:
RE(auto_attenuate_plan(
    det=my_detector,           # any ophyd detector
    attenuator=attenB,         # attenuator device with .rz motor
    shutter=fs,                # fast shutter device
    shutter_open_cmd=fs_open,  # plan stub to open shutter
    shutter_close_cmd=fs_close,# plan stub to close shutter
    sample_name="Ceria",
    energy_keV=63,
    target_intensity=45000,
    darkfile="/path/to/dark.h5",
    data_dir="/path/to/save/",
))
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `det` | required | Detector object (cam + hdf1 plugin) |
| `attenuator` | required | Attenuator device with .rz motor |
| `shutter` | required | Fast shutter device |
| `shutter_open_cmd` | required | Plan stub to open shutter |
| `shutter_close_cmd` | required | Plan stub to close shutter |
| `sample_name` | required | Sample name for file naming |
| `energy_keV` | required | X-ray energy in keV |
| `target_intensity` | 45000 | Target max pixel intensity |
| `target_window` | (0.7, 0.9) | Acceptable fraction of target |
| `initial_att_pos` | 3 | Starting attenuator position (conservative) |
| `initial_acq_time` | 1.0 | Starting acquisition time (s) |
| `nframes` | 5 | Frames per acquisition |
| `darkfile` | None | Dark file for preprocessing |
| `data_dir` | cwd | Where to save calibration files |
| `max_iterations` | 10 | Maximum acquisition attempts |

### Safety

- The plan starts at a conservative attenuator position (default att3) to avoid saturating the detector
- Maximum iterations prevent infinite loops
- The module does NOT connect to hardware when imported -- only when executed via `RE()`
- Files are saved with descriptive names for later review

## Module Contents


| File                   | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| `analysis.py`          | Offline analysis: fit S*I0 from saved data files                  |
| `auto_attenuate.py`    | Bluesky plan: automatic attenuation tuning (live acquisition)     |
| `stats.py`             | Per-frame pixel intensity statistics                              |
| `beer_lambert.py`      | Beer-Lambert law: forward calculation, fitting, outlier detection |
| `nist_data.py`         | NIST Cu attenuation coefficient loading and interpolation         |
| `data/Cu_att_data.txt` | NIST XCOM tabulated data for Cu                                   |


