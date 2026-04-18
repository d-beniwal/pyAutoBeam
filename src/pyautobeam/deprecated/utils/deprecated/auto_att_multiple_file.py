"""Multi-file attenuation workflow.

Processes all HDF5 data files in a directory, extracts max intensity
from each (after dark subtraction and masking), fits the Beer-Lambert
law to determine I0 and mu, and recommends acquisition times for all
attenuator positions.
"""

import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyautobeam.io.hdf5_reader import read_hdf5
from pyautobeam.processing.dark import subtract_dark
from pyautobeam.processing.mask import (
    apply_mask,
    create_dark_mask,
    create_percentile_mask,
    load_mask,
)
from pyautobeam.physics.attenuation import estimate_mu_linear
from pyautobeam.physics.beer_lambert import (
    beer_lambert_intensity,
    check_residuals,
    fit_beer_lambert,
)


# ── attenuator lookup ───────────────────────────────────────────────

# Beamline-specific mapping: attenuator position -> Cu thickness (mm)
_POS_THICKNESS = {
    0: 0.00, 1: 0.50, 2: 1.00, 3: 1.50, 4: 2.00, 5: 2.39,
    6: 4.78, 8: 7.14, 9: 9.53, 10: 11.91, 11: 14.30, 12: 16.66,
}

ALL_ATTENUATOR_POSITIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]


def att_thickness_from_pos(pos):
    """Map attenuator position index to Cu thickness in mm.

    Returns None for unknown positions.
    """
    return _POS_THICKNESS.get(pos, None)


# ── filename parsing ────────────────────────────────────────────────

_ATT_RE = re.compile(r"att(\d+)_(\d+)p(\d+)s")
_SAMPLE_RE = re.compile(r"^(.+?)_att")
_ENERGY_RE = re.compile(r"(\d+)keV")


def parse_attenuation_filenames(directory, filestem="", file_ext=".h5"):
    """Scan *directory* for HDF5 files and extract attenuation parameters.

    Returns a list of dicts with keys: filepath, att, acq_t, filename.
    """
    pattern = _ATT_RE
    results = []

    files = sorted(f for f in os.listdir(directory) if f.endswith(file_ext))
    if filestem:
        files = [f for f in files if f.startswith(filestem)]

    for filename in files:
        m = pattern.search(filename)
        if m:
            results.append({
                "filepath": os.path.join(os.path.abspath(directory), filename),
                "att": int(m.group(1)),
                "acq_t": float(f"{m.group(2)}.{m.group(3)}"),
                "filename": filename,
            })

    return results


# ── info-file catalogue ─────────────────────────────────────────────

def generate_info_file(datadir, output_path, file_ext=".h5"):
    """Scan *datadir* for HDF5 files and write a JSON catalogue."""
    datasets = []

    files = sorted(f for f in os.listdir(datadir) if f.endswith(file_ext))

    for filename in files:
        att_match = _ATT_RE.search(filename)
        if not att_match:
            continue

        att_pos = int(att_match.group(1))
        acq_time = float(f"{att_match.group(2)}.{att_match.group(3)}")
        thickness = att_thickness_from_pos(att_pos)

        sample_match = _SAMPLE_RE.search(filename)
        if sample_match:
            filestem = sample_match.group(1) + "_"
            sample = filename.split("_")[0]
        else:
            filestem = ""
            sample = "Unknown"

        datasets.append({
            "sample": sample,
            "filestem": filestem,
            "att_pos": att_pos,
            "att_thickness_mm": thickness,
            "acq_time_s": acq_time,
            "filepath": os.path.join(os.path.abspath(datadir), filename),
        })

    info = {"datasets": datasets}
    with open(output_path, "w") as fh:
        json.dump(info, fh, indent=2)

    print(f"Info file written: {output_path}  ({len(datasets)} datasets)")
    return info


def load_info_file(info_path, sample_filter=None):
    """Load a JSON catalogue, optionally filtering by sample name."""
    with open(info_path) as fh:
        info = json.load(fh)
    datasets = info["datasets"]
    if sample_filter:
        datasets = [d for d in datasets if d["sample"] == sample_filter]
    return datasets


# ── intensity extraction ─────────────────────────────────────────────

def extract_intensity(frames, dark=None, mask=None, percentile_mask_val=100.0,
                      skip_frames=None):
    """Extract the max intensity from a frame stack.

    Parameters
    ----------
    frames : numpy.ndarray
        3D array (N, Y, X) of detector frames.
    dark : numpy.ndarray or None
        Mean dark frame (2D).  Subtracted if provided.
    mask : numpy.ndarray or None
        2D binary mask (1=good, 0=bad).  Applied after dark subtraction.
    percentile_mask_val : float
        Percentile of pixels to keep (100 = no masking).
    skip_frames : list of int or None
        Frame indices to remove (default ``[0]``).

    Returns
    -------
    float
    """
    if skip_frames is None:
        skip_frames = [0]

    data = np.array(frames, dtype=np.float32)

    indices = [i for i in skip_frames if i < data.shape[0]]
    if indices:
        data = np.delete(data, indices, axis=0)

    if data.shape[0] == 0:
        return 0.0

    # Check before dark subtraction
    if np.max(data) <= 0:
        return 0.0

    if dark is not None:
        data = subtract_dark(data, dark)
        np.clip(data, 0, None, out=data)
        # Check after dark subtraction
        if np.max(data) <= 0:
            return 0.0

    if mask is not None:
        data = apply_mask(data, mask)

    if percentile_mask_val < 100.0:
        pct_mask = create_percentile_mask(data, percentile=percentile_mask_val)
        data = apply_mask(data, pct_mask)

    return float(np.max(data))


# ── plotting ─────────────────────────────────────────────────────────

def plot_fit(collected_data, fit_result, output_path="fit.png"):
    """Two-panel plot: Beer-Lambert fit (left) + residuals (right)."""
    valid_data = [d for d in collected_data if d["intensity"] > 0]
    if len(valid_data) < 2:
        return

    thicknesses = [d["thickness"] for d in valid_data]
    log_rates = [np.log(d["intensity"] / d["acq_time"]) for d in valid_data]
    n_pts = len(thicknesses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_pts))
    ax1.scatter(thicknesses, log_rates, c=colors, s=80, edgecolors="k",
                zorder=5, label="Data Points")

    for i, d in enumerate(valid_data):
        label = f"att{d['att_pos']}_{d['acq_time']}s"
        ax1.annotate(label, (thicknesses[i], log_rates[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    x_fit = np.linspace(min(thicknesses) - 0.2, max(thicknesses) + 0.2, 100)
    y_fit = fit_result["slope"] * x_fit + fit_result["intercept"]
    ax1.plot(x_fit, y_fit, "r--", linewidth=2,
             label=(f"Fit: mu={fit_result['mu']:.4f} /mm, "
                    f"I0={fit_result['I0']:.2e}\n"
                    f"R^2={fit_result['r_squared']:.6f}"))
    ax1.set_xlabel("Copper Thickness (mm)")
    ax1.set_ylabel("log(Intensity / time)")
    ax1.set_title(f"Beer-Lambert Fit ({n_pts} points)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    slope = fit_result["slope"]
    intercept = fit_result["intercept"]
    residuals = [log_rates[i] - (slope * thicknesses[i] + intercept)
                 for i in range(n_pts)]

    bar_colors = ["blue"] * n_pts
    if len(residuals) > 2:
        std_r = np.std(residuals)
        if std_r > 0:
            for i, r in enumerate(residuals):
                if abs(r / std_r) > 2.0:
                    bar_colors[i] = "red"

    labels = [f"att{d['att_pos']}\n{d['acq_time']}s" for d in valid_data]
    ax2.bar(labels, residuals, color=bar_colors, edgecolor="k", alpha=0.7)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel("Data Point")
    ax2.set_ylabel("Residual (log-space)")
    ax2.set_title("Fit Residuals")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ── main workflow ────────────────────────────────────────────────────

def auto_attenuate(datadir, sample, target_counts=50000,
                   skip_frames_str="0", darkfile=None, dark_mask=True,
                   maskfile=None, percentile_mask=100.0,
                   output_plot="fit.png", info_file=None,
                   file_ext=".h5", energy_keV=None):
    """Process all files in a directory and fit the Beer-Lambert law.

    Parameters
    ----------
    datadir : str
        Directory containing HDF5 data files.
    sample : str
        Sample name to filter from the info catalogue.
    target_counts : int
        Target counts for acquisition recommendations.
    skip_frames_str : str
        Dash-separated frame indices to skip (default ``"0"``).
    darkfile : str or None
        Path to a separate dark HDF5 file.  Dark frames are read from
        ``exchange/data`` and averaged.
    dark_mask : bool
        If True (default) and *darkfile* is provided, create a
        dead/hot pixel mask from the dark file and apply it.
    maskfile : str or None
        Path to a binary mask file (.tif, .npy).  1 = good, 0 = bad.
    percentile_mask : float
        Percentile of pixels to keep (100 = no masking).
    output_plot : str
        Path for the output fit plot.
    info_file : str or None
        Path to the JSON info file (default: ``<datadir>/data_info.json``).
    file_ext : str
        File extension filter.
    energy_keV : float or None
        X-ray photon energy in keV.  When provided, the initial mu
        estimate is computed from NIST Cu attenuation data.  If not
        provided, attempts to parse from the first data filename.

    Returns
    -------
    dict
        mu, I0, r_squared, collected_data, recommendations
    """
    skip_frames = [int(x) for x in skip_frames_str.split("-") if x.strip()]

    info_path = info_file or os.path.join(datadir, "data_info.json")

    if not os.path.exists(info_path):
        print(f"Info file not found at {info_path}. Generating...")
        generate_info_file(datadir, info_path, file_ext)

    # Load catalogue
    available_datasets = load_info_file(info_path, sample_filter=sample)
    if not available_datasets:
        print(f"ERROR: No datasets for sample '{sample}'.")
        return None

    # Parse energy from filename if not provided
    if energy_keV is None:
        first_file = os.path.basename(available_datasets[0]["filepath"])
        m = _ENERGY_RE.search(first_file)
        if m:
            energy_keV = float(m.group(1))

    # Compute physics-based mu estimate from NIST data
    mu_nist = None
    if energy_keV is not None:
        mu_nist = estimate_mu_linear(energy_keV)

    # Load dark: read exchange/data and average
    dark_mean = None
    pixel_mask = None
    if darkfile:
        dark_result = read_hdf5(darkfile)
        dark_frames = dark_result["data"]
        if dark_frames is not None:
            dark_mean = np.mean(dark_frames.astype(np.float32), axis=0)
            print(f"Dark loaded from {os.path.basename(darkfile)} "
                  f"({dark_frames.shape[0]} frames averaged).")
        else:
            print("WARNING: No data found in dark file.")

        if dark_mask and dark_mean is not None:
            pixel_mask, _ = create_dark_mask(darkfile)

    # User-provided mask
    if maskfile:
        user_mask = load_mask(maskfile)
        if pixel_mask is not None:
            pixel_mask = pixel_mask * user_mask
        else:
            pixel_mask = user_mask
        n_bad = int(np.sum(user_mask < 0.5))
        print(f"User mask applied ({n_bad} bad pixels).")

    # Banner
    print("\n" + "=" * 60)
    print("MULTI-FILE ATTENUATION ANALYSIS")
    print("=" * 60)
    print(f"Data directory : {datadir}")
    print(f"Sample         : {sample}")
    if energy_keV is not None:
        print(f"Energy         : {energy_keV} keV")
        print(f"mu (NIST)      : {mu_nist:.4f} /mm")
    print(f"Target counts  : {target_counts}")
    print(f"Skip frames    : {skip_frames}")
    print(f"Dark file      : {darkfile or 'None'}")
    print(f"Dark mask      : {'Yes' if dark_mask and darkfile else 'No'}")
    print(f"User mask      : {maskfile or 'None'}")
    if percentile_mask < 100.0:
        print(f"Percentile mask: {percentile_mask}%")
    print("=" * 60)

    # ── Process all files ───────────────────────────────────────────
    print(f"\nProcessing {len(available_datasets)} datasets...")
    print(f"{'File':<50} {'Att':>4} {'Thick(mm)':>10} "
          f"{'Acq(s)':>8} {'Max Intensity':>14} {'Status':>8}")
    print("-" * 100)

    collected_data = []
    for ds in available_datasets:
        basename = os.path.basename(ds["filepath"])
        att_pos = ds["att_pos"]
        thickness = ds["att_thickness_mm"]
        acq_time = ds["acq_time_s"]

        result = read_hdf5(ds["filepath"])
        frames = result["data"]

        intensity = extract_intensity(frames, dark_mean, pixel_mask,
                                      percentile_mask, skip_frames)

        if intensity <= 0:
            status = "SKIPPED"
        else:
            status = "OK"
            collected_data.append({
                "att_pos": att_pos,
                "thickness": thickness,
                "acq_time": acq_time,
                "filepath": ds["filepath"],
                "intensity": intensity,
            })

        print(f"{basename:<50} {att_pos:>4} {thickness:>10.2f} "
              f"{acq_time:>8.1f} {intensity:>14.1f} {status:>8}")

    # ── Fit Beer-Lambert ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("FIT RESULTS")
    print(f"{'=' * 60}")

    if len(collected_data) < 2:
        print("ERROR: Need at least 2 valid data points for fitting. "
              f"Got {len(collected_data)}.")
        return None

    thicknesses = [d["thickness"] for d in collected_data]
    intensities = [d["intensity"] for d in collected_data]
    acq_times = [d["acq_time"] for d in collected_data]

    fit_result = fit_beer_lambert(thicknesses, intensities, acq_times)

    fitted_mu = fit_result["mu"]
    fitted_I0 = fit_result["I0"]

    print(f"Valid data points : {len(collected_data)}")
    print(f"Fitted mu         : {fitted_mu:.4f} /mm")
    print(f"Fitted I0         : {fitted_I0:.2e} cts/s")
    print(f"R^2               : {fit_result['r_squared']:.6f}")

    if mu_nist is not None:
        pct_diff = 100.0 * abs(fitted_mu - mu_nist) / mu_nist
        print(f"mu (NIST)         : {mu_nist:.4f} /mm  "
              f"(difference: {pct_diff:.1f}%)")

    # Outlier check
    outlier_indices = check_residuals(fit_result, threshold=2.0)
    if outlier_indices:
        outlier_labels = [f"att{collected_data[i]['att_pos']}"
                          for i in outlier_indices]
        print(f"Outliers detected : {outlier_labels}")

    # Plot
    plot_fit(collected_data, fit_result, output_plot)
    print(f"Plot saved to     : {output_plot}")

    # ── Acquisition time recommendations ────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"ACQUISITION TIME RECOMMENDATIONS "
          f"(for 90% of target = {0.9 * target_counts:.0f} counts)")
    print(f"{'=' * 60}")
    target_90 = 0.9 * target_counts
    print(f"{'Att Pos':<10} {'Thickness (mm)':<16} "
          f"{'Rec. Time (s)':<16} {'Pred. Counts':<14} {'Note'}")
    print("-" * 70)

    recommendations = {}
    for pos in ALL_ATTENUATOR_POSITIONS:
        thick = att_thickness_from_pos(pos)
        if thick is None:
            continue
        pred_rate = beer_lambert_intensity(fitted_I0, fitted_mu, thick)
        if pred_rate > 0:
            rec_time = target_90 / pred_rate
            pred_counts = pred_rate * rec_time
            note = ""
            if rec_time < 0.005:
                note = "Too Fast"
            elif rec_time > 100:
                note = "Too Slow"
            recommendations[pos] = {
                "thickness_mm": thick,
                "recommended_time_s": rec_time,
                "predicted_counts": pred_counts,
            }
            print(f"{pos:<10} {thick:<16.2f} {rec_time:<16.4f} "
                  f"{pred_counts:<14.0f} {note}")

    return {
        "mu": fitted_mu,
        "I0": fitted_I0,
        "r_squared": fit_result["r_squared"],
        "collected_data": collected_data,
        "recommendations": recommendations,
    }
