"""Attenuation analysis using the scattering model.

Physical model::

    I_measured = S * I0 * exp(-mu * thickness) * acq_time

where:
- mu is fixed from NIST Cu mass attenuation data (interpolated at the
  X-ray energy)
- S is the sample scattering factor
- I0 is the incident beam intensity

The code fits for S*I0 from the data.  When I0 is provided by the
user, S is reported separately.

Supports both single-file and multi-file (directory) analysis.  For
single-file mode, I0 must be provided.
"""

import argparse
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


# ── attenuator lookup ───────────────────────────────────────────────

_POS_THICKNESS = {
    0: 0.00, 1: 0.50, 2: 1.00, 3: 1.50, 4: 2.00, 5: 2.39,
    6: 4.78, 8: 7.14, 9: 9.53, 10: 11.91, 11: 14.30, 12: 16.66,
}

ALL_ATTENUATOR_POSITIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]


def att_thickness_from_pos(pos):
    """Map attenuator position index to Cu thickness in mm."""
    return _POS_THICKNESS.get(pos, None)


# ── filename parsing ────────────────────────────────────────────────

_ATT_RE = re.compile(r"att(\d+)_(\d+)p(\d+)s")
_ENERGY_RE = re.compile(r"(\d+)keV")


def parse_filename(filepath):
    """Extract energy, attenuator position, and acquisition time.

    Returns
    -------
    dict or None
        Keys: energy_keV (or None), att_pos, acq_time_s, thickness_mm.
        Returns None if the filename doesn't match the att pattern.
    """
    basename = os.path.basename(filepath)

    m = _ATT_RE.search(basename)
    if not m:
        return None

    att_pos = int(m.group(1))
    acq_time_s = float(f"{m.group(2)}.{m.group(3)}")
    thickness_mm = att_thickness_from_pos(att_pos)

    energy_keV = None
    em = _ENERGY_RE.search(basename)
    if em:
        energy_keV = float(em.group(1))

    return {
        "energy_keV": energy_keV,
        "att_pos": att_pos,
        "acq_time_s": acq_time_s,
        "thickness_mm": thickness_mm,
    }


# ── file discovery ──────────────────────────────────────────────────

def discover_files(path, filestem="", file_ext=".h5"):
    """Find data files from a path (single file or directory).

    Parameters
    ----------
    path : str
        Path to a single HDF5 file or a directory.
    filestem : str
        If *path* is a directory, only include files whose name
        starts with this string.
    file_ext : str
        File extension filter for directory mode.

    Returns
    -------
    list of dict
        Each dict has keys: filepath, att_pos, acq_time_s,
        thickness_mm, energy_keV (may be None).
    """
    if os.path.isfile(path):
        meta = parse_filename(path)
        if meta is None:
            print(f"ERROR: Cannot parse att/acq_time from '{path}'.")
            return []
        meta["filepath"] = os.path.abspath(path)
        return [meta]

    if not os.path.isdir(path):
        print(f"ERROR: '{path}' is not a file or directory.")
        return []

    files = sorted(f for f in os.listdir(path) if f.endswith(file_ext))
    if filestem:
        files = [f for f in files if f.startswith(filestem)]

    datasets = []
    for filename in files:
        filepath = os.path.join(os.path.abspath(path), filename)
        meta = parse_filename(filepath)
        if meta is not None:
            meta["filepath"] = filepath
            datasets.append(meta)

    return datasets


# ── intensity extraction ────────────────────────────────────────────

def extract_intensity(frames, dark=None, mask=None, percentile_mask_val=100.0,
                      skip_frames=None):
    """Extract the max intensity from a frame stack.

    Parameters
    ----------
    frames : numpy.ndarray
        3D array (N, Y, X) of detector frames.
    dark : numpy.ndarray or None
        Mean dark frame (2D).
    mask : numpy.ndarray or None
        2D binary mask (1=good, 0=bad).
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

    if np.max(data) <= 0:
        return 0.0

    if dark is not None:
        data = subtract_dark(data, dark)
        np.clip(data, 0, None, out=data)
        if np.max(data) <= 0:
            return 0.0

    if mask is not None:
        data = apply_mask(data, mask)

    if percentile_mask_val < 100.0:
        pct_mask = create_percentile_mask(data, percentile=percentile_mask_val)
        data = apply_mask(data, pct_mask)

    return float(np.max(data))


# ── plotting ─────────────────────────────────────────────────────────

def plot_fit(collected_data, mu, C, output_path="attenuation_fit.png"):
    """Plot data points against the fixed-slope model line."""
    valid = [d for d in collected_data if d["intensity"] > 0]
    if not valid:
        return

    thicknesses = [d["thickness"] for d in valid]
    log_rates = [d["log_rate"] for d in valid]
    n_pts = len(thicknesses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Blues(np.linspace(0.3, 1.0, n_pts))
    ax1.scatter(thicknesses, log_rates, c=colors, s=80, edgecolors="k",
                zorder=5, label="Data Points")

    for i, d in enumerate(valid):
        label = f"att{d['att_pos']}_{d['acq_time']}s"
        ax1.annotate(label, (thicknesses[i], log_rates[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    x_fit = np.linspace(
        min(thicknesses) - 0.2,
        max(max(thicknesses), 0.5) + 0.2,
        100,
    )
    y_fit = C - mu * x_fit
    SI0 = math.exp(C)
    ax1.plot(x_fit, y_fit, "r--", linewidth=2,
             label=(f"Fixed mu={mu:.4f} /mm\n"
                    f"S*I0={SI0:.2e} cts/s"))
    ax1.set_xlabel("Copper Thickness (mm)")
    ax1.set_ylabel("log(Intensity / time)")
    ax1.set_title(f"Attenuation Fit ({n_pts} points, mu fixed from NIST)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    if n_pts > 1:
        residuals = [d["C_i"] - C for d in valid]
        bar_colors = ["blue"] * n_pts
        if n_pts > 2:
            std_r = np.std(residuals)
            if std_r > 0:
                for i, r in enumerate(residuals):
                    if abs(r / std_r) > 2.0:
                        bar_colors[i] = "red"
        labels = [f"att{d['att_pos']}\n{d['acq_time']}s" for d in valid]
        ax2.bar(labels, residuals, color=bar_colors, edgecolor="k", alpha=0.7)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.set_xlabel("Data Point")
        ax2.set_ylabel("Residual (C_i - C)")
        ax2.set_title("Model Consistency")
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "Single point\n(no residuals)",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Model Consistency")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ── main analysis ────────────────────────────────────────────────────

def analyze(path, target_counts=50000, filestem="",
            skip_frames_str="0", darkfile=None, dark_mask=True,
            maskfile=None, percentile_mask=100.0,
            min_intensity=1000,
            output_plot="attenuation_fit.png",
            file_ext=".h5", energy_keV=None, I0=None):
    """Run attenuation analysis on a single file or directory.

    Uses the model: I = S * I0 * exp(-mu * thickness) * acq_time,
    where mu is fixed from NIST Cu attenuation data.

    Parameters
    ----------
    path : str
        Path to a single HDF5 file or a directory of files.
    target_counts : float
        Target intensity for acquisition recommendations.
    filestem : str
        When *path* is a directory, only process files whose name
        starts with this string (e.g. ``"Ceria"``).
    skip_frames_str : str
        Dash-separated frame indices to skip (default ``"0"``).
    darkfile : str or None
        Path to dark HDF5 file (``exchange/data`` averaged).
    dark_mask : bool
        Create dead/hot pixel mask from dark file (default True).
    maskfile : str or None
        Path to external binary mask (.tif, .npy).
    percentile_mask : float
        Percentile of pixels to keep (100 = no masking).
    min_intensity : float
        Minimum max-pixel intensity to accept a file.  Files below
        this threshold are skipped.  Default 1000.
    output_plot : str
        Path for fit plot.
    file_ext : str
        File extension filter for directory mode.
    energy_keV : float or None
        X-ray energy in keV.  Parsed from filename if not given.
    I0 : float or None
        Incident beam intensity (counts/s).  Required for single-file
        analysis.  For multi-file, if given, S is reported separately
        from S*I0.

    Returns
    -------
    dict
        mu, C, SI0, S (if I0 given), r2 (if multi-file),
        collected_data, recommendations
    """
    skip_frames = [int(x) for x in skip_frames_str.split("-") if x.strip()]

    # ── Discover files ─────────────────────────────────────────────
    datasets = discover_files(path, filestem=filestem, file_ext=file_ext)
    if not datasets:
        print("ERROR: No matching data files found.")
        return None

    is_single = len(datasets) == 1

    # ── Determine energy ───────────────────────────────────────────
    if energy_keV is None:
        for ds in datasets:
            if ds["energy_keV"] is not None:
                energy_keV = ds["energy_keV"]
                break
    if energy_keV is None:
        print("ERROR: Cannot determine energy from filenames. "
              "Use --energy to provide it.")
        return None

    mu = estimate_mu_linear(energy_keV)

    # ── Validate single-file requires I0 ───────────────────────────
    if is_single and I0 is None:
        print("ERROR: I0 must be provided for single-file analysis "
              "(use --I0).")
        return None

    # ── Load dark ──────────────────────────────────────────────────
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

    if maskfile:
        user_mask = load_mask(maskfile)
        if pixel_mask is not None:
            pixel_mask = pixel_mask * user_mask
        else:
            pixel_mask = user_mask
        n_bad = int(np.sum(user_mask < 0.5))
        print(f"User mask applied ({n_bad} bad pixels).")

    # ── Banner ─────────────────────────────────────────────────────
    mode = "SINGLE-FILE" if is_single else "MULTI-FILE"
    print("\n" + "=" * 60)
    print(f"ATTENUATION ANALYSIS ({mode})")
    print("=" * 60)
    print(f"Path           : {path}")
    if filestem:
        print(f"File stem      : {filestem}")
    print(f"Energy         : {energy_keV} keV")
    print(f"mu (NIST)      : {mu:.4f} /mm")
    if I0 is not None:
        print(f"I0 (provided)  : {I0:.2f} cts/s")
    elif not is_single:
        print(f"I0             : not provided (will fit S*I0)")
    print(f"Target counts  : {target_counts}")
    print(f"Min intensity  : {min_intensity}")
    print(f"Skip frames    : {skip_frames}")
    print(f"Dark file      : {darkfile or 'None'}")
    print(f"Dark mask      : {'Yes' if dark_mask and darkfile else 'No'}")
    print(f"User mask      : {maskfile or 'None'}")
    if percentile_mask < 100.0:
        print(f"Percentile mask: {percentile_mask}%")
    print("=" * 60)

    # ── Process files ──────────────────────────────────────────────
    print(f"\nProcessing {len(datasets)} file(s)...")
    print(f"{'File':<50} {'Att':>4} {'Thick(mm)':>10} "
          f"{'Acq(s)':>8} {'Max Int':>10} {'log(I/t)':>10} "
          f"{'C_i':>10} {'Status':>8}")
    print("-" * 116)

    collected_data = []
    for ds in datasets:
        basename = os.path.basename(ds["filepath"])
        att_pos = ds["att_pos"]
        thickness = ds["thickness_mm"]
        acq_time = ds["acq_time_s"]

        if thickness is None:
            print(f"{basename:<50} {att_pos:>4} {'?':>10} "
                  f"{acq_time:>8.1f} {'':>10} {'':>10} "
                  f"{'':>10} {'BAD ATT':>8}")
            continue

        result = read_hdf5(ds["filepath"])
        frames = result["data"]

        intensity = extract_intensity(frames, dark_mean, pixel_mask,
                                      percentile_mask, skip_frames)

        if intensity < min_intensity:
            reason = "LOW" if intensity > 0 else "ZERO"
            print(f"{basename:<50} {att_pos:>4} {thickness:>10.2f} "
                  f"{acq_time:>8.1f} {intensity:>10.1f} {'':>10} "
                  f"{'':>10} {reason:>8}")
            continue

        log_rate = math.log(intensity / acq_time)
        C_i = log_rate + mu * thickness

        collected_data.append({
            "att_pos": att_pos,
            "thickness": thickness,
            "acq_time": acq_time,
            "filepath": ds["filepath"],
            "intensity": intensity,
            "log_rate": log_rate,
            "C_i": C_i,
        })

        print(f"{basename:<50} {att_pos:>4} {thickness:>10.2f} "
              f"{acq_time:>8.1f} {intensity:>10.1f} {log_rate:>10.4f} "
              f"{C_i:>10.4f} {'OK':>8}")

    # ── Fit ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("FIT RESULTS")
    print(f"{'=' * 60}")

    if not collected_data:
        print(f"ERROR: No files with max intensity >= {min_intensity}.")
        return None

    C_values = [d["C_i"] for d in collected_data]
    C = float(np.mean(C_values))
    SI0 = math.exp(C)

    print(f"Valid data points : {len(collected_data)}")
    print(f"mu (NIST, fixed)  : {mu:.4f} /mm")
    print(f"C = log(S*I0)     : {C:.4f}")
    print(f"S*I0              : {SI0:.2f} cts/s")

    # R^2 (only meaningful for multi-file)
    r2 = None
    if len(collected_data) > 1:
        log_rates = np.array([d["log_rate"] for d in collected_data])
        thicknesses = np.array([d["thickness"] for d in collected_data])
        predicted = C - mu * thicknesses
        residuals = log_rates - predicted
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((log_rates - np.mean(log_rates)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"R^2               : {r2:.6f}")

    S = None
    if I0 is not None:
        S = SI0 / I0
        print(f"I0 (provided)     : {I0:.2f} cts/s")
        print(f"S (scattering)    : {S:.6f}")

    # ── Per-point consistency ──────────────────────────────────────
    if len(collected_data) > 1:
        C_std = float(np.std(C_values))
        C_range = max(C_values) - min(C_values)
        print(f"\nPer-point consistency:")
        print(f"  C_i std         : {C_std:.4f}")
        print(f"  C_i range       : {C_range:.4f}")
        for d in collected_data:
            residual = d["C_i"] - C
            flag = "  ***" if abs(residual) > 2 * C_std and C_std > 0 else ""
            print(f"  att{d['att_pos']} {d['acq_time']}s: "
                  f"C_i = {d['C_i']:.4f}  "
                  f"residual = {residual:+.4f}{flag}")

    # Plot
    plot_fit(collected_data, mu, C, output_plot)
    print(f"\nPlot saved to     : {output_plot}")

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
        pred_rate = SI0 * math.exp(-mu * thick)
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

    output = {
        "mu": mu,
        "C": C,
        "SI0": SI0,
        "collected_data": collected_data,
        "recommendations": recommendations,
    }
    if r2 is not None:
        output["r2"] = r2
    if S is not None:
        output["S"] = S
        output["I0"] = I0

    return output


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attenuation analysis using fixed NIST mu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Multi-file (directory)
  %(prog)s test_data/Ceria --filestem Ceria --target_intensity 50000 --darkfile dark.h5

  # Single file (I0 required)
  %(prog)s data/Ceria_63keV_att0_1p0s.h5 --target_intensity 50000 --I0 100000

  # With energy override and percentile mask
  %(prog)s data/ --filestem LaB6 --energy 63 --percentile_mask 99.99 --target_intensity 50000
""",
    )
    parser.add_argument("path", help="HDF5 file or directory of files")
    parser.add_argument("--target_intensity", type=float, required=True,
                        help="Target intensity in counts")
    parser.add_argument("--filestem", default="",
                        help="Only process files starting with this string")
    parser.add_argument("--energy", type=float, default=None,
                        help="X-ray energy in keV (fallback if not in filename)")
    parser.add_argument("--I0", type=float, default=None,
                        help="Incident beam intensity (counts/s). "
                             "Required for single-file mode.")
    parser.add_argument("--darkfile", default=None,
                        help="Path to dark HDF5 file")
    parser.add_argument("--dark_mask", type=int, default=1, choices=[0, 1],
                        help="Create mask from dark file (default: 1)")
    parser.add_argument("--maskfile", default=None,
                        help="Path to binary mask file (.tif, .npy)")
    parser.add_argument("--percentile_mask", type=float, default=100.0,
                        help="Percentile of pixels to keep (default: 100)")
    parser.add_argument("--min_intensity", type=float, default=1000,
                        help="Skip files with max intensity below this "
                             "(default: 1000)")
    parser.add_argument("--skip_frames", default="0",
                        help="Frame indices to skip, dash-separated "
                             "(default: '0')")
    parser.add_argument("--output_plot", default="attenuation_fit.png",
                        help="Path for output plot")

    args = parser.parse_args()

    analyze(
        path=args.path,
        target_counts=args.target_intensity,
        filestem=args.filestem,
        skip_frames_str=args.skip_frames,
        darkfile=args.darkfile,
        dark_mask=bool(args.dark_mask),
        maskfile=args.maskfile,
        percentile_mask=args.percentile_mask,
        min_intensity=args.min_intensity,
        output_plot=args.output_plot,
        energy_keV=args.energy,
        I0=args.I0,
    )


if __name__ == "__main__":
    main()
