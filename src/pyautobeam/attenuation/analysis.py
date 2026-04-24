"""Attenuation analysis using the scattering model.

Physical model::

    I_measured = S * I0 * exp(-mu * thickness) * acq_time

Single-file mode:
    mu is fixed from NIST data.  S*I0 is computed directly from the
    one measurement.

Multi-file mode:
    Both mu and S*I0 are fitted via linear regression in log-space,
    giving the best fit to the data.  NIST mu is reported for
    comparison.

In both cases, predictions use the fitted/computed S*I0 and mu.
"""

import argparse
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

import h5py

from pyautobeam.io.hdf5_reader import read_hdf5
from pyautobeam.processing.dark import subtract_dark
from pyautobeam.processing.mask import (
    apply_mask,
    create_dark_mask,
    create_percentile_mask,
    load_mask,
)
from pyautobeam.attenuation.nist_data import estimate_mu_linear


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
    """Find data files from a path (single file or directory)."""
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
                      skip_frames=1):
    """Extract the max intensity from a frame stack.

    Parameters
    ----------
    skip_frames : int
        Number of frames to skip from the start of the stack.
        Default 1 (skip the first frame).
    """
    data = np.array(frames, dtype=np.float32)

    n_skip = max(0, int(skip_frames))
    if n_skip >= data.shape[0]:
        return 0.0
    if n_skip > 0:
        data = data[n_skip:]

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

def plot_fit(collected_data, mu, C, mu_nist=None,
             output_path="attenuation_fit.png"):
    """Plot data points against the model line."""
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
    SI0 = math.exp(C)

    # Fitted line
    y_fit = C - mu * x_fit
    fit_label = f"mu={mu:.4f} /mm"
    if mu_nist is not None and abs(mu - mu_nist) > 0.0001:
        fit_label += " (fitted)"
    else:
        fit_label += " (NIST)"
    ax1.plot(x_fit, y_fit, "r--", linewidth=2,
             label=f"{fit_label}\nS*I0={SI0:.2e} cts/s")

    # NIST reference line (if different from fitted)
    if mu_nist is not None and abs(mu - mu_nist) > 0.0001:
        y_nist = C - mu_nist * x_fit
        ax1.plot(x_fit, y_nist, "g:", linewidth=1.5, alpha=0.6,
                 label=f"mu={mu_nist:.4f} /mm (NIST)")

    ax1.set_xlabel("Copper Thickness (mm)")
    ax1.set_ylabel("log(Intensity / time)")
    ax1.set_title(f"Attenuation Fit ({n_pts} points)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    if n_pts > 1:
        residuals = [lr - (C - mu * t) for lr, t in zip(log_rates, thicknesses)]
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
        ax2.set_ylabel("Residual")
        ax2.set_title("Fit Residuals")
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "Single point\n(no residuals)",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Fit Residuals")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ── main analysis ────────────────────────────────────────────────────

def analyze(path, target_counts=50000, filestem="",
            skip_frames=1, darkfile=None, dark_mask=True,
            maskfile=None, percentile_mask=100.0,
            min_intensity=1000,
            output_plot=None,
            file_ext=".h5", energy_keV=None):
    """Run attenuation analysis on a single file or directory.

    Uses the model: I = S * I0 * exp(-mu * thickness) * acq_time.

    - **Single file**: mu is fixed from NIST, S*I0 computed directly.
    - **Multiple files**: mu and S*I0 are fitted via linear regression.
      NIST mu is reported for comparison.

    Parameters
    ----------
    path : str
        Path to a single HDF5 file or a directory of files.
    target_counts : float
        Target intensity for acquisition recommendations.
    filestem : str
        When *path* is a directory, only process files whose name
        starts with this string.
    skip_frames : int
        Number of frames to skip from the start of each stack
        (default 1).  Set to 0 to use all frames.
    darkfile : str or None
        Path to dark HDF5 file (``exchange/data`` averaged).
    dark_mask : bool
        Create dead/hot pixel mask from dark file (default True).
    maskfile : str or None
        Path to external binary mask (.tif, .npy).
    percentile_mask : float
        Percentile of pixels to keep (100 = no masking).
    min_intensity : float
        Minimum max-pixel intensity to accept a file.  Default 1000.
    output_plot : str or None
        Path for fit plot.  If None (default), no plot is saved.
    file_ext : str
        File extension filter for directory mode.
    energy_keV : float or None
        X-ray energy in keV.  Parsed from filename if not given.

    Returns
    -------
    dict
        mu, mu_nist, C, SI0, r2 (if multi-file),
        collected_data, recommendations
    """
    skip_frames = max(0, int(skip_frames))

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

    mu_nist = estimate_mu_linear(energy_keV)

    # ── Load dark ──────────────────────────────────────────────────
    _DARK_KEYS = ["exchange/dark", "exchange/data_dark", "exchange/dark_data"]

    dark_mean = None
    pixel_mask = None
    dark_source = None

    print("\n" + "-" * 60)
    print("PREPROCESSING: Dark & Mask")
    print("-" * 60)

    if darkfile:
        # External dark file provided
        dark_result = read_hdf5(darkfile)
        dark_frames = dark_result["data"]
        if dark_frames is not None:
            dark_mean = np.mean(dark_frames.astype(np.float32), axis=0)
            dark_source = os.path.basename(darkfile)
            print(f"Dark loaded from {dark_source} "
                  f"({dark_frames.shape[0]} frames averaged).")
            print("Dark subtraction will be applied to all data files.")
        else:
            print("WARNING: No data found in dark file.")

        if dark_mask and dark_mean is not None:
            pixel_mask, _ = create_dark_mask(darkfile)

    else:
        # No external dark file — look inside the first data file
        first_filepath = datasets[0]["filepath"]
        print(f"No dark file provided. Checking "
              f"{os.path.basename(first_filepath)} for dark data...")

        with h5py.File(first_filepath, "r") as f:
            for key in _DARK_KEYS:
                if key in f:
                    dark_frames = np.array(f[key], dtype=np.float32)
                    if np.max(dark_frames) > 0:
                        dark_mean = np.mean(dark_frames, axis=0)
                        dark_source = f"{os.path.basename(first_filepath)}:{key}"
                        print(f"Dark found in '{key}' "
                              f"({dark_frames.shape[0]} frames averaged).")
                        print("Dark subtraction will be applied to "
                              "all data files.")
                        break
                    else:
                        print(f"Dark container '{key}' exists but "
                              "contains all zeros. Skipping.")

        if dark_mean is None:
            print("No usable dark data found. "
                  "Proceeding without dark subtraction.")

    if maskfile:
        user_mask = load_mask(maskfile)
        # Mask convention: 0 = good, 1 = bad. Combine masks via logical OR
        # (a pixel is bad if any source flags it).
        if pixel_mask is not None:
            pixel_mask = np.maximum(pixel_mask, user_mask)
        else:
            pixel_mask = user_mask
        n_bad = int(np.sum(user_mask > 0.5))
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
    print(f"mu (NIST)      : {mu_nist:.4f} /mm")
    if is_single:
        print(f"Fitting        : S*I0 (mu fixed from NIST)")
    else:
        print(f"Fitting        : mu and S*I0 (linear regression)")
    print(f"Target counts  : {target_counts}")
    print(f"Min intensity  : {min_intensity}")
    print(f"Skip first N   : {skip_frames}")
    print(f"Dark file      : {darkfile or 'None'}")
    print(f"Dark mask      : {'Yes' if dark_mask and darkfile else 'No'}")
    print(f"User mask      : {maskfile or 'None'}")
    if percentile_mask < 100.0:
        print(f"Percentile mask: {percentile_mask}%")
    print("=" * 60)

    # ── Process files ──────────────────────────────────────────────
    print(f"\nProcessing {len(datasets)} file(s)...")
    print(f"{'File':<50} {'Att':>4} {'Thick(mm)':>10} "
          f"{'Acq(s)':>8} {'Max Int':>10} {'log(I/t)':>10} {'Status':>8}")
    print("-" * 96)

    collected_data = []
    for ds in datasets:
        basename = os.path.basename(ds["filepath"])
        att_pos = ds["att_pos"]
        thickness = ds["thickness_mm"]
        acq_time = ds["acq_time_s"]

        if thickness is None:
            print(f"{basename:<50} {att_pos:>4} {'?':>10} "
                  f"{acq_time:>8.1f} {'':>10} {'':>10} {'BAD ATT':>8}")
            continue

        result = read_hdf5(ds["filepath"])
        frames = result["data"]

        intensity = extract_intensity(frames, dark_mean, pixel_mask,
                                      percentile_mask, skip_frames)

        if intensity < min_intensity:
            reason = "LOW" if intensity > 0 else "ZERO"
            print(f"{basename:<50} {att_pos:>4} {thickness:>10.2f} "
                  f"{acq_time:>8.1f} {intensity:>10.1f} {'':>10} "
                  f"{reason:>8}")
            continue

        log_rate = math.log(intensity / acq_time)

        collected_data.append({
            "att_pos": att_pos,
            "thickness": thickness,
            "acq_time": acq_time,
            "filepath": ds["filepath"],
            "intensity": intensity,
            "log_rate": log_rate,
        })

        print(f"{basename:<50} {att_pos:>4} {thickness:>10.2f} "
              f"{acq_time:>8.1f} {intensity:>10.1f} {log_rate:>10.4f} "
              f"{'OK':>8}")

    # ── Fit ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("FIT RESULTS")
    print(f"{'=' * 60}")

    if not collected_data:
        print(f"ERROR: No files with max intensity >= {min_intensity}.")
        return None

    thicknesses = np.array([d["thickness"] for d in collected_data])
    log_rates = np.array([d["log_rate"] for d in collected_data])

    r2 = None

    if len(collected_data) == 1:
        # Single file: use NIST mu, compute S*I0 directly
        mu = mu_nist
        C = log_rates[0] + mu * thicknesses[0]
        SI0 = math.exp(C)

        print(f"Valid data points : 1")
        print(f"mu (NIST, fixed)  : {mu:.4f} /mm")
        print(f"C = log(S*I0)     : {C:.4f}")
        print(f"S*I0              : {SI0:.2f} cts/s")

    else:
        # Multi-file: fit mu and S*I0 via linear regression
        # log(I/t) = log(S*I0) - mu * thickness
        #          = C         - mu * thickness
        slope, intercept, r_value, _, _ = linregress(thicknesses, log_rates)
        mu = -slope
        C = intercept
        SI0 = math.exp(C)
        r2 = r_value ** 2

        # Also compute what S*I0 would be with NIST mu (for comparison)
        C_nist_values = log_rates + mu_nist * thicknesses
        C_nist = float(np.mean(C_nist_values))
        SI0_nist = math.exp(C_nist)

        print(f"Valid data points : {len(collected_data)}")
        print()
        print("--- Fitted (linear regression) ---")
        print(f"  mu              : {mu:.4f} /mm")
        print(f"  C = log(S*I0)   : {C:.4f}")
        print(f"  S*I0            : {SI0:.2f} cts/s")
        print(f"  R^2             : {r2:.6f}")
        print()
        print("--- NIST reference ---")
        print(f"  mu (NIST)       : {mu_nist:.4f} /mm")
        print(f"  S*I0 (NIST mu)  : {SI0_nist:.2f} cts/s")
        mu_diff = 100.0 * abs(mu - mu_nist) / mu_nist
        print(f"  mu difference   : {mu_diff:.1f}%")

        # Per-point residuals
        predicted = C - mu * thicknesses
        residuals = log_rates - predicted
        if len(residuals) > 2:
            std_r = float(np.std(residuals))
            print(f"\nPer-point residuals:")
            for i, d in enumerate(collected_data):
                flag = ("  ***" if abs(residuals[i]) > 2 * std_r
                        and std_r > 0 else "")
                print(f"  att{d['att_pos']} {d['acq_time']}s: "
                      f"residual = {residuals[i]:+.4f}{flag}")

    # Plot
    if output_plot:
        plot_fit(collected_data, mu, C,
                 mu_nist=mu_nist if len(collected_data) > 1 else None,
                 output_path=output_plot)
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
        "mu_nist": mu_nist,
        "C": C,
        "SI0": SI0,
        "collected_data": collected_data,
        "recommendations": recommendations,
    }
    if r2 is not None:
        output["r2"] = r2

    return output


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attenuation analysis with scattering model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Multi-file (fits mu and S*I0)
  %(prog)s --datapath test_data/Ceria --filestem Ceria --target_intensity 50000 --darkfile dark.h5

  # Single file (mu fixed from NIST)
  %(prog)s --datapath data/Ceria_63keV_att0_1p0s.h5 --target_intensity 50000

  # With options
  %(prog)s --datapath data/ --filestem LaB6 --energy 63 --percentile_mask 99.99 --target_intensity 50000
""",
    )
    parser.add_argument("--datapath", required=True,
                        help="HDF5 file or directory of files")
    parser.add_argument("--target_intensity", type=float, required=True,
                        help="Target intensity in counts")
    parser.add_argument("--filestem", default="",
                        help="Only process files starting with this string")
    parser.add_argument("--energy", type=float, default=None,
                        help="X-ray energy in keV (fallback if not in filename)")
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
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Number of frames to skip from the start "
                             "(default: 1)")
    parser.add_argument("--output_plot", default=None,
                        help="Path for output plot (not saved if omitted)")

    args = parser.parse_args()

    analyze(
        path=args.datapath,
        target_counts=args.target_intensity,
        filestem=args.filestem,
        skip_frames=args.skip_frames,
        darkfile=args.darkfile,
        dark_mask=bool(args.dark_mask),
        maskfile=args.maskfile,
        percentile_mask=args.percentile_mask,
        min_intensity=args.min_intensity,
        output_plot=args.output_plot,
        energy_keV=args.energy,
    )


if __name__ == "__main__":
    main()
