"""Scattering-model attenuation analysis.

Uses the physical model:

    I_measured = S * I0 * exp(-mu * thickness) * acq_time

where mu is fixed (from NIST Cu attenuation data) and the code fits
for the product S*I0 from data.  When I0 is provided by the user,
S is reported separately.

For single-file analysis, I0 must be provided.
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyautobeam.io.hdf5_reader import read_hdf5
from pyautobeam.processing.mask import (
    apply_mask,
    create_dark_mask,
    create_percentile_mask,
    load_mask,
)
from pyautobeam.physics.attenuation import estimate_mu_linear
from pyautobeam.utils.auto_att_multiple_file import (
    ALL_ATTENUATOR_POSITIONS,
    _ATT_RE,
    _ENERGY_RE,
    att_thickness_from_pos,
    extract_intensity,
    generate_info_file,
    load_info_file,
)


# ── plotting ─────────────────────────────────────────────────────────

def plot_scattering_fit(collected_data, mu, C, output_path="scattering_fit.png"):
    """Plot data points against the fixed-slope Beer-Lambert line."""
    valid = [d for d in collected_data if d["intensity"] > 0]
    if len(valid) < 1:
        return

    thicknesses = [d["thickness"] for d in valid]
    log_rates = [d["log_rate"] for d in valid]
    n_pts = len(thicknesses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: data + fixed-slope fit
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
    ax1.set_title(f"Scattering Model Fit ({n_pts} points, mu fixed)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: residuals (C_i - C)
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

def scattering_analysis(datadir, sample, target_counts=50000,
                        skip_frames_str="0", darkfile=None, dark_mask=True,
                        maskfile=None, percentile_mask=100.0,
                        min_intensity=1000,
                        output_plot="scattering_fit.png",
                        info_file=None, file_ext=".h5",
                        energy_keV=None, I0=None):
    """Fit the scattering factor from attenuation data.

    Uses the model: I = S * I0 * exp(-mu * thickness) * acq_time,
    where mu is fixed from NIST Cu attenuation data.

    For single-file analysis, *I0* must be provided so that S can
    be determined.  For multi-file analysis, the product S*I0 is
    fitted from the data; if *I0* is also provided, S is reported
    separately.

    Parameters
    ----------
    datadir : str
        Directory containing HDF5 data files.
    sample : str
        Sample name to filter from the catalogue.
    target_counts : float
        Target intensity for acquisition recommendations.
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
        Minimum max-pixel intensity to accept a file.  Files with
        max intensity below this threshold are skipped.  Default 1000.
    output_plot : str
        Path for fit plot.
    info_file : str or None
        Path to JSON catalogue (auto-generated if missing).
    file_ext : str
        File extension filter.
    energy_keV : float or None
        X-ray energy in keV (parsed from filename if not given).
    I0 : float or None
        Incident beam intensity in counts/s.  Required for single-file
        analysis.  Optional for multi-file (if given, S is reported
        separately from S*I0).

    Returns
    -------
    dict
        mu, C, SI0, S (if I0 given), collected_data, recommendations
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
        else:
            print("ERROR: Cannot determine energy. Provide energy_keV.")
            return None

    # mu is FIXED from NIST data
    mu = estimate_mu_linear(energy_keV)

    # Check single-file requires I0
    if len(available_datasets) == 1 and I0 is None:
        print("ERROR: I0 must be provided for single-file analysis.")
        return None

    # Load dark
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

    # User mask
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
    print("SCATTERING-MODEL ATTENUATION ANALYSIS")
    print("=" * 60)
    print(f"Data directory : {datadir}")
    print(f"Sample         : {sample}")
    print(f"Energy         : {energy_keV} keV")
    print(f"mu (NIST, fixed): {mu:.4f} /mm")
    if I0 is not None:
        print(f"I0 (provided)  : {I0:.2f} cts/s")
    else:
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

    # ── Process all files ───────────────────────────────────────────
    print(f"\nProcessing {len(available_datasets)} datasets...")
    print(f"{'File':<50} {'Att':>4} {'Thick(mm)':>10} "
          f"{'Acq(s)':>8} {'Max Int':>10} {'log(I/t)':>10} "
          f"{'C_i':>10} {'Status':>8}")
    print("-" * 116)

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

    if len(collected_data) == 0:
        print("ERROR: No files with max intensity >= "
              f"{min_intensity} counts.")
        return None

    C_values = [d["C_i"] for d in collected_data]
    C = float(np.mean(C_values))
    SI0 = math.exp(C)

    # R^2 for fixed-mu model
    log_rates = np.array([d["log_rate"] for d in collected_data])
    thicknesses = np.array([d["thickness"] for d in collected_data])
    predicted = C - mu * thicknesses
    residuals = log_rates - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_rates - np.mean(log_rates)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"Valid data points : {len(collected_data)}")
    print(f"mu (NIST, fixed)  : {mu:.4f} /mm")
    print(f"C = log(S*I0)     : {C:.4f}")
    print(f"S*I0              : {SI0:.2f} cts/s")
    if len(collected_data) > 1:
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
    plot_scattering_fit(collected_data, mu, C, output_plot)
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

    result = {
        "mu": mu,
        "C": C,
        "SI0": SI0,
        "r2": r2,
        "collected_data": collected_data,
        "recommendations": recommendations,
    }
    if S is not None:
        result["S"] = S
        result["I0"] = I0

    return result
