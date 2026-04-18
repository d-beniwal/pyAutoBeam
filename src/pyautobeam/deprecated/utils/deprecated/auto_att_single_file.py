"""Single-file attenuation analysis with CLI.

Analyzes a single HDF5 data file: parses metadata from its filename,
estimates the Cu attenuation coefficient from NIST data, processes the
frames (dark subtraction, masking), computes I0, reports per-pixel
intensity statistics against a user target, and recommends acquisition
times for all attenuator positions.
"""

import argparse
import math
import os
import re
import sys

import numpy as np

from pyautobeam.io.hdf5_reader import read_hdf5
from pyautobeam.physics.attenuation import estimate_mu_linear
from pyautobeam.physics.beer_lambert import beer_lambert_intensity
from pyautobeam.processing.dark import subtract_dark
from pyautobeam.processing.mask import (
    apply_mask,
    create_dark_mask,
    create_percentile_mask,
    load_mask,
)
from pyautobeam.utils.auto_att_multiple_file import (
    ALL_ATTENUATOR_POSITIONS,
    _ATT_RE,
    att_thickness_from_pos,
)

_ENERGY_RE = re.compile(r"(\d+)keV")


# ── filename parsing ────────────────────────────────────────────────

def parse_filename(filepath, energy_override=None):
    """Extract energy, attenuator position, and acquisition time from
    an HDF5 filename.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    energy_override : float or None
        If provided, used instead of parsing from filename.

    Returns
    -------
    dict
        Keys: energy_keV, att_pos, acq_time_s, thickness_mm
    """
    basename = os.path.basename(filepath)

    # Energy
    if energy_override is not None:
        energy_keV = float(energy_override)
    else:
        m = _ENERGY_RE.search(basename)
        if m:
            energy_keV = float(m.group(1))
        else:
            raise ValueError(
                f"Cannot parse energy from filename '{basename}'. "
                "Use --energy to provide it."
            )

    # Attenuator position and acquisition time
    m = _ATT_RE.search(basename)
    if not m:
        raise ValueError(
            f"Cannot parse attenuator/acq_time from filename '{basename}'. "
            "Expected pattern like 'att3_1p0s'."
        )
    att_pos = int(m.group(1))
    acq_time_s = float(f"{m.group(2)}.{m.group(3)}")

    thickness_mm = att_thickness_from_pos(att_pos)
    if thickness_mm is None:
        raise ValueError(
            f"Unknown attenuator position {att_pos}. "
            f"Known positions: {ALL_ATTENUATOR_POSITIONS}"
        )

    return {
        "energy_keV": energy_keV,
        "att_pos": att_pos,
        "acq_time_s": acq_time_s,
        "thickness_mm": thickness_mm,
    }


def parse_skip_frames(skip_str):
    """Parse a dash-separated string of frame indices.

    ``"0-1-3-9"`` → ``[0, 1, 3, 9]``
    """
    return [int(x) for x in skip_str.split("-") if x.strip()]


# ── pixel statistics ────────────────────────────────────────────────

def _build_bin_headers(target):
    """Build column headers combining % range and intensity range."""
    t = float(target)
    return [
        f"0-25% (0-{0.25*t:.0f})",
        f"25-50% ({0.25*t:.0f}-{0.5*t:.0f})",
        f"50-75% ({0.5*t:.0f}-{0.75*t:.0f})",
        f"75-100% ({0.75*t:.0f}-{t:.0f})",
        f">100% (>{t:.0f})",
    ]


def compute_pixel_stats(pixels, target):
    """Compute intensity bin counts for all valid pixels.

    Returns
    -------
    dict
        total, counts (list of int), pcts (list of float)
    """
    total = len(pixels)
    if total == 0:
        return {"total": 0, "counts": [0] * 5, "pcts": [0.0] * 5}

    t = float(target)
    edges = [-np.inf, 0.25 * t, 0.5 * t, 0.75 * t, 1.0 * t, np.inf]

    counts = []
    pcts = []
    for i in range(5):
        count = int(np.sum((pixels >= edges[i]) & (pixels < edges[i + 1])))
        counts.append(count)
        pcts.append(100.0 * count / total)

    return {"total": total, "counts": counts, "pcts": pcts}


def _format_cell(count, pct, width):
    """Format a cell showing count and percentage."""
    return f"{count:,}({pct:.1f}%)"


def print_combined_stats_table(per_frame_stats, pool_stats, target,
                               frame_indices):
    """Print stats with frames as rows and intensity bins as columns.

    Parameters
    ----------
    frame_indices : list of int
        Original frame indices (before any were skipped).
    """
    headers = _build_bin_headers(target)
    col_widths = [max(len(h), 18) for h in headers]

    # Header row
    row_label_w = 10
    hdr = f"  {'Frame':<{row_label_w}}"
    for h, w in zip(headers, col_widths):
        hdr += f"  {h:>{w}}"
    print(hdr)
    print(f"  {'-' * (row_label_w + sum(w + 2 for w in col_widths))}")

    # Per-frame rows
    for i, orig_idx in enumerate(frame_indices):
        row = f"  {f'F{orig_idx}':<{row_label_w}}"
        for j, w in enumerate(col_widths):
            cell = _format_cell(per_frame_stats[i]['counts'][j],
                                per_frame_stats[i]['pcts'][j], w)
            row += f"  {cell:>{w}}"
        print(row)

    # Separator + pooled row
    print(f"  {'-' * (row_label_w + sum(w + 2 for w in col_widths))}")
    row = f"  {'ALL':<{row_label_w}}"
    for j, w in enumerate(col_widths):
        cell = _format_cell(pool_stats['counts'][j],
                            pool_stats['pcts'][j], w)
        row += f"  {cell:>{w}}"
    print(row)


# ── main analysis ───────────────────────────────────────────────────

def analyze_single_file(
    filepath,
    target_intensity,
    energy_override=None,
    darkfile=None,
    dark_mask=True,
    maskfile=None,
    percentile_mask=100.0,
    skip_frames_str="0",
    active_learn=False,
):
    """Analyze a single HDF5 attenuation data file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 data file.
    target_intensity : float
        Target intensity in counts.
    energy_override : float or None
        X-ray energy in keV (overrides filename parsing).
    darkfile : str or None
        Path to a separate dark HDF5 file.
    dark_mask : bool
        If True (default) and *darkfile* is provided, create a
        dead/hot pixel mask from the dark file and apply it.
    maskfile : str or None
        Path to a binary mask file (.tif, .npy).
    percentile_mask : float
        Percentile of pixels to keep (100 = no masking).
    skip_frames_str : str
        Dash-separated frame indices to exclude (e.g. "0-1-3").
    active_learn : bool
        If True, suggest two next (att_pos, acq_time) data points.

    Returns
    -------
    dict
        energy_keV, att_pos, acq_time_s, thickness_mm, mu, I0,
        pixel_stats, recommendations, suggestions (if active_learn)
    """

    # ── Step 1: Parse filename ──────────────────────────────────────
    meta = parse_filename(filepath, energy_override)
    energy_keV = meta["energy_keV"]
    att_pos = meta["att_pos"]
    acq_time_s = meta["acq_time_s"]
    thickness_mm = meta["thickness_mm"]

    # ── Step 2: Estimate mu from NIST table ─────────────────────────
    mu = estimate_mu_linear(energy_keV)

    print("=" * 60)
    print("SINGLE-FILE ATTENUATION ANALYSIS")
    print("=" * 60)
    print(f"File           : {os.path.basename(filepath)}")
    print(f"Energy         : {energy_keV} keV")
    print(f"Attenuator     : pos {att_pos}  ({thickness_mm:.2f} mm Cu)")
    print(f"Acq. time      : {acq_time_s} s")
    print(f"mu (NIST)      : {mu:.4f} /mm")
    print(f"Target         : {target_intensity} counts")
    print("=" * 60)

    # ── Step 3: Load data and dark ──────────────────────────────────
    result = read_hdf5(filepath)
    data = np.array(result["data"], dtype=np.float32)

    if darkfile:
        dark_result = read_hdf5(darkfile)
        dark_frames = dark_result["data"]
        if dark_frames is not None:
            dark = np.mean(dark_frames.astype(np.float32), axis=0)
        else:
            dark = None
            print("WARNING: No data found in dark file.")
    else:
        dark = result["dark"]

    if dark is None:
        print("INFO: No dark data available. Proceeding without "
              "dark subtraction.")

    print(f"Loaded {data.shape[0]} frames, "
          f"frame size {data.shape[1]} x {data.shape[2]} px")

    # ── Step 4: Skip frames ─────────────────────────────────────────
    skip_indices = parse_skip_frames(skip_frames_str)
    valid_skip = [i for i in skip_indices if i < data.shape[0]]
    # Track which original frame indices are kept
    kept_frame_indices = [i for i in range(data.shape[0])
                          if i not in valid_skip]
    if valid_skip:
        data = np.delete(data, valid_skip, axis=0)
        print(f"Skipped frames : {valid_skip}  "
              f"({data.shape[0]} frames remaining)")

    if data.shape[0] == 0:
        print("ERROR: No frames remaining after skip.")
        return None

    # Check that data has signal before proceeding
    if np.max(data) <= 0:
        print("ERROR: No data since all pixels are <= 0.")
        return None

    # ── Step 5: Dark subtraction ────────────────────────────────────
    if dark is not None:
        data = subtract_dark(data, dark)
        np.clip(data, 0, None, out=data)
        print("Dark subtraction applied (negatives clipped to 0).")

        if np.max(data) <= 0:
            print("ERROR: No data since all pixels are <= 0 "
                  "(after dark removal).")
            return None

    # ── Step 6: Apply masks ─────────────────────────────────────────
    # Build combined valid-pixel mask (2D, True = valid)
    frame_shape = data.shape[1:]
    valid_mask = np.ones(frame_shape, dtype=bool)

    if dark_mask and darkfile:
        dm, dm_info = create_dark_mask(darkfile)
        data = apply_mask(data, dm)
        valid_mask &= (dm > 0.5)

    if maskfile:
        user_mask = load_mask(maskfile)
        data = apply_mask(data, user_mask)
        valid_mask &= (user_mask > 0.5)
        n_bad = int(np.sum(~(user_mask > 0.5)))
        print(f"User mask applied ({n_bad} bad pixels masked).")

    if percentile_mask < 100.0:
        pct_mask = create_percentile_mask(data, percentile=percentile_mask)
        data = apply_mask(data, pct_mask)
        valid_mask &= (pct_mask > 0.5)
        n_hot = int(np.sum(~(pct_mask > 0.5)))
        print(f"Percentile mask ({percentile_mask}%) applied "
              f"({n_hot} hot pixels masked).")

    n_valid = int(np.sum(valid_mask))
    n_total = valid_mask.size
    print(f"Valid pixels   : {n_valid:,} / {n_total:,} "
          f"({100.0 * n_valid / n_total:.1f}%)")

    # ── Step 7: Compute I0 ──────────────────────────────────────────
    # Use the max intensity across valid pixels.  Hot-pixel removal is
    # handled upstream by the masking steps, so max of the remaining
    # pixels is the true peak signal — the value that must stay below
    # the detector saturation limit.
    all_valid_pixels = []
    frame_max = []
    for i in range(data.shape[0]):
        frame_pixels = data[i][valid_mask]
        all_valid_pixels.append(frame_pixels)
        frame_max.append(float(np.max(frame_pixels)) if len(frame_pixels) > 0 else 0.0)

    pool = np.concatenate(all_valid_pixels)
    peak_intensity = float(np.max(pool)) if len(pool) > 0 else 0.0

    # I = I0 * exp(-mu * x) * t  =>  I0 = I / (t * exp(-mu * x))
    transmission = math.exp(-mu * thickness_mm)
    I0 = peak_intensity / (acq_time_s * transmission) if transmission > 0 else 0.0

    print(f"\nMax intensity  : {peak_intensity:.1f} counts")
    print(f"Transmission factor exp(-mu*x): {transmission:.6f}")
    print(f"Estimated I0   : {I0:.2f} counts/s")

    # ── Step 8: Pixel intensity statistics ──────────────────────────
    n_frames = data.shape[0]

    print("\n" + "=" * 60)
    print("PIXEL INTENSITY STATISTICS")
    print(f"(target = {target_intensity} counts, all valid pixels)")
    print("=" * 60)

    per_frame_stats = []
    for i in range(n_frames):
        stats = compute_pixel_stats(all_valid_pixels[i], target_intensity)
        per_frame_stats.append(stats)

    pool_pixels = np.concatenate(all_valid_pixels)
    pool_stats = compute_pixel_stats(pool_pixels, target_intensity)

    print_combined_stats_table(per_frame_stats, pool_stats,
                               target_intensity, kept_frame_indices)

    # ── Step 9: Acquisition time recommendations ────────────────────
    target_90 = 0.9 * target_intensity
    print("\n" + "=" * 60)
    print(f"ACQUISITION TIME RECOMMENDATIONS (for 90% of target = "
          f"{target_90:.0f} counts)")
    print("=" * 60)
    print(f"{'Att Pos':<10} {'Thickness (mm)':<16} "
          f"{'Rec. Time (s)':<16} {'Pred. Counts':<14} {'Note'}")
    print("-" * 70)

    recommendations = {}
    for pos in ALL_ATTENUATOR_POSITIONS:
        thick = att_thickness_from_pos(pos)
        if thick is None:
            continue
        pred_rate = beer_lambert_intensity(I0, mu, thick)
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

    # ── Step 10: Active learning suggestions ────────────────────────
    suggestions = []
    if active_learn:
        print("\n" + "=" * 60)
        print("ACTIVE LEARNING SUGGESTIONS")
        print("=" * 60)

        low = 0.2 * target_intensity
        high = 0.7 * target_intensity

        # 3 evenly spaced target intensities across [20%, 70%] of target
        target_values = [
            low + (high - low) * k / 2.0 for k in range(3)
        ]
        # target_values = [20%, 45%, 70%] of target_intensity

        # Determine which target value the current measurement is
        # closest to, then suggest data points for the other two
        current_intensity = peak_intensity
        closest_idx = min(
            range(len(target_values)),
            key=lambda i: abs(target_values[i] - current_intensity),
        )

        needed_targets = [
            tv for i, tv in enumerate(target_values) if i != closest_idx
        ]

        print(f"Current intensity     : {current_intensity:.1f} counts")
        print(f"Target range          : [{low:.0f}, {high:.0f}] counts "
              f"(20-70% of {target_intensity})")
        print(f"Current closest to    : {target_values[closest_idx]:.0f} "
              f"counts ({100*target_values[closest_idx]/target_intensity:.0f}%"
              f" of target)")
        print()

        for j, i_target in enumerate(needed_targets, 1):
            best_pos = None
            best_time = None
            best_score = float("inf")

            for pos in ALL_ATTENUATOR_POSITIONS:
                thick = att_thickness_from_pos(pos)
                if thick is None:
                    continue
                pred_rate = beer_lambert_intensity(I0, mu, thick)
                if pred_rate <= 0:
                    continue
                t_needed = i_target / pred_rate

                # Prefer practical acquisition times (0.01s to 100s)
                if t_needed < 0.005 or t_needed > 200:
                    continue

                # Score: prefer times closer to 1s (log-distance)
                score = abs(math.log10(t_needed))
                if score < best_score:
                    best_score = score
                    best_pos = pos
                    best_time = t_needed

            if best_pos is not None:
                thick = att_thickness_from_pos(best_pos)
                pred_counts = beer_lambert_intensity(I0, mu, thick) * best_time
                suggestion = {
                    "att_pos": best_pos,
                    "thickness_mm": thick,
                    "acq_time_s": round(best_time, 3),
                    "predicted_intensity": pred_counts,
                    "target_fraction": i_target / target_intensity,
                }
                suggestions.append(suggestion)
                print(f"  Suggestion {j}: att{best_pos} "
                      f"({thick:.2f} mm Cu), "
                      f"acq_time = {best_time:.3f} s")
                print(f"    Predicted intensity: {pred_counts:.0f} counts "
                      f"({100*pred_counts/target_intensity:.0f}% of target)")
            else:
                print(f"  Suggestion {j}: No practical combination found.")

    # ── Return summary ──────────────────────────────────────────────
    result = {
        "energy_keV": energy_keV,
        "att_pos": att_pos,
        "acq_time_s": acq_time_s,
        "thickness_mm": thickness_mm,
        "mu": mu,
        "I0": I0,
        "peak_intensity": peak_intensity,
        "pixel_stats_pooled": pool_stats,
        "recommendations": recommendations,
    }
    if active_learn:
        result["suggestions"] = suggestions

    return result


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single HDF5 attenuation data file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s data/Ceria_63keV_att0_1p0s.h5 --target_intensity 50000
  %(prog)s data/scan.h5 --target_intensity 50000 --energy 63 --percentile_mask 99.99
  %(prog)s data/scan.h5 --target_intensity 50000 --darkfile dark.h5 --active_learn
""",
    )
    parser.add_argument(
        "filepath",
        help="Path to the HDF5 data file",
    )
    parser.add_argument(
        "--target_intensity",
        type=float,
        required=True,
        help="Target intensity in counts",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=None,
        help="X-ray energy in keV (fallback if not in filename)",
    )
    parser.add_argument(
        "--darkfile",
        default=None,
        help="Path to separate dark HDF5 file",
    )
    parser.add_argument(
        "--dark_mask",
        type=int,
        default=1,
        choices=[0, 1],
        help="Create dead/hot pixel mask from dark file (default: 1 = yes)",
    )
    parser.add_argument(
        "--maskfile",
        default=None,
        help="Path to binary mask file (.tif, .npy)",
    )
    parser.add_argument(
        "--percentile_mask",
        type=float,
        default=100.0,
        help="Percentile of pixels to keep (default: 100 = no masking)",
    )
    parser.add_argument(
        "--skip_frames",
        default="0",
        help="Frame indices to exclude, dash-separated (default: '0')",
    )
    parser.add_argument(
        "--active_learn",
        action="store_true",
        help="Suggest two next data points for even intensity coverage",
    )

    args = parser.parse_args()

    analyze_single_file(
        filepath=args.filepath,
        target_intensity=args.target_intensity,
        energy_override=args.energy,
        darkfile=args.darkfile,
        dark_mask=bool(args.dark_mask),
        maskfile=args.maskfile,
        percentile_mask=args.percentile_mask,
        skip_frames_str=args.skip_frames,
        active_learn=args.active_learn,
    )


if __name__ == "__main__":
    main()
