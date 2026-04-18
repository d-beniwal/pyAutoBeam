"""Per-frame pixel intensity statistics.

Preprocesses an HDF5 data file (dark subtraction, masking) and reports
per-frame pixel counts in user-defined intensity bins.
"""

import argparse
import os

import h5py
import numpy as np

from pyautobeam.io.hdf5_reader import read_hdf5
from pyautobeam.processing.dark import subtract_dark
from pyautobeam.processing.mask import (
    apply_mask,
    create_dark_mask,
    create_percentile_mask,
    load_mask,
)

_DARK_KEYS = ["exchange/dark", "exchange/data_dark", "exchange/dark_data"]


def frame_stats(path, lowI=500, highI=10000, targetI=40000,
                skip_frames_str="0", darkfile=None, dark_mask=True,
                maskfile=None, percentile_mask=100.0):
    """Compute per-frame pixel intensity statistics.

    Parameters
    ----------
    path : str
        Path to an HDF5 data file.
    lowI : float
        Lower intensity threshold.
    highI : float
        Upper intensity threshold.
    targetI : float
        Target intensity threshold.
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

    Returns
    -------
    dict
        kept_indices, per_frame (list of dicts), totals
    """
    skip_frames = [int(x) for x in skip_frames_str.split("-") if x.strip()]

    # ── Load data ──────────────────────────────────────────────────
    result = read_hdf5(path)
    data = np.array(result["data"], dtype=np.float32)

    print(f"Loaded {data.shape[0]} frames, "
          f"frame size {data.shape[1]} x {data.shape[2]} px "
          f"from {os.path.basename(path)}")

    # ── Skip frames ───────────────────────────────────────────────
    kept_indices = [i for i in range(data.shape[0]) if i not in skip_frames]
    valid_skip = [i for i in skip_frames if i < data.shape[0]]
    if valid_skip:
        data = np.delete(data, valid_skip, axis=0)
        print(f"Skipped frames: {valid_skip} "
              f"({data.shape[0]} frames remaining)")

    if data.shape[0] == 0:
        print("ERROR: No frames remaining after skip.")
        return None

    # ── Dark subtraction ──────────────────────────────────────────
    dark_mean = None
    pixel_mask = None

    print("\n" + "-" * 60)
    print("PREPROCESSING: Dark & Mask")
    print("-" * 60)

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

    else:
        print(f"No dark file provided. Checking "
              f"{os.path.basename(path)} for dark data...")
        with h5py.File(path, "r") as f:
            for key in _DARK_KEYS:
                if key in f:
                    dark_frames = np.array(f[key], dtype=np.float32)
                    if np.max(dark_frames) > 0:
                        dark_mean = np.mean(dark_frames, axis=0)
                        print(f"Dark found in '{key}' "
                              f"({dark_frames.shape[0]} frames averaged).")
                        break
                    else:
                        print(f"Dark container '{key}' exists but "
                              "contains all zeros. Skipping.")

        if dark_mean is None:
            print("No usable dark data found. "
                  "Proceeding without dark subtraction.")

    if dark_mean is not None:
        data = subtract_dark(data, dark_mean)
        np.clip(data, 0, None, out=data)
        print("Dark subtraction applied (negatives clipped to 0).")

    # ── Masks ─────────────────────────────────────────────────────
    if maskfile:
        user_mask = load_mask(maskfile)
        if pixel_mask is not None:
            pixel_mask = pixel_mask * user_mask
        else:
            pixel_mask = user_mask
        n_bad = int(np.sum(user_mask < 0.5))
        print(f"User mask applied ({n_bad} bad pixels).")

    if pixel_mask is not None:
        data = apply_mask(data, pixel_mask)

    if percentile_mask < 100.0:
        pct_mask = create_percentile_mask(data, percentile=percentile_mask)
        data = apply_mask(data, pct_mask)
        n_hot = int(np.sum(pct_mask < 0.5))
        print(f"Percentile mask ({percentile_mask}%) applied "
              f"({n_hot} hot pixels masked).")

    # ── Build valid pixel mask (for counting) ─────────────────────
    frame_shape = data.shape[1:]
    valid_mask = np.ones(frame_shape, dtype=bool)
    if pixel_mask is not None:
        valid_mask &= (pixel_mask > 0.5)
    if percentile_mask < 100.0:
        valid_mask &= (pct_mask > 0.5)

    n_valid = int(np.sum(valid_mask))
    print(f"Valid pixels: {n_valid:,} / {valid_mask.size:,}")

    # ── Compute statistics ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"PIXEL INTENSITY STATISTICS")
    print(f"  lowI = {lowI}, highI = {highI}, targetI = {targetI}")
    print(f"{'=' * 70}")

    col_low = f"< {lowI:.0f}"
    col_mid = f"{lowI:.0f}-{highI:.0f}"
    col_high = f"> {highI:.0f}"
    col_target = f"> {targetI:.0f}"

    hdr = (f"  {'Frame':<8} "
           f"{'Nr pixels':<12} "
           f"{col_low:>14} "
           f"{col_mid:>14} "
           f"{col_high:>14} "
           f"{col_target:>14}")
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    per_frame = []
    totals = {"below_low": 0, "between": 0, "above_high": 0,
              "above_target": 0, "n_valid": 0}

    for i, orig_idx in enumerate(kept_indices):
        pixels = data[i][valid_mask]
        n = len(pixels)

        below_low = int(np.sum(pixels < lowI))
        between = int(np.sum((pixels >= lowI) & (pixels <= highI)))
        above_high = int(np.sum(pixels > highI))
        above_target = int(np.sum(pixels > targetI))

        frame_data = {
            "frame_idx": orig_idx,
            "n_valid": n,
            "below_low": below_low,
            "between": between,
            "above_high": above_high,
            "above_target": above_target,
        }
        per_frame.append(frame_data)

        totals["below_low"] += below_low
        totals["between"] += between
        totals["above_high"] += above_high
        totals["above_target"] += above_target
        totals["n_valid"] += n

        print(f"  F{orig_idx:<7} "
              f"{n:<12,} "
              f"{below_low:>14,} "
              f"{between:>14,} "
              f"{above_high:>14,} "
              f"{above_target:>14,}")

    # Totals row
    print(f"  {'-' * (len(hdr) - 2)}")
    print(f"  {'ALL':<8} "
          f"{totals['n_valid']:<12,} "
          f"{totals['below_low']:>14,} "
          f"{totals['between']:>14,} "
          f"{totals['above_high']:>14,} "
          f"{totals['above_target']:>14,}")

    return {
        "kept_indices": kept_indices,
        "per_frame": per_frame,
        "totals": totals,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-frame pixel intensity statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s data/scan_att0_1p0s.h5 --lowI 1000 --highI 40000 --targetI 50000
  %(prog)s data/scan.h5 --lowI 500 --highI 30000 --targetI 50000 --darkfile dark.h5
""",
    )
    parser.add_argument("path", help="Path to HDF5 data file")
    parser.add_argument("--lowI", type=float, default=500,
                        help="Lower intensity threshold (default: 500)")
    parser.add_argument("--highI", type=float, default=10000,
                        help="Upper intensity threshold (default: 10000)")
    parser.add_argument("--targetI", type=float, default=40000,
                        help="Target intensity threshold (default: 40000)")
    parser.add_argument("--darkfile", default=None,
                        help="Path to dark HDF5 file")
    parser.add_argument("--dark_mask", type=int, default=1, choices=[0, 1],
                        help="Create mask from dark file (default: 1)")
    parser.add_argument("--maskfile", default=None,
                        help="Path to binary mask file (.tif, .npy)")
    parser.add_argument("--percentile_mask", type=float, default=100.0,
                        help="Percentile of pixels to keep (default: 100)")
    parser.add_argument("--skip_frames", default="0",
                        help="Frame indices to skip, dash-separated "
                             "(default: '0')")

    args = parser.parse_args()

    frame_stats(
        path=args.path,
        lowI=args.lowI,
        highI=args.highI,
        targetI=args.targetI,
        skip_frames_str=args.skip_frames,
        darkfile=args.darkfile,
        dark_mask=bool(args.dark_mask),
        maskfile=args.maskfile,
        percentile_mask=args.percentile_mask,
    )


if __name__ == "__main__":
    main()
