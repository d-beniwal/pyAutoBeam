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


def frame_stats(path, low=1.0, high=40000.0,
                skip_frames=1, darkfile=None, dark_mask=True,
                maskfile=None, percentile_mask=100.0):
    """Compute per-frame pixel intensity statistics.

    Reports per-frame Min / Max / Mean and counts of pixels below
    ``low`` / above ``high``.  Stats are computed on the
    fully-preprocessed data (frame skipping, dark subtraction,
    masking, percentile mask), and percentages use the full frame
    area as the denominator.

    Parameters
    ----------
    path : str
        Path to an HDF5 data file.
    low : float
        Lower intensity threshold.  Pixels below this count toward
        the ``< low`` column.
    high : float
        Upper intensity threshold.  Pixels above this count toward
        the ``> high`` column.
    skip_frames : int
        Number of frames to skip from the start of the stack
        (default 1).  Set to 0 to use all frames.
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
        kept_indices, per_frame (list of dicts with min/max/mean/
        n_low/n_high), summary (min/max/mean of each column).
    """
    skip_frames = max(0, int(skip_frames))

    # ── Load data ──────────────────────────────────────────────────
    result = read_hdf5(path)
    data = np.array(result["data"], dtype=np.float32)

    print(f"Loaded {data.shape[0]} frames, "
          f"frame size {data.shape[1]} x {data.shape[2]} px "
          f"from {os.path.basename(path)}")

    # ── Skip frames ───────────────────────────────────────────────
    n_total = data.shape[0]
    n_skip = min(skip_frames, n_total)
    kept_indices = list(range(n_skip, n_total))
    if n_skip > 0:
        data = data[n_skip:]
        print(f"Skipped first {n_skip} frame(s) "
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
    # Convention: 0 = good pixel, 1 = bad pixel.  Combine via logical OR.
    if maskfile:
        user_mask = load_mask(maskfile)
        if pixel_mask is not None:
            pixel_mask = np.maximum(pixel_mask, user_mask)
        else:
            pixel_mask = user_mask
        n_bad = int(np.sum(user_mask > 0.5))
        print(f"User mask applied ({n_bad} bad pixels).")

    if pixel_mask is not None:
        data = apply_mask(data, pixel_mask)

    if percentile_mask < 100.0:
        pct_mask = create_percentile_mask(data, percentile=percentile_mask)
        data = apply_mask(data, pct_mask)
        n_hot = int(np.sum(pct_mask > 0.5))
        print(f"Percentile mask ({percentile_mask}%) applied "
              f"({n_hot} hot pixels masked).")

    # ── Build valid pixel mask (for counting) ─────────────────────
    # A pixel is valid (good) when its mask value is 0.
    frame_shape = data.shape[1:]
    valid_mask = np.ones(frame_shape, dtype=bool)
    if pixel_mask is not None:
        valid_mask &= (pixel_mask < 0.5)
    if percentile_mask < 100.0:
        valid_mask &= (pct_mask < 0.5)

    n_valid = int(np.sum(valid_mask))
    print(f"Valid pixels: {n_valid:,} / {valid_mask.size:,}")

    # ── Compute statistics ────────────────────────────────────────
    total_pixels = int(np.prod(data.shape[1:]))

    print(f"\n{'=' * 90}")
    print(f"PIXEL INTENSITY STATISTICS")
    print(f"  low = {low}, high = {high}")
    print(f"{'=' * 90}")

    col_low = f"< {low}"
    col_high = f"> {high}"

    hdr = (f"  {'Frame':>7s}  {'Min':>12s}  {'Max':>12s}  {'Mean':>12s}"
           f"  {col_low:>18s}  {col_high:>18s}")
    sep = f"  {'-' * (len(hdr) - 2)}"
    print(hdr)
    print(sep)

    def _fmt_count(c, total):
        return f"{int(c)}({100.0 * c / total:.2f}%)"

    per_frame = []
    all_min, all_max, all_mean, all_low, all_high = [], [], [], [], []

    for i, orig_idx in enumerate(kept_indices):
        frame = data[i]
        f_min = float(frame.min())
        f_max = float(frame.max())
        f_mean = float(frame.mean())
        n_low = int(np.sum(frame < low))
        n_high = int(np.sum(frame > high))

        per_frame.append({
            "frame_idx": orig_idx,
            "min": f_min, "max": f_max, "mean": f_mean,
            "n_low": n_low, "n_high": n_high,
        })
        all_min.append(f_min)
        all_max.append(f_max)
        all_mean.append(f_mean)
        all_low.append(n_low)
        all_high.append(n_high)

        print(f"  {orig_idx:>7d}  {f_min:>12.2f}  {f_max:>12.2f}  {f_mean:>12.2f}"
              f"  {_fmt_count(n_low, total_pixels):>18s}"
              f"  {_fmt_count(n_high, total_pixels):>18s}")

    print(sep)

    summary = {
        "min":  {"min": min(all_min),  "max": max(all_min),  "mean": float(np.mean(all_min))},
        "max":  {"min": min(all_max),  "max": max(all_max),  "mean": float(np.mean(all_max))},
        "mean": {"min": min(all_mean), "max": max(all_mean), "mean": float(np.mean(all_mean))},
        "n_low":  {"min": min(all_low),  "max": max(all_low),  "mean": float(np.mean(all_low))},
        "n_high": {"min": min(all_high), "max": max(all_high), "mean": float(np.mean(all_high))},
    }

    for label, agg in (("Min", "min"), ("Max", "max"), ("Mean", "mean")):
        m_min  = summary["min"][agg]
        m_max  = summary["max"][agg]
        m_mean = summary["mean"][agg]
        m_low  = summary["n_low"][agg]
        m_high = summary["n_high"][agg]
        print(f"  {label:>7s}  {m_min:>12.2f}  {m_max:>12.2f}  {m_mean:>12.2f}"
              f"  {_fmt_count(m_low, total_pixels):>18s}"
              f"  {_fmt_count(m_high, total_pixels):>18s}")

    return {
        "kept_indices": kept_indices,
        "per_frame": per_frame,
        "summary": summary,
        "total_pixels": total_pixels,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-frame pixel intensity statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --datapath data/scan_att0_1p0s.h5 --low 1 --high 40000
  %(prog)s --datapath data/scan.h5 --darkfile dark.h5 --skip_frames 0
""",
    )
    parser.add_argument("--datapath", required=True,
                        help="Path to HDF5 data file")
    parser.add_argument("--low", type=float, default=1.0,
                        help="Lower intensity threshold (default: 1.0)")
    parser.add_argument("--high", type=float, default=40000.0,
                        help="Upper intensity threshold (default: 40000.0)")
    parser.add_argument("--darkfile", default=None,
                        help="Path to dark HDF5 file")
    parser.add_argument("--dark_mask", type=int, default=1, choices=[0, 1],
                        help="Create mask from dark file (default: 1)")
    parser.add_argument("--maskfile", default=None,
                        help="Path to binary mask file (.tif, .npy)")
    parser.add_argument("--percentile_mask", type=float, default=100.0,
                        help="Percentile of pixels to keep (default: 100)")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Number of frames to skip from the start "
                             "(default: 1)")

    args = parser.parse_args()

    frame_stats(
        path=args.datapath,
        low=args.low,
        high=args.high,
        skip_frames=args.skip_frames,
        darkfile=args.darkfile,
        dark_mask=bool(args.dark_mask),
        maskfile=args.maskfile,
        percentile_mask=args.percentile_mask,
    )


if __name__ == "__main__":
    main()
