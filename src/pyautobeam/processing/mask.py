"""Pixel mask loading, application, and creation."""

import os

import numpy as np
from scipy.ndimage import uniform_filter

from pyautobeam.io.hdf5_reader import read_hdf5


def load_mask(filepath):
    """Load a binary pixel mask from a TIFF or NumPy file.

    The mask convention is: 1 = good pixel, 0 = bad pixel.

    Parameters
    ----------
    filepath : str or Path
        Path to a ``.tif`` / ``.tiff`` or ``.npy`` mask file.

    Returns
    -------
    numpy.ndarray
        2D mask array.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = os.path.splitext(str(filepath))[1].lower()

    if ext in (".tif", ".tiff"):
        import tifffile
        mask = tifffile.imread(str(filepath))
    elif ext == ".npy":
        mask = np.load(str(filepath))
    else:
        raise ValueError(
            f"Unsupported mask file extension '{ext}'. Use .tif, .tiff, or .npy"
        )

    return mask.astype(np.float32)


def apply_mask(data, mask):
    """Apply a binary pixel mask to detector data.

    Parameters
    ----------
    data : numpy.ndarray
        Detector data, 2D (Y, X) or 3D (N, Y, X).
    mask : numpy.ndarray
        2D mask (Y, X) with 1 = good pixel, 0 = bad pixel.
        Spatial dimensions must match *data*.

    Returns
    -------
    numpy.ndarray
        Masked data (float32).

    Raises
    ------
    ValueError
        If the spatial dimensions of *data* and *mask* do not match.
    """
    spatial = data.shape[-2:]
    if mask.shape != spatial:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data spatial "
            f"dimensions {spatial}"
        )

    return data.astype(np.float32) * mask.astype(np.float32)


def create_percentile_mask(data, percentile=99.99):
    """Create a mask that removes the highest-intensity pixels.

    Pixels whose intensity exceeds the given percentile threshold are
    marked as bad (0).

    Parameters
    ----------
    data : numpy.ndarray
        2D (Y, X) or 3D (N, Y, X) detector data.  If 3D, the mean
        frame (averaged along axis 0) is used to compute the threshold.
    percentile : float
        Percentile threshold (0--100). Pixels above this percentile are
        masked out.

    Returns
    -------
    numpy.ndarray
        2D binary mask (float32): 1 = below threshold (keep),
        0 = above threshold (remove).
    """
    if data.ndim == 3:
        frame = np.mean(data, axis=0, dtype=np.float32)
    else:
        frame = data.astype(np.float32)

    threshold = np.percentile(frame, percentile)
    mask = np.where(frame <= threshold, 1.0, 0.0).astype(np.float32)
    return mask


def create_dark_mask(darkfile, n_sigma=5, local_window=101, verbose=True):
    """Create a dead- and hot-pixel mask from a dark file.

    With no sample signal present, the dark frames reveal detector
    defects directly:

    - **Dead pixels**: standard deviation across dark frames is
      near zero (the pixel never fluctuates — it is unresponsive).
    - **Hot pixels**: mean dark value is significantly above the
      *local* dark level.  A local window is used to compute the
      neighborhood mean and std at each pixel, which handles
      detectors with spatial non-uniformity in the dark pedestal
      (e.g., different readout quadrants).

    Parameters
    ----------
    darkfile : str or Path
        Path to an HDF5 dark file.  Dark frames are read from
        ``exchange/data``.
    n_sigma : float
        Number of local standard deviations above the local mean
        to flag a pixel as hot.  Default 5.
    local_window : int
        Side length of the square local averaging window (must be
        odd).  Default 101 (a 101x101 region).  This should be
        large enough to average over many pixels but small enough
        to capture spatial non-uniformity in the dark pedestal.
    verbose : bool
        If True, print summary statistics.

    Returns
    -------
    mask : numpy.ndarray
        2D float32 array (Y, X).  1 = good pixel, 0 = bad pixel.
    info : dict
        Diagnostic information:

        - ``n_dead``: dead pixels flagged
        - ``n_hot``: hot pixels flagged
        - ``n_total_bad``: total bad (dead + hot, may overlap)
        - ``local_window``: window size used
        - ``mean_image``: per-pixel mean dark frame (Y, X)
        - ``std_image``: per-pixel std across dark frames (Y, X)
        - ``local_mean``: local neighborhood mean (Y, X)
        - ``local_std``: local neighborhood std (Y, X)
    """
    dark_result = read_hdf5(darkfile)
    frames = dark_result["data"].astype(np.float32)

    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError(
            f"Dark file must contain at least 2 frames. "
            f"Got shape {frames.shape}."
        )

    # Ensure odd window size
    if local_window % 2 == 0:
        local_window += 1

    mean_image = np.mean(frames, axis=0)
    std_image = np.std(frames, axis=0)

    # Dead pixels: no fluctuation across frames
    dead_std_cutoff = 0.5
    dead_mask = std_image < dead_std_cutoff

    # Hot pixels: compare each pixel against its local neighborhood.
    # Use uniform_filter to compute local mean and local mean-of-squares,
    # then derive local std = sqrt(E[X^2] - E[X]^2).
    local_mean = uniform_filter(mean_image.astype(np.float64),
                                size=local_window, mode='reflect')
    local_sq_mean = uniform_filter(
        (mean_image.astype(np.float64)) ** 2,
        size=local_window, mode='reflect',
    )
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0.0)  # numerical safety
    local_std = np.sqrt(local_var)

    # A pixel is hot if it exceeds its local mean by n_sigma * local_std
    local_threshold = local_mean + n_sigma * local_std
    hot_mask = mean_image > local_threshold

    bad_mask = dead_mask | hot_mask
    mask = np.where(bad_mask, 0.0, 1.0).astype(np.float32)

    n_dead = int(np.sum(dead_mask))
    n_hot = int(np.sum(hot_mask))
    n_total_bad = int(np.sum(bad_mask))
    n_total = mask.size

    if verbose:
        # Report global stats for context
        bulk_pixels = mean_image[mean_image < np.percentile(mean_image, 99)]
        bulk_mean = float(np.mean(bulk_pixels))
        bulk_std = float(np.std(bulk_pixels))

        print(f"Dark mask from {frames.shape[0]} frames "
              f"({frames.shape[1]}x{frames.shape[2]} px)")
        print(f"  Global dark level: {bulk_mean:.1f} (std: {bulk_std:.1f})")
        print(f"  Local window     : {local_window}x{local_window} px")
        print(f"  Hot criterion    : pixel > local_mean + {n_sigma}*local_std")
        print(f"  Dead pixels (std < {dead_std_cutoff}): "
              f"{n_dead:,} ({100.0*n_dead/n_total:.3f}%)")
        print(f"  Hot pixels       : "
              f"{n_hot:,} ({100.0*n_hot/n_total:.3f}%)")
        print(f"  Total bad        : {n_total_bad:,} / {n_total:,} "
              f"({100.0*n_total_bad/n_total:.3f}%)")
        print(f"  Good pixels      : {n_total - n_total_bad:,} "
              f"({100.0*(n_total - n_total_bad)/n_total:.3f}%)")

    info = {
        "n_dead": n_dead,
        "n_hot": n_hot,
        "n_total_bad": n_total_bad,
        "local_window": local_window,
        "mean_image": mean_image,
        "std_image": std_image,
        "local_mean": local_mean.astype(np.float32),
        "local_std": local_std.astype(np.float32),
    }

    return mask, info
