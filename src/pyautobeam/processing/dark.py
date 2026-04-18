"""Dark frame subtraction for detector data."""

import numpy as np


def subtract_dark(data, dark):
    """Subtract dark current from detector frames.

    Parameters
    ----------
    data : numpy.ndarray
        Detector data, 2D (Y, X) or 3D (N, Y, X).
    dark : numpy.ndarray
        Dark frame(s), 2D (Y, X) or 3D (M, Y, X).
        If 3D, averaged along axis 0 to produce a single mean dark frame.

    Returns
    -------
    numpy.ndarray
        Dark-subtracted data in float32, same dimensionality as input.
    """
    if dark.ndim == 3:
        dark_mean = np.mean(dark, axis=0, dtype=np.float32)
    else:
        dark_mean = dark.astype(np.float32)

    return data.astype(np.float32) - dark_mean