"""HDF5 data reader for beamline detector data."""

import h5py
import numpy as np

DEFAULT_DARK_KEYS = ["exchange/dark", "exchange/data_dark", "exchange/dark_data"]


def read_hdf5(filepath, data_key="exchange/data", dark_keys=None,
              bright_key="exchange/bright"):
    """Read detector frames, dark, bright, and metadata from an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.
    data_key : str
        HDF5 dataset path for the image data.
    dark_keys : list of str or None
        HDF5 dataset paths to try for dark frames (first match wins).
        Defaults to ["exchange/dark", "exchange/data_dark", "exchange/dark_data"].
    bright_key : str
        HDF5 dataset path for bright/flat-field frames.

    Returns
    -------
    dict
        {"data": ndarray, "dark": ndarray|None, "bright": ndarray|None,
         "metadata": dict}
    """
    if dark_keys is None:
        dark_keys = DEFAULT_DARK_KEYS

    with h5py.File(filepath, "r") as f:
        if data_key not in f:
            raise KeyError(f"Data key '{data_key}' not found in {filepath}")
        data = f[data_key][:]

        dark = None
        for dk in dark_keys:
            if dk in f:
                dark = f[dk][:]
                break

        bright = None
        if bright_key in f:
            bright = f[bright_key][:]

        metadata = _extract_metadata(f)

    return {"data": data, "dark": dark, "bright": bright, "metadata": metadata}


def read_hdf5_dark(filepath, dark_keys=None):
    """Read only the dark frames from an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file containing dark data.
    dark_keys : list of str or None
        HDF5 dataset paths to try (first match wins).

    Returns
    -------
    numpy.ndarray or None
        Dark frame array, or None if no dark data found.
    """
    if dark_keys is None:
        dark_keys = DEFAULT_DARK_KEYS

    with h5py.File(filepath, "r") as f:
        for dk in dark_keys:
            if dk in f:
                return f[dk][:]
    return None


def list_hdf5_contents(filepath):
    """List all datasets in an HDF5 file with their shapes and dtypes.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        Mapping of dataset paths to {"shape": tuple, "dtype": str}.
    """
    contents = {}

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            contents[name] = {"shape": obj.shape, "dtype": str(obj.dtype)}

    with h5py.File(filepath, "r") as f:
        f.visititems(_visitor)

    return contents


def _extract_metadata(f):
    """Extract metadata from WM/ and measurement/ groups.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.

    Returns
    -------
    dict
        Metadata key-value pairs. Scalar datasets are unwrapped to Python
        types; byte-strings are decoded to str.
    """
    metadata = {}

    # WM group -- per-frame metadata; take first value as representative
    if "WM" in f:
        for key in f["WM"]:
            ds = f[f"WM/{key}"]
            if ds.ndim == 0:
                val = ds[()]
            else:
                val = ds[0]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            elif isinstance(val, np.generic):
                val = val.item()
            metadata[key] = val

    # measurement/process -- scan dates and parameters
    if "measurement/process" in f:
        proc = f["measurement/process"]
        for key in ["start_date", "end_date"]:
            if key in proc:
                val = proc[key][0]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                metadata[f"process_{key}"] = val

        if "scan_parameters" in proc:
            sp = proc["scan_parameters"]
            for key in sp:
                val = sp[key][0]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                elif isinstance(val, np.generic):
                    val = val.item()
                metadata[f"scan_{key}"] = val

    return metadata
