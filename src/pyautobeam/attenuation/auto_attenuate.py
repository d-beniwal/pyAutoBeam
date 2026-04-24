"""Automatic attenuation tuning -- Bluesky plan.

A bluesky plan that iteratively acquires data at different attenuator
positions and acquisition times to find the optimal combination that
places the max pixel intensity within a target window.

Physical model::

    I_measured = S * I0 * exp(-mu * thickness) * acq_time

The plan:
1. Takes an initial quick exposure
2. Estimates S*I0 from the measurement using NIST mu
3. Predicts the optimal (att_pos, acq_time) for the target intensity
4. Acquires at the predicted settings
5. Repeats until the measured intensity is within the target window

IMPORTANT: This module defines bluesky plans (generators).  It does NOT
connect to or control any hardware when imported.  Hardware interaction
only occurs when a plan is executed via ``RE(plan())``.

Usage::

    from pyautobeam.attenuation.auto_attenuate import auto_attenuate_plan

    # In a bluesky session:
    RE(auto_attenuate_plan(
        det=my_detector,
        attenuator=attenB,
        shutter=fs,
        sample_name="Ceria",
        energy_keV=63,
        target_intensity=45000,
        darkfile="/path/to/dark.h5",
    ))
"""

import math
import os
import time

import numpy as np

from pyautobeam.attenuation.nist_data import estimate_mu_linear

# Attenuator position -> Cu thickness (mm)
_POS_THICKNESS = {
    0: 0.00, 1: 0.50, 2: 1.00, 3: 1.50, 4: 2.00, 5: 2.39,
    6: 4.78, 8: 7.14, 9: 9.53, 10: 11.91, 11: 14.30, 12: 16.66,
}
ALL_ATTENUATOR_POSITIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]


def att_thickness_from_pos(pos):
    """Map attenuator position to Cu thickness in mm."""
    return _POS_THICKNESS.get(pos, None)


def _predict_acq_time(SI0, mu, att_pos, target_intensity):
    """Predict acquisition time to reach target intensity.

    Returns
    -------
    float or None
        Predicted time in seconds, or None if att_pos is unknown.
    """
    thickness = att_thickness_from_pos(att_pos)
    if thickness is None or SI0 <= 0:
        return None
    pred_rate = SI0 * math.exp(-mu * thickness)
    if pred_rate <= 0:
        return None
    return target_intensity / pred_rate


def _find_best_settings(SI0, mu, target_intensity,
                        min_acq_time=0.01, max_acq_time=100.0):
    """Find the (att_pos, acq_time) that best achieves target intensity.

    Prefers acquisition times closest to 1 second (log-distance).

    Returns
    -------
    tuple (att_pos, acq_time) or (None, None)
    """
    best_pos = None
    best_time = None
    best_score = float("inf")

    for pos in ALL_ATTENUATOR_POSITIONS:
        t = _predict_acq_time(SI0, mu, pos, target_intensity)
        if t is None:
            continue
        if t < min_acq_time or t > max_acq_time:
            continue
        # Prefer times near 1s
        score = abs(math.log10(t))
        if score < best_score:
            best_score = score
            best_pos = pos
            best_time = t

    return best_pos, best_time


def _read_max_intensity(filepath, darkfile=None, skip_frames=1):
    """Read an HDF5 file and return max intensity after preprocessing.

    This is called AFTER acquisition is complete and the file is on
    disk.  It does NOT interact with any hardware.

    Parameters
    ----------
    skip_frames : int
        Number of frames to skip from the start of the stack.
    """
    # Lazy imports to avoid pulling in h5py at plan definition time
    from pyautobeam.io.hdf5_reader import read_hdf5
    from pyautobeam.processing.dark import subtract_dark

    result = read_hdf5(filepath)
    data = np.array(result["data"], dtype=np.float32)

    n_skip = max(0, int(skip_frames))
    if n_skip >= data.shape[0]:
        return 0.0
    if n_skip > 0:
        data = data[n_skip:]
    if data.shape[0] == 0:
        return 0.0

    # Dark subtraction
    if darkfile:
        dark_result = read_hdf5(darkfile)
        dark_frames = dark_result["data"]
        if dark_frames is not None:
            dark_mean = np.mean(dark_frames.astype(np.float32), axis=0)
            data = subtract_dark(data, dark_mean)
            np.clip(data, 0, None, out=data)

    return float(np.max(data))


# ── Bluesky plan ─────────────────────────────────────────────────────

def auto_attenuate_plan(
    det,
    attenuator,
    shutter,
    shutter_open_cmd,
    shutter_close_cmd,
    sample_name,
    energy_keV,
    target_intensity=45000,
    target_window=(0.7, 0.9),
    initial_att_pos=3,
    initial_acq_time=1.0,
    nframes=5,
    darkfile=None,
    data_dir=None,
    min_acq_time=0.01,
    max_acq_time=100.0,
    max_iterations=10,
    skip_frames=1,
    verbose=True,
):
    """Bluesky plan: automatically find optimal attenuation settings.

    This is a generator function.  It only interacts with hardware
    when executed by the RunEngine: ``RE(auto_attenuate_plan(...))``.

    Parameters
    ----------
    det : ophyd Device
        Detector object with ``cam.acquire_time``, ``cam.num_images``,
        ``cam.acquire``, and ``hdf1`` plugin.
    attenuator : ophyd Device
        Attenuator device with ``rz`` motor component.
    shutter : ophyd Device
        Fast shutter device.
    shutter_open_cmd : callable
        Bluesky plan stub to open the shutter.
        Called as ``yield from shutter_open_cmd()``.
    shutter_close_cmd : callable
        Bluesky plan stub to close the shutter.
        Called as ``yield from shutter_close_cmd()``.
    sample_name : str
        Sample name for file naming.
    energy_keV : float
        X-ray energy in keV.
    target_intensity : float
        Desired max pixel intensity (counts).  Default 45000.
    target_window : tuple of float
        (low_fraction, high_fraction) of target_intensity that defines
        the acceptable window.  Default (0.7, 0.9) meaning 70-90%.
    initial_att_pos : int
        Starting attenuator position.  Default 3 (conservative).
    initial_acq_time : float
        Starting acquisition time in seconds.  Default 1.0.
    nframes : int
        Number of frames per acquisition.  Default 5.
    darkfile : str or None
        Path to dark HDF5 file for preprocessing.
    data_dir : str or None
        Directory to save calibration files.  If None, uses current
        working directory.
    min_acq_time : float
        Minimum allowed acquisition time.  Default 0.01 s.
    max_acq_time : float
        Maximum allowed acquisition time.  Default 100.0 s.
    max_iterations : int
        Maximum number of acquisition attempts.  Default 10.
    skip_frames : int
        Number of frames to skip from the start of each stack
        (default 1).  Set to 0 to use all frames.
    verbose : bool
        Print progress to terminal.

    Yields
    ------
    bluesky Msg objects (via bps.mv, bps.trigger, etc.)

    Returns
    -------
    dict (via return, accessible from RunEngine)
        att_pos, acq_time, max_intensity, SI0, mu, iterations,
        history (list of per-iteration results)
    """
    # Lazy import so this module can be imported without bluesky
    import bluesky.plan_stubs as bps

    skip_frames = max(0, int(skip_frames))
    if data_dir is None:
        data_dir = os.getcwd()

    mu = estimate_mu_linear(energy_keV)
    low_bound = target_window[0] * target_intensity
    high_bound = target_window[1] * target_intensity

    if verbose:
        print("=" * 60)
        print("AUTO-ATTENUATE PLAN")
        print("=" * 60)
        print(f"Detector       : {det.name}")
        print(f"Attenuator     : {attenuator.name}")
        print(f"Sample         : {sample_name}")
        print(f"Energy         : {energy_keV} keV")
        print(f"mu (NIST)      : {mu:.4f} /mm")
        print(f"Target         : {target_intensity} counts")
        print(f"Window         : {low_bound:.0f} - {high_bound:.0f} counts "
              f"({target_window[0]*100:.0f}-{target_window[1]*100:.0f}%)")
        print(f"Initial        : att{initial_att_pos}, {initial_acq_time}s")
        print(f"Frames/acq     : {nframes}")
        print(f"Dark file      : {darkfile or 'None'}")
        print(f"Data dir       : {data_dir}")
        print("=" * 60)

    current_att_pos = initial_att_pos
    current_acq_time = initial_acq_time
    SI0 = None
    history = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        if verbose:
            thickness = att_thickness_from_pos(current_att_pos)
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")
            print(f"Attenuator: pos {current_att_pos} "
                  f"({thickness:.2f} mm Cu)")
            print(f"Acq time  : {current_acq_time:.3f} s")

        # ── Build filename ─────────────────────────────────────────
        file_name = (f"{sample_name}_{energy_keV:.0f}keV_"
                     f"att{current_att_pos}_{current_acq_time:.3f}s_"
                     f"cal{iteration:02d}")
        filepath = os.path.join(data_dir, file_name + ".h5")

        # ── Move attenuator ────────────────────────────────────────
        yield from bps.mv(attenuator.rz, current_att_pos)
        if verbose:
            print(f"Attenuator moved to position {current_att_pos}.")

        # ── Configure detector ─────────────────────────────────────
        yield from bps.mv(
            det.cam.acquire_time, current_acq_time,
            det.cam.num_images, nframes,
        )

        # Set file name on HDF plugin
        yield from bps.mv(det.hdf1.file_name, file_name)
        yield from bps.mv(det.hdf1.file_path, data_dir)

        # ── Open shutter ───────────────────────────────────────────
        yield from shutter_open_cmd()

        # ── Acquire ────────────────────────────────────────────────
        if verbose:
            print(f"Acquiring {nframes} frames...")
        yield from bps.trigger(det.cam.acquire, wait=True)

        # ── Close shutter ──────────────────────────────────────────
        yield from shutter_close_cmd()

        # ── Wait for file to be written ────────────────────────────
        yield from bps.sleep(1.0)

        # ── Analyze the acquired data ──────────────────────────────
        # Find the actual file path (may have auto-increment number)
        actual_path = filepath
        if not os.path.exists(actual_path):
            # Try common HDF5 naming patterns
            for candidate in [
                filepath,
                os.path.join(data_dir, file_name + "_000000.h5"),
                os.path.join(data_dir, file_name + "_000001.h5"),
            ]:
                if os.path.exists(candidate):
                    actual_path = candidate
                    break

        if not os.path.exists(actual_path):
            if verbose:
                print(f"WARNING: Cannot find output file. "
                      f"Expected near {filepath}")
                print("Skipping analysis for this iteration.")
            history.append({
                "iteration": iteration,
                "att_pos": current_att_pos,
                "acq_time": current_acq_time,
                "max_intensity": None,
                "filepath": filepath,
                "status": "FILE_NOT_FOUND",
            })
            continue

        max_intensity = _read_max_intensity(
            actual_path, darkfile=darkfile, skip_frames=skip_frames,
        )

        if verbose:
            print(f"Max intensity : {max_intensity:.1f} counts")

        # ── Update S*I0 estimate ───────────────────────────────────
        thickness = att_thickness_from_pos(current_att_pos)
        if max_intensity > 0 and thickness is not None:
            transmission = math.exp(-mu * thickness)
            SI0 = max_intensity / (current_acq_time * transmission)
            if verbose:
                print(f"S*I0 estimate : {SI0:.2f} cts/s")

        # ── Record history ─────────────────────────────────────────
        status = "OK"
        if max_intensity <= 0:
            status = "ZERO"
        elif max_intensity < low_bound:
            status = "TOO LOW"
        elif max_intensity > high_bound:
            status = "TOO HIGH"
        else:
            status = "IN WINDOW"
            converged = True

        history.append({
            "iteration": iteration,
            "att_pos": current_att_pos,
            "acq_time": current_acq_time,
            "thickness": thickness,
            "max_intensity": max_intensity,
            "SI0": SI0,
            "filepath": actual_path,
            "status": status,
        })

        if verbose:
            print(f"Status        : {status}")

        # ── Check convergence ──────────────────────────────────────
        if converged:
            if verbose:
                print(f"\nCONVERGED after {iteration} iteration(s).")
                print(f"  Attenuator  : pos {current_att_pos} "
                      f"({thickness:.2f} mm Cu)")
                print(f"  Acq time    : {current_acq_time:.3f} s")
                print(f"  Max intensity: {max_intensity:.1f} counts "
                      f"(target window: {low_bound:.0f}-{high_bound:.0f})")
            break

        # ── Predict next settings ──────────────────────────────────
        if SI0 is None or SI0 <= 0:
            # No good estimate yet — try less attenuation
            if current_att_pos > 0:
                idx = ALL_ATTENUATOR_POSITIONS.index(current_att_pos)
                if idx > 0:
                    current_att_pos = ALL_ATTENUATOR_POSITIONS[idx - 1]
            else:
                current_acq_time = min(current_acq_time * 2, max_acq_time)
            continue

        # Target the center of the window
        target_center = (low_bound + high_bound) / 2.0
        best_pos, best_time = _find_best_settings(
            SI0, mu, target_center,
            min_acq_time=min_acq_time,
            max_acq_time=max_acq_time,
        )

        if best_pos is not None and best_time is not None:
            current_att_pos = best_pos
            current_acq_time = round(best_time, 3)
            if verbose:
                print(f"Next prediction: att{best_pos}, "
                      f"{current_acq_time:.3f}s")
        else:
            if verbose:
                print("WARNING: No valid prediction found. "
                      "Halving acq time.")
            current_acq_time = max(current_acq_time / 2, min_acq_time)

    # ── Final summary ──────────────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print("AUTO-ATTENUATE SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Iter':<6} {'Att':>4} {'Thick(mm)':>10} "
              f"{'Acq(s)':>8} {'Max Int':>10} {'S*I0':>12} {'Status':>12}")
        print("-" * 66)
        for h in history:
            si0_str = (f"{h.get('SI0', 0):.1f}"
                       if h.get('SI0') else "--")
            mi_str = (f"{h['max_intensity']:.1f}"
                      if h['max_intensity'] is not None else "--")
            thick = h.get('thickness', 0) or 0
            print(f"{h['iteration']:<6} {h['att_pos']:>4} "
                  f"{thick:>10.2f} "
                  f"{h['acq_time']:>8.3f} {mi_str:>10} "
                  f"{si0_str:>12} {h['status']:>12}")

        if converged:
            final = history[-1]
            print(f"\nRECOMMENDED SETTINGS:")
            print(f"  Attenuator position : {final['att_pos']}")
            print(f"  Acquisition time    : {final['acq_time']:.3f} s")
            print(f"  Expected intensity  : {final['max_intensity']:.0f} counts")
        else:
            print(f"\nDid not converge within {max_iterations} iterations.")
            if history:
                last = history[-1]
                print(f"  Last attempt: att{last['att_pos']}, "
                      f"{last['acq_time']:.3f}s -> "
                      f"{last.get('max_intensity', '--')} counts")

    result = {
        "converged": converged,
        "att_pos": history[-1]["att_pos"] if history else None,
        "acq_time": history[-1]["acq_time"] if history else None,
        "max_intensity": (history[-1]["max_intensity"]
                          if history else None),
        "SI0": SI0,
        "mu": mu,
        "iterations": len(history),
        "history": history,
    }

    return result
