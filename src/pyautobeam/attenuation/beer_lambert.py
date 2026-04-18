"""Beer-Lambert law fitting and related calculations."""

import math

import numpy as np
from scipy.stats import linregress


def beer_lambert_intensity(I0, mu, thickness):
    """Compute transmitted intensity via Beer-Lambert law.

    I(x) = I0 * exp(-mu * x)

    Parameters
    ----------
    I0 : float
        Incident beam intensity (counts/s).
    mu : float
        Linear attenuation coefficient (mm^-1).
    thickness : float
        Absorber thickness (mm).

    Returns
    -------
    float
        Transmitted intensity.
    """
    return I0 * math.exp(-mu * thickness)


def fit_beer_lambert(thicknesses, intensities, acq_times):
    """Fit the Beer-Lambert law in log-space via linear regression.

    log(I / t) = log(I0) - mu * x

    Parameters
    ----------
    thicknesses : array-like
        Absorber thicknesses in mm.
    intensities : array-like
        Measured intensities (counts).
    acq_times : array-like
        Acquisition times in seconds.

    Returns
    -------
    dict
        mu : float – attenuation coefficient (mm^-1)
        I0 : float – unattenuated beam intensity (counts/s)
        r_squared : float – goodness of fit
        slope, intercept : float – raw regression parameters
        residuals : ndarray – observed minus predicted (log-space)
        std_residuals : ndarray – standardised residuals
        thicknesses : ndarray
        log_rates : ndarray
    """
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    acq_times = np.asarray(acq_times, dtype=np.float64)

    rates = intensities / acq_times
    log_rates = np.log(rates)

    # Need at least 2 distinct x-values for a meaningful slope
    if len(set(thicknesses)) < 2:
        mean_log_rate = np.mean(log_rates)
        return {
            "mu": 0.0,
            "I0": np.exp(mean_log_rate),
            "r_squared": 0.0,
            "slope": 0.0,
            "intercept": mean_log_rate,
            "residuals": log_rates - mean_log_rate,
            "std_residuals": np.zeros_like(log_rates),
            "thicknesses": thicknesses,
            "log_rates": log_rates,
        }

    slope, intercept, r_value, _, _ = linregress(thicknesses, log_rates)

    predicted = slope * thicknesses + intercept
    residuals = log_rates - predicted

    if len(residuals) > 2 and np.std(residuals) > 0:
        std_residuals = residuals / np.std(residuals)
    else:
        std_residuals = np.zeros_like(residuals)

    return {
        "mu": -slope,
        "I0": np.exp(intercept),
        "r_squared": r_value ** 2,
        "slope": slope,
        "intercept": intercept,
        "residuals": residuals,
        "std_residuals": std_residuals,
        "thicknesses": thicknesses,
        "log_rates": log_rates,
    }


def check_residuals(fit_result, threshold=2.0, r_squared_min=0.995):
    """Identify outlier data points in a Beer-Lambert fit.

    When R^2 >= *r_squared_min*, outliers are points whose absolute
    standardised residual exceeds *threshold*.  When R^2 is lower, a
    leave-one-out analysis finds the single point whose removal most
    improves R^2.

    Parameters
    ----------
    fit_result : dict
        Output of :func:`fit_beer_lambert`.
    threshold : float
        Standardised-residual cutoff (default 2.0).
    r_squared_min : float
        R^2 below which leave-one-out detection is used.

    Returns
    -------
    list of int
        Indices of outlier points (empty if the fit is clean).
    """
    n = len(fit_result["std_residuals"])
    if n <= 2:
        return []

    if fit_result["r_squared"] >= r_squared_min:
        return [i for i, sr in enumerate(fit_result["std_residuals"])
                if abs(sr) > threshold]

    # Leave-one-out when R^2 is poor
    thicknesses = fit_result["thicknesses"]
    log_rates = fit_result["log_rates"]

    best_improvement = 0.0
    worst_index = -1

    for i in range(n):
        t_loo = np.delete(thicknesses, i)
        lr_loo = np.delete(log_rates, i)
        if len(t_loo) < 2 or len(set(t_loo)) < 2:
            continue
        _, _, r_val_loo, _, _ = linregress(t_loo, lr_loo)
        improvement = r_val_loo ** 2 - fit_result["r_squared"]
        if improvement > best_improvement:
            best_improvement = improvement
            worst_index = i

    if worst_index >= 0 and best_improvement > 0.001:
        return [worst_index]

    return []
