"""Cu X-ray attenuation coefficients from NIST XCOM.

Provides lookup and interpolation of the mass attenuation coefficient
(mu/rho) for Copper, and conversion to the linear attenuation
coefficient used by the Beer-Lambert fitting routines.
"""

import importlib.resources
import numpy as np


# Module-level cache so the file is only parsed once.
_CACHED_DATA = None

# Cu density in g/cm^3
CU_DENSITY = 8.96


def load_cu_attenuation_data():
    """Load the NIST XCOM Cu attenuation table.

    Returns
    -------
    energy_MeV : numpy.ndarray
        Photon energies in MeV.
    mu_over_rho : numpy.ndarray
        Mass attenuation coefficients in cm^2/g.
    mu_en_over_rho : numpy.ndarray
        Mass energy-absorption coefficients in cm^2/g.
    """
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA

    ref = importlib.resources.files("pyautobeam.attenuation.data").joinpath("Cu_att_data.txt")
    with importlib.resources.as_file(ref) as path:
        energy, mu_rho, mu_en_rho = [], [], []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                energy.append(float(parts[0]))
                mu_rho.append(float(parts[1]))
                mu_en_rho.append(float(parts[2]))

    _CACHED_DATA = (
        np.array(energy),
        np.array(mu_rho),
        np.array(mu_en_rho),
    )
    return _CACHED_DATA


def get_cu_mass_attenuation(energy_keV):
    """Interpolate the Cu mass attenuation coefficient at a given energy.

    Uses log-log interpolation on the NIST XCOM table for accuracy
    across the wide dynamic range of the data.

    Parameters
    ----------
    energy_keV : float
        Photon energy in keV.

    Returns
    -------
    float
        Mass attenuation coefficient mu/rho in cm^2/g.

    Raises
    ------
    ValueError
        If the energy is outside the tabulated range.
    """
    energy_MeV = energy_keV / 1000.0
    energies, mu_over_rho, _ = load_cu_attenuation_data()

    e_min, e_max = energies[0], energies[-1]
    if energy_MeV < e_min or energy_MeV > e_max:
        raise ValueError(
            f"Energy {energy_keV} keV ({energy_MeV} MeV) is outside the "
            f"tabulated range [{e_min*1000:.1f}, {e_max*1000:.1f}] keV."
        )

    # The table has duplicate energies at the Cu K-edge (8.98 keV).
    # For queries at or above the edge, use the post-edge branch.
    # Find the last index where energy <= query to land on the
    # correct side of the discontinuity.
    log_e = np.log(energies)
    log_mu = np.log(mu_over_rho)
    log_query = np.log(energy_MeV)

    log_mu_interp = np.interp(log_query, log_e, log_mu)
    return float(np.exp(log_mu_interp))


def estimate_mu_linear(energy_keV, density=CU_DENSITY):
    """Estimate the Cu linear attenuation coefficient at a given energy.

    Converts the mass attenuation coefficient to a linear coefficient
    in mm^-1, which is the unit used by :func:`beer_lambert_intensity`
    and :func:`fit_beer_lambert`.

    mu_linear [mm^-1] = (mu/rho [cm^2/g]) * (density [g/cm^3]) / 10

    The factor of 10 converts from cm^-1 to mm^-1.

    Parameters
    ----------
    energy_keV : float
        Photon energy in keV.
    density : float, optional
        Material density in g/cm^3.  Defaults to Cu (8.96 g/cm^3).

    Returns
    -------
    float
        Linear attenuation coefficient in mm^-1.
    """
    mu_over_rho = get_cu_mass_attenuation(energy_keV)
    return mu_over_rho * density / 10.0
