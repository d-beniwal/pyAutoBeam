"""Attenuation analysis module.

Provides X-ray attenuation analysis using the scattering model
with NIST Cu attenuation coefficients.
"""

from pyautobeam.attenuation.analysis import analyze
from pyautobeam.attenuation.beer_lambert import (
    beer_lambert_intensity,
    check_residuals,
    fit_beer_lambert,
)
from pyautobeam.attenuation.nist_data import (
    estimate_mu_linear,
    get_cu_mass_attenuation,
)
from pyautobeam.attenuation.stats import frame_stats
from pyautobeam.attenuation.auto_attenuate import auto_attenuate_plan
