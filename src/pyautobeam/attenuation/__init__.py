"""Attenuation analysis module.

Provides X-ray attenuation analysis using the scattering model
with NIST Cu attenuation coefficients.

Submodules are lazily imported via PEP 562 ``__getattr__`` so that
``python -m pyautobeam.attenuation.<submodule>`` does not trigger
the runpy "found in sys.modules" warning.  Public names listed in
``__all__`` resolve on first access from their original submodule.
"""

# (name -> submodule) mapping for lazy re-exports
_LAZY_ATTRS = {
    "analyze": "pyautobeam.attenuation.analysis",
    "beer_lambert_intensity": "pyautobeam.attenuation.beer_lambert",
    "check_residuals": "pyautobeam.attenuation.beer_lambert",
    "fit_beer_lambert": "pyautobeam.attenuation.beer_lambert",
    "estimate_mu_linear": "pyautobeam.attenuation.nist_data",
    "get_cu_mass_attenuation": "pyautobeam.attenuation.nist_data",
    "frame_stats": "pyautobeam.attenuation.stats",
    "auto_attenuate_plan": "pyautobeam.attenuation.auto_attenuate",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib
        module = importlib.import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr  # cache for subsequent accesses
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
