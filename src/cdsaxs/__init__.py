"""
Lightweight package initialization for cdsaxs.

Historically this module eagerly imported the entire data, plotting, reduction,
and fitting stack. That made simple fitting-only workflows fail at import time
whenever optional image-processing dependencies such as scikit-image or
scikit-learn were not installed in the active environment.

Keep plain ``import cdsaxs`` lightweight while preserving the legacy top-level
access pattern through lazy attribute loading. Fitting-only workflows no longer
pull in unrelated optional dependencies, while code that later reaches for
``cdsaxs.loaders`` or ``cdsaxs.plotting`` still resolves those modules on
demand.
"""

from importlib import import_module

from .Fitting import create_model


def _load_legacy_top_level(name):
    if name == "Fitting":
        return import_module("cdsaxs.Fitting")

    if name == "calculators":
        return import_module("cdsaxs.calculators")

    if name == "diffraction":
        return import_module("cdsaxs.diffraction")

    if name == "tools":
        return import_module("cdsaxs.tools")

    if name == "sample":
        return import_module("cdsaxs.sample")

    if name == "reduction":
        return import_module("cdsaxs.reduction")

    if name == "loaders":
        package = import_module("cdsaxs.loaders")
        import_module("cdsaxs.loaders.load_data")
        return package

    if name == "plotting":
        package = import_module("cdsaxs.plotting")
        import_module("cdsaxs.plotting.plotting")
        return package

    if name == "data":
        package = import_module("cdsaxs.data")
        import_module("cdsaxs.data.data_image")
        import_module("cdsaxs.data.data2d")
        import_module("cdsaxs.data.reduced_data1d")
        import_module("cdsaxs.data.reduced_slice")
        import_module("cdsaxs.data.dataset")
        import_module("cdsaxs.data.metadata")
        return package

    raise AttributeError(f"module 'cdsaxs' has no attribute '{name}'")


def __getattr__(name):
    value = _load_legacy_top_level(name)
    globals()[name] = value
    return value


__all__ = [
    "create_model",
    "Fitting",
    "calculators",
    "data",
    "diffraction",
    "loaders",
    "plotting",
    "reduction",
    "sample",
    "tools",
]
