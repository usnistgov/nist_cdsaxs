# cdsaxs_cython/__init__.py
"""
Cython acceleration package for CDSAXS.

This package provides accelerated versions of computationally intensive CDSAXS functions
using Cython with automatic fallback to pure Python implementations.
"""

__version__ = "1.0.0"

# Try to import Cython modules
try:
    from . import base_cython
    from . import trapezoid_cython  
    from . import cylinder_cython
    _cython_available = True
except ImportError:
    _cython_available = False

__all__ = [
    'base_cython',
    'trapezoid_cython', 
    'cylinder_cython',
    '_cython_available'
]

if _cython_available:
    print("CDSAXS Cython acceleration loaded successfully")
else:
    print("CDSAXS Cython acceleration not available - using pure Python fallback")