"""
Compatible setup.py for CDSAXS Cython extensions.
Keeps SSE but disables higher-level vectorization that causes issues.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Ensure directory exists
os.makedirs("cdsaxs_cython", exist_ok=True)
if not os.path.exists("cdsaxs_cython/__init__.py"):
    with open("cdsaxs_cython/__init__.py", "w") as f:
        f.write('"""Cython acceleration package"""\n')

# Compatible flags - keep basic SSE but disable problematic vectorization
compile_args = [
    "-O1",                      # Minimal optimization
    "-fno-tree-vectorize",      # Disable tree vectorization  
    "-fno-fast-math",           # No fast math
    "-mno-avx",                 # No AVX (this is what causes the SVML issues)
    "-mno-avx2",                # No AVX2
    "-mno-fma",                 # No FMA
    # Keep SSE enabled as system headers need it
]

link_args = ["-lm"]

extensions = [
    Extension(
        "cdsaxs_cython.base_cython",
        ["cdsaxs_cython/base_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        libraries=["m"]
    ),
    Extension(
        "cdsaxs_cython.trapezoid_cython",
        ["cdsaxs_cython/trapezoid_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        libraries=["m"]
    ),
    Extension(
        "cdsaxs_cython.cylinder_cython",
        ["cdsaxs_cython/cylinder_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        libraries=["m"]
    )
]

setup(
    name="cdsaxs_cython",
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'language_level': 3
    }),
    zip_safe=False
)