"""
Setup script for building Cython extensions for CDSAXS optimization.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define extensions
extensions = [
    Extension(
        "cdsaxs_cython.trapezoid_cython",
        ["cdsaxs_cython/trapezoid_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"]
    ),
    Extension(
        "cdsaxs_cython.cylinder_cython",
        ["cdsaxs_cython/cylinder_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"]
    ),
    Extension(
        "cdsaxs_cython.base_cython",
        ["cdsaxs_cython/base_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"]
    )
]

setup(
    name="cdsaxs_cython",
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'profile': False,
        'language_level': 3
    }),
    zip_safe=False,
    packages=['cdsaxs_cython'],
    package_dir={'cdsaxs_cython': 'cdsaxs_cython'}
)