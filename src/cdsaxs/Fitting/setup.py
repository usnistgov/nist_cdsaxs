"""
Setup script for building Cython extensions for CDSAXS optimization.
Fixed for math library compatibility issues.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import sys
import platform

# Create the cdsaxs_cython directory if it doesn't exist
os.makedirs("cdsaxs_cython", exist_ok=True)

# Create __init__.py file for the package
init_content = """# Cython acceleration package for CDSAXS
"""

with open("cdsaxs_cython/__init__.py", "w") as f:
    f.write(init_content)

# Determine platform-specific compiler flags
def get_compiler_flags():
    """Get appropriate compiler flags for the platform."""
    base_flags = ["-O3"]
    link_flags = ["-O3"]
    
    # Check if we're on Linux
    if platform.system() == "Linux":
        # More conservative flags to avoid vectorization issues
        base_flags.extend([
            "-fno-fast-math",  # Disable fast math that can cause symbol issues
            "-fno-vectorize",  # Disable automatic vectorization
            "-fno-slp-vectorize",  # Disable SLP vectorization
        ])
        
        # Add math library explicitly
        link_flags.extend(["-lm"])
        
    elif platform.system() == "Darwin":  # macOS
        base_flags.extend(["-ffast-math"])
        
    elif platform.system() == "Windows":
        # Windows-specific flags
        base_flags = ["/O2"]
        link_flags = []
    
    return base_flags, link_flags

# Get platform-appropriate flags
compile_flags, link_flags = get_compiler_flags()

print(f"Using compile flags: {compile_flags}")
print(f"Using link flags: {link_flags}")

# Define extensions with robust compilation settings
extensions = [
    Extension(
        "cdsaxs_cython.base_cython",
        ["cdsaxs_cython/base_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_WITHOUT_ASSERTIONS", "1")
        ],
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
        libraries=["m"] if platform.system() == "Linux" else []
    ),
    Extension(
        "cdsaxs_cython.trapezoid_cython",
        ["cdsaxs_cython/trapezoid_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_WITHOUT_ASSERTIONS", "1")
        ],
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
        libraries=["m"] if platform.system() == "Linux" else []
    ),
    Extension(
        "cdsaxs_cython.cylinder_cython",
        ["cdsaxs_cython/cylinder_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_WITHOUT_ASSERTIONS", "1")
        ],
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
        libraries=["m"] if platform.system() == "Linux" else []
    )
]

# Conservative compiler directives to avoid issues
compiler_directives = {
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'profile': False,
    'language_level': 3,
    'embedsignature': True,
    'initializedcheck': False,
    'overflowcheck': False
}

setup(
    name="cdsaxs_cython",
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    zip_safe=False,
    packages=['cdsaxs_cython'],
    package_dir={'cdsaxs_cython': 'cdsaxs_cython'},
    install_requires=[
        'numpy',
        'scipy',
        'cython'
    ]
)