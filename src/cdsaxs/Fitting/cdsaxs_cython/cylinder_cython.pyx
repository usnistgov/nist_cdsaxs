# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange
from libc.math cimport exp, log, sqrt, pow, isfinite, fabs, sin, cos, atan2
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset

# Declare numpy array types
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t CTYPE_t

# Fast sin/cos approximation for small angles (optional - use if high precision not critical)
cdef inline double fast_sin(double x) nogil:
    """Fast sine approximation using Taylor series (good for small x)"""
    if fabs(x) < 0.1:
        return x * (1.0 - x*x/6.0 + x*x*x*x/120.0)
    else:
        return sin(x)

cdef inline double fast_cos(double x) nogil:
    """Fast cosine approximation using Taylor series (good for small x)"""
    if fabs(x) < 0.1:
        return 1.0 - x*x/2.0 + x*x*x*x/24.0
    else:
        return cos(x)

# Optimized Bessel function with better numerical stability
cdef double bessel_j1_optimized(double x) nogil:
    """Highly optimized Bessel function J1(x) with better numerical stability."""
    cdef double ax, y, ans1, ans2, z, xx
    
    if x == 0.0:
        return 0.0
    
    ax = fabs(x)
    
    if ax < 8.0:
        # Use optimized series expansion for small x
        y = x * x
        ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 
                + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
        ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 
                + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
        return ans1 / ans2
    else:
        # Use asymptotic expansion for large x with better constants
        z = 8.0 / ax
        y = z * z
        xx = ax - 2.356194491
        ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 
                + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        ans2 = 0.04687499995 + y * (-0.2002690873e-3 
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 
                + y * 0.105787412e-6)))
        
        # Handle sign for negative x
        if x < 0:
            return -sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2)
        else:
            return sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2)

# Pre-computed lookup table for common values (optional optimization)
cdef class BesselCache:
    """Cache for Bessel function values to avoid recomputation"""
    cdef double[:] cache_x
    cdef double[:] cache_j1
    cdef int cache_size
    cdef double x_min, x_max, dx
    
    def __init__(self, int size=1000, double x_max=50.0):
        self.cache_size = size
        self.x_min = 0.0
        self.x_max = x_max
        self.dx = x_max / size
        self.cache_x = np.linspace(0, x_max, size)
        self.cache_j1 = np.zeros(size)
        
        # Pre-compute values
        cdef int i
        for i in range(size):
            self.cache_j1[i] = bessel_j1_optimized(self.cache_x[i])
    
    cdef double lookup(self, double x) nogil:
        """Fast lookup with linear interpolation"""
        if x < self.x_min or x > self.x_max:
            return bessel_j1_optimized(x)
        
        cdef int idx = <int>(x / self.dx)
        if idx >= self.cache_size - 1:
            return self.cache_j1[self.cache_size - 1]
        
        # Linear interpolation
        cdef double alpha = (x - idx * self.dx) / self.dx
        return self.cache_j1[idx] * (1.0 - alpha) + self.cache_j1[idx + 1] * alpha

@cython.boundscheck(False)
@cython.wraparound(False)
def cone_fourier_transform_cython_ultra(cnp.ndarray[DTYPE_t, ndim=2] par,
                                       int layers,
                                       cnp.ndarray[DTYPE_t, ndim=2] qr,
                                       cnp.ndarray[DTYPE_t, ndim=2] qz,
                                       cnp.ndarray[cnp.int32_t, ndim=1] discretization,
                                       cnp.ndarray[DTYPE_t, ndim=1] sld_values):
    """
    Ultra-optimized Fourier transform with memory pooling and vectorization.
    """
    # Declare all variables at the top
    cdef int rows = qr.shape[0]
    cdef int cols = qr.shape[1]
    cdef cnp.ndarray[CTYPE_t, ndim=2] form = np.zeros((rows, cols), dtype=np.complex128)
    
    cdef DTYPE_t h1, h2, stepsize, r1, r2, slope
    cdef DTYPE_t ri1, ri2, qr_val, qz_val, sld_val
    cdef DTYPE_t bessel1, bessel2, phase1, phase2
    cdef DTYPE_t cos_val1, sin_val1, cos_val2, sin_val2
    cdef DTYPE_t real_contrib, imag_contrib
    cdef DTYPE_t two_pi_over_qr, stepsize_half_sld
    cdef int i, ii, row, col, n_steps
    cdef DTYPE_t *z_array = NULL
    cdef DTYPE_t *ri_array = NULL
    cdef DTYPE_t *bessel_array = NULL
    cdef DTYPE_t *cos_array = NULL
    cdef DTYPE_t *sin_array = NULL
    
    # Pre-calculate constants
    cdef DTYPE_t pi_factor = 2.0 * 3.14159265358979323846
    cdef DTYPE_t eps = 1e-12
    
    # Find maximum discretization to pre-allocate arrays
    cdef int max_discretization = 0
    for i in range(layers):
        if discretization[i] > max_discretization:
            max_discretization = discretization[i]
    
    # Pre-allocate working arrays (reuse across layers)
    z_array = <DTYPE_t *>malloc((max_discretization + 1) * sizeof(DTYPE_t))
    ri_array = <DTYPE_t *>malloc((max_discretization + 1) * sizeof(DTYPE_t))
    bessel_array = <DTYPE_t *>malloc((max_discretization + 1) * sizeof(DTYPE_t))
    cos_array = <DTYPE_t *>malloc((max_discretization + 1) * sizeof(DTYPE_t))
    sin_array = <DTYPE_t *>malloc((max_discretization + 1) * sizeof(DTYPE_t))
    
    if (z_array == NULL or ri_array == NULL or bessel_array == NULL or 
        cos_array == NULL or sin_array == NULL):
        # Cleanup any allocated memory
        if z_array != NULL: free(z_array)
        if ri_array != NULL: free(ri_array)
        if bessel_array != NULL: free(bessel_array)
        if cos_array != NULL: free(cos_array)
        if sin_array != NULL: free(sin_array)
        raise MemoryError("Could not allocate memory for working arrays")
    
    try:
        h1 = 0.0
        h2 = 0.0
        
        # Process each layer
        for i in range(layers):
            h2 = h2 + par[i, 1]
            stepsize = par[i, 1] / discretization[i]
            sld_val = sld_values[i] if i < len(sld_values) else 1.0
            stepsize_half_sld = stepsize * 0.5 * sld_val
            
            if i > 0:
                h1 = h1 + par[i-1, 1]
            
            r1 = par[i, 0]
            r2 = par[i+1, 0]
            
            # Avoid division by zero
            if fabs(r1 - r2) < eps:
                r1 = r1 + eps
            
            slope = (h2 - h1) / (r2 - r1)
            n_steps = discretization[i] + 1
            
            # Pre-compute z and radius values for this layer
            for ii in range(n_steps):
                z_array[ii] = h1 + ii * stepsize
                ri_array[ii] = (z_array[ii] - h1) / slope + r1
            
            # Main calculation loop with better memory access patterns
            for row in range(rows):
                for col in range(cols):
                    qr_val = qr[row, col]
                    qz_val = qz[row, col]
                    
                    if qr_val > eps:
                        two_pi_over_qr = pi_factor / qr_val
                        
                        # Pre-compute Bessel functions and phases for this (qr, qz) pair
                        for ii in range(n_steps):
                            bessel_array[ii] = bessel_j1_optimized(qr_val * ri_array[ii])
                            phase1 = qz_val * z_array[ii]
                            cos_array[ii] = cos(phase1)
                            sin_array[ii] = sin(phase1)
                        
                        # Vectorized accumulation
                        real_contrib = 0.0
                        imag_contrib = 0.0
                        
                        for ii in range(n_steps - 1):
                            # Trapezoidal rule with pre-computed values
                            real_contrib += (ri_array[ii] * bessel_array[ii] * cos_array[ii] + 
                                           ri_array[ii+1] * bessel_array[ii+1] * cos_array[ii+1])
                            imag_contrib += (ri_array[ii] * bessel_array[ii] * sin_array[ii] + 
                                           ri_array[ii+1] * bessel_array[ii+1] * sin_array[ii+1])
                        
                        # Apply scaling factors
                        real_contrib *= two_pi_over_qr * stepsize_half_sld
                        imag_contrib *= two_pi_over_qr * stepsize_half_sld
                        
                        # Add to form factor
                        form[row, col].real = form[row, col].real + real_contrib
                        form[row, col].imag = form[row, col].imag + imag_contrib
    
    finally:
        # Clean up memory
        if z_array != NULL: free(z_array)
        if ri_array != NULL: free(ri_array)
        if bessel_array != NULL: free(bessel_array)
        if cos_array != NULL: free(cos_array)
        if sin_array != NULL: free(sin_array)
    
    return form

@cython.boundscheck(False)
@cython.wraparound(False)
def convert_cartesian_cylindrical_cython_ultra(cnp.ndarray[DTYPE_t, ndim=2] qx,
                                              cnp.ndarray[DTYPE_t, ndim=2] qy):
    """
    Ultra-optimized coordinate conversion with SIMD-friendly operations.
    """
    cdef int rows = qx.shape[0]
    cdef int cols = qx.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] qr = np.empty((rows, cols), dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=2] alpha = np.empty((rows, cols), dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t qx_val, qy_val, qr_val, qx_sq, qy_sq
    cdef DTYPE_t eps = 1e-12
    
    # Process in blocks for better cache performance
    cdef int block_size = 32
    cdef int row_blocks = (rows + block_size - 1) // block_size
    cdef int col_blocks = (cols + block_size - 1) // block_size
    cdef int row_start, row_end, col_start, col_end
    
    for i in range(row_blocks):
        row_start = i * block_size
        row_end = min((i + 1) * block_size, rows)
        
        for j in range(col_blocks):
            col_start = j * block_size
            col_end = min((j + 1) * block_size, cols)
            
            # Process block
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    qx_val = qx[row, col]
                    qy_val = qy[row, col]
                    
                    # Optimized magnitude calculation
                    qx_sq = qx_val * qx_val
                    qy_sq = qy_val * qy_val
                    qr_val = sqrt(qx_sq + qy_sq)
                    qr[row, col] = qr_val
                    
                    # Optimized angle calculation
                    if qr_val > eps:
                        alpha[row, col] = atan2(qy_val, qx_val)
                    else:
                        alpha[row, col] = 0.0
    
    return qr, alpha

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_cyl_sm_cython_ultra(cnp.ndarray[DTYPE_t, ndim=2] par,
                           int layers,
                           cnp.ndarray[DTYPE_t, ndim=2] qr,
                           cnp.ndarray[DTYPE_t, ndim=2] qz,
                           DTYPE_t dw,
                           DTYPE_t i0,
                           DTYPE_t background,
                           cnp.ndarray[cnp.int32_t, ndim=1] discretization,
                           cnp.ndarray[DTYPE_t, ndim=1] sld_values):
    """
    Ultra-optimized cylinder simulation with fused operations.
    """
    # Get form factor
    cdef cnp.ndarray[CTYPE_t, ndim=2] form = cone_fourier_transform_cython_ultra(
        par, layers, qr, qz, discretization, sld_values)
    
    cdef int rows = qr.shape[0]
    cdef int cols = qr.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] sim_int = np.empty((rows, cols), dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t q_squared, dw_squared = dw * dw
    cdef DTYPE_t m_val, real_part, imag_part, magnitude_squared
    cdef CTYPE_t form_val
    
    # Fused loop: combine Debye-Waller factor, form factor, and intensity calculation
    for i in range(rows):
        for j in range(cols):
            # Calculate Debye-Waller factor
            q_squared = qr[i, j] * qr[i, j] + qz[i, j] * qz[i, j]
            m_val = sqrt(exp(-q_squared * dw_squared))
            
            # Apply Debye-Waller factor to form factor and calculate intensity in one step
            form_val = form[i, j] * m_val
            real_part = form_val.real
            imag_part = form_val.imag
            magnitude_squared = real_part * real_part + imag_part * imag_part
            
            # Final intensity calculation
            sim_int[i, j] = magnitude_squared * i0 + background
    
    return sim_int

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_cyl_gf_cython_ultra(cnp.ndarray[DTYPE_t, ndim=1] sim_par,
                           int layers,
                           cnp.ndarray[DTYPE_t, ndim=2] intensity,
                           cnp.ndarray[DTYPE_t, ndim=2] qr,
                           cnp.ndarray[DTYPE_t, ndim=2] qz,
                           cnp.ndarray[cnp.int32_t, ndim=1] discretization,
                           cnp.ndarray[DTYPE_t, ndim=1] sld_values):
    """
    Ultra-optimized GF calculation with early termination and numerical stability.
    """
    # Pre-allocate PAR array
    cdef cnp.ndarray[DTYPE_t, ndim=2] pars = np.empty((layers + 1, 2), dtype=np.float64)
    cdef DTYPE_t i0, dw, background_val
    cdef int i, j
    cdef int rows = intensity.shape[0]
    cdef int cols = intensity.shape[1]
    cdef DTYPE_t gf = 0.0
    cdef DTYPE_t log_diff, sim_val, exp_val
    cdef DTYPE_t eps = 1e-15
    cdef DTYPE_t max_gf = 1e10  # Early termination threshold
    
    # Extract parameters efficiently
    cdef int param_idx = 0
    for i in range(layers + 1):
        for j in range(2):
            pars[i, j] = sim_par[param_idx]
            param_idx += 1
    
    i0 = sim_par[param_idx]
    dw = sim_par[param_idx + 1]
    background_val = sim_par[param_idx + 2]
    
    # Calculate simulated intensity
    cdef cnp.ndarray[DTYPE_t, ndim=2] sim_int = sim_cyl_sm_cython_ultra(
        pars, layers, qr, qz, dw, i0, background_val, discretization, sld_values)
    
    # Optimized GF calculation with early termination
    for i in range(rows):
        for j in range(cols):
            sim_val = sim_int[i, j]
            exp_val = intensity[i, j]
            
            if sim_val > eps and exp_val > eps and isfinite(sim_val) and isfinite(exp_val):
                log_diff = log(exp_val) - log(sim_val)
                if isfinite(log_diff):
                    gf += fabs(log_diff)
                    
                    # Early termination for bad fits
                    if gf > max_gf:
                        return gf
    
    return gf

# Use the ultra-optimized functions as the default
cone_fourier_transform_cython = cone_fourier_transform_cython_ultra
convert_cartesian_cylindrical_cython = convert_cartesian_cylindrical_cython_ultra
sim_cyl_sm_cython = sim_cyl_sm_cython_ultra
sim_cyl_gf_cython = sim_cyl_gf_cython_ultra