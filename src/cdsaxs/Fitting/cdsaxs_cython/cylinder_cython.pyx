# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp, log, sqrt, pow, isfinite, fabs, sin, cos, atan2
from libc.stdlib cimport malloc, free

# Declare numpy array types
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t CTYPE_t

cdef double bessel_j1_fast(double x) nogil:
    """Fast approximation for Bessel function J1(x)."""
    # Declare all variables at the top
    cdef double ax, y, ans1, ans2, z, xx
    
    if x == 0.0:
        return 0.0
    elif fabs(x) < 8.0:
        # Use series expansion for small x
        ax = fabs(x)
        y = x * x
        ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 
                + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
        ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 
                + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
        return ans1 / ans2
    else:
        # Use asymptotic expansion for large x
        z = 8.0 / x
        y = z * z
        xx = x - 2.356194491
        ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 
                + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
        ans2 = 0.04687499995 + y * (-0.2002690873e-3 
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 
                + y * 0.105787412e-6)))
        return sqrt(0.636619772 / x) * (cos(xx) * ans1 - z * sin(xx) * ans2)

@cython.boundscheck(False)
@cython.wraparound(False)
def cone_fourier_transform_cython(cnp.ndarray[DTYPE_t, ndim=2] par,
                                 int layers,
                                 cnp.ndarray[DTYPE_t, ndim=2] qr,
                                 cnp.ndarray[DTYPE_t, ndim=2] qz,
                                 cnp.ndarray[cnp.int32_t, ndim=1] discretization,
                                 cnp.ndarray[DTYPE_t, ndim=1] sld_values):
    """
    Optimized Cythonized Fourier transform for a cone in cylindrical coordinates.
    """
    # Declare all variables at the top
    cdef int rows = qr.shape[0]
    cdef int cols = qr.shape[1]
    cdef cnp.ndarray[CTYPE_t, ndim=2] form = np.zeros((rows, cols), dtype=np.complex128)
    
    cdef DTYPE_t h1 = 0.0
    cdef DTYPE_t h2 = 0.0
    cdef int i, ii, row, col
    cdef DTYPE_t stepsize, r1, r2, slope
    cdef DTYPE_t ri1, ri2, qr_val, qz_val, sld_val
    cdef DTYPE_t bessel1, bessel2, pi_factor, phase1, phase2
    cdef DTYPE_t cos_val1, sin_val1, cos_val2, sin_val2
    cdef DTYPE_t real_contrib, imag_contrib
    cdef DTYPE_t two_pi_over_qr, stepsize_half_sld
    cdef int n_steps
    cdef DTYPE_t *z_array
    
    # Pre-calculate constants
    pi_factor = 2.0 * np.pi
    
    for i in range(layers):
        h2 = h2 + par[i, 1]
        stepsize = par[i, 1] / discretization[i]
        stepsize_half_sld = stepsize * 0.5 * sld_values[i] if i < len(sld_values) else stepsize * 0.5
        
        if i > 0:
            h1 = h1 + par[i-1, 1]
        
        r1 = par[i, 0]
        r2 = par[i+1, 0]
        
        # Avoid division by zero
        if fabs(r1 - r2) < 1e-6:
            r1 = r1 + 1e-6
        
        slope = (h2 - h1) / (r2 - r1)
        
        # Pre-calculate z points for this layer
        n_steps = int((h2 - h1) / stepsize) + 1
        z_array = <DTYPE_t *>malloc(n_steps * sizeof(DTYPE_t))
        
        if z_array == NULL:
            raise MemoryError("Could not allocate memory for z_array")
        
        try:
            # Fill z array
            for ii in range(n_steps):
                z_array[ii] = h1 + ii * stepsize
            
            # Main calculation loop - optimized
            for row in range(rows):
                for col in range(cols):
                    qr_val = qr[row, col]
                    qz_val = qz[row, col]
                    
                    if qr_val > 1e-10:  # Avoid very small values
                        two_pi_over_qr = pi_factor / qr_val
                        
                        # Inner loop over z steps
                        for ii in range(n_steps - 1):
                            ri1 = (z_array[ii] - h1) / slope + r1
                            ri2 = (z_array[ii+1] - h1) / slope + r1
                            
                            # Calculate Bessel functions using fast approximation
                            bessel1 = bessel_j1_fast(qr_val * ri1)
                            bessel2 = bessel_j1_fast(qr_val * ri2)
                            
                            # Calculate phases
                            phase1 = qz_val * z_array[ii]
                            phase2 = qz_val * z_array[ii+1]
                            
                            # Calculate complex contributions efficiently
                            cos_val1 = cos(phase1)
                            sin_val1 = sin(phase1)
                            cos_val2 = cos(phase2)
                            sin_val2 = sin(phase2)
                            
                            # Combine real and imaginary parts
                            real_contrib = (ri1 * bessel1 * cos_val1 + ri2 * bessel2 * cos_val2) * two_pi_over_qr
                            imag_contrib = (ri1 * bessel1 * sin_val1 + ri2 * bessel2 * sin_val2) * two_pi_over_qr
                            
                            # Add to form factor
                            form[row, col].real = form[row, col].real + real_contrib * stepsize_half_sld
                            form[row, col].imag = form[row, col].imag + imag_contrib * stepsize_half_sld
        
        finally:
            free(z_array)
    
    return form

@cython.boundscheck(False) 
@cython.wraparound(False)
def convert_cartesian_cylindrical_cython(cnp.ndarray[DTYPE_t, ndim=2] qx,
                                        cnp.ndarray[DTYPE_t, ndim=2] qy):
    """
    Highly optimized conversion from Cartesian to cylindrical coordinates.
    """
    # Declare all variables at the top
    cdef int rows = qx.shape[0]
    cdef int cols = qx.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] qr = np.empty((rows, cols), dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=2] alpha = np.empty((rows, cols), dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t qx_val, qy_val, qr_val
    
    for i in range(rows):
        for j in range(cols):
            qx_val = qx[i, j]
            qy_val = qy[i, j]
            
            # Fast magnitude calculation
            qr_val = sqrt(qx_val * qx_val + qy_val * qy_val)
            qr[i, j] = qr_val
            
            # Fast angle calculation
            if qr_val > 1e-12:
                alpha[i, j] = atan2(qy_val, qx_val)
            else:
                alpha[i, j] = 0.0
    
    return qr, alpha

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_cyl_sm_cython(cnp.ndarray[DTYPE_t, ndim=2] par,
                     int layers,
                     cnp.ndarray[DTYPE_t, ndim=2] qr,
                     cnp.ndarray[DTYPE_t, ndim=2] qz,
                     DTYPE_t dw,
                     DTYPE_t i0,
                     DTYPE_t background,
                     cnp.ndarray[cnp.int32_t, ndim=1] discretization,
                     cnp.ndarray[DTYPE_t, ndim=1] sld_values):
    """
    Cythonized cylinder simulation.
    """
    # Declare all variables at the top
    cdef cnp.ndarray[CTYPE_t, ndim=2] form = cone_fourier_transform_cython(par, layers, qr, qz, discretization, sld_values)
    cdef int rows = qr.shape[0]
    cdef int cols = qr.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] sim_int = np.zeros((rows, cols), dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t q_squared, dw_squared = dw * dw
    cdef DTYPE_t m_val
    cdef CTYPE_t form_val
    cdef DTYPE_t real_part, imag_part, magnitude_squared
    
    for i in range(rows):
        for j in range(cols):
            # Calculate Debye-Waller factor
            q_squared = qr[i, j] * qr[i, j] + qz[i, j] * qz[i, j]
            m_val = sqrt(exp(-q_squared * dw_squared))
            
            # Apply Debye-Waller factor to form factor
            form_val = form[i, j] * m_val
            
            # Calculate magnitude squared
            real_part = form_val.real
            imag_part = form_val.imag
            magnitude_squared = real_part * real_part + imag_part * imag_part
            
            # Calculate intensity
            sim_int[i, j] = magnitude_squared * i0 + background
    
    return sim_int

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_cyl_gf_cython(cnp.ndarray[DTYPE_t, ndim=1] sim_par,
                     int layers,
                     cnp.ndarray[DTYPE_t, ndim=2] intensity,
                     cnp.ndarray[DTYPE_t, ndim=2] qr,
                     cnp.ndarray[DTYPE_t, ndim=2] qz,
                     cnp.ndarray[cnp.int32_t, ndim=1] discretization,
                     cnp.ndarray[DTYPE_t, ndim=1] sld_values):
    """
    Cythonized cylinder GF calculation.
    """
    # Declare all variables at the top
    cdef cnp.ndarray[DTYPE_t, ndim=2] pars = np.zeros((layers + 1, 2), dtype=np.float64)
    cdef DTYPE_t i0, dw, background
    cdef int i, j
    cdef int rows = intensity.shape[0]
    cdef int cols = intensity.shape[1]
    cdef DTYPE_t gf = 0.0
    cdef DTYPE_t log_diff, sim_val, exp_val
    cdef cnp.ndarray[DTYPE_t, ndim=2] sim_int
    
    # Extract parameters
    pars[:, 0:2] = sim_par[0:(layers + 1) * 2].reshape((layers + 1, 2))
    i0 = sim_par[(layers + 1) * 2]
    dw = sim_par[(layers + 1) * 2 + 1]
    background = sim_par[(layers + 1) * 2 + 2]
    
    # Calculate simulated intensity
    sim_int = sim_cyl_sm_cython(pars, layers, qr, qz, dw, i0, background, discretization, sld_values)
    
    # Calculate goodness of fit
    for i in range(rows):
        for j in range(cols):
            sim_val = sim_int[i, j]
            exp_val = intensity[i, j]
            
            if sim_val > 0 and exp_val > 0 and isfinite(sim_val) and isfinite(exp_val):
                log_diff = log(exp_val) - log(sim_val)
                if isfinite(log_diff):
                    gf += fabs(log_diff)
    
    return gf