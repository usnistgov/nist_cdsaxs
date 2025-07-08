# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, exp, sqrt, pow, isfinite, fabs, atan2, cos
from libc.stdlib cimport malloc, free

# Declare numpy array types
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t CTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def gf_calc_cython(cnp.ndarray[DTYPE_t, ndim=2] sim_int, 
                   cnp.ndarray[DTYPE_t, ndim=2] intensity):
    """
    Cythonized goodness of fit calculation.
    
    Parameters:
    -----------
    sim_int : ndarray[float64, ndim=2]
        Simulated intensity array
    intensity : ndarray[float64, ndim=2]
        Experimental intensity array
        
    Returns:
    --------
    float
        Goodness of fit value
    """
    cdef int rows = sim_int.shape[0]
    cdef int cols = sim_int.shape[1]
    cdef int i, j
    cdef DTYPE_t gf_sum = 0.0
    cdef DTYPE_t log_diff, sim_val, exp_val
    
    for i in range(rows):
        for j in range(cols):
            sim_val = sim_int[i, j]
            exp_val = intensity[i, j]
            
            # Check for valid values
            if sim_val > 0 and exp_val > 0 and isfinite(sim_val) and isfinite(exp_val):
                log_diff = log(exp_val) - log(sim_val)
                if isfinite(log_diff):
                    gf_sum += fabs(log_diff)
    
    return gf_sum

@cython.boundscheck(False)
@cython.wraparound(False)
def debye_waller_factor_cython(cnp.ndarray[DTYPE_t, ndim=2] qx,
                               cnp.ndarray[DTYPE_t, ndim=2] qz,
                               DTYPE_t dw):
    """
    Cythonized Debye-Waller factor calculation.
    
    Parameters:
    -----------
    qx : ndarray[float64, ndim=2]
        X-component of scattering vector
    qz : ndarray[float64, ndim=2]
        Z-component of scattering vector
    dw : float
        Debye-Waller factor
        
    Returns:
    --------
    ndarray[float64, ndim=2]
        Debye-Waller factor array
    """
    cdef int rows = qx.shape[0]
    cdef int cols = qx.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] result = np.zeros((rows, cols), dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t q_squared, dw_squared = dw * dw
    
    for i in range(rows):
        for j in range(cols):
            q_squared = qx[i, j] * qx[i, j] + qz[i, j] * qz[i, j]
            result[i, j] = sqrt(exp(-q_squared * dw_squared))
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def intensity_calculation_cython(cnp.ndarray[CTYPE_t, ndim=2] form_factor,
                                 cnp.ndarray[DTYPE_t, ndim=2] debye_waller,
                                 DTYPE_t i0,
                                 cnp.ndarray[DTYPE_t, ndim=1] background):
    """
    Cythonized intensity calculation with array background support.
    
    Parameters:
    -----------
    form_factor : ndarray[complex128, ndim=2]
        Complex form factor array
    debye_waller : ndarray[float64, ndim=2]
        Debye-Waller factor array
    i0 : float
        Intensity scaling factor
    background : ndarray[float64, ndim=1]
        Background array (one value per column)
        
    Returns:
    --------
    ndarray[float64, ndim=2]
        Calculated intensity
    """
    cdef int rows = form_factor.shape[0]
    cdef int cols = form_factor.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] result = np.zeros((rows, cols), dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t real_part, imag_part, magnitude_squared
    cdef CTYPE_t form_val
    
    for i in range(rows):
        for j in range(cols):
            # Apply Debye-Waller factor to form factor
            form_val = form_factor[i, j] * debye_waller[i, j]
            
            # Calculate magnitude squared
            real_part = form_val.real
            imag_part = form_val.imag
            magnitude_squared = real_part * real_part + imag_part * imag_part
            
            # Calculate intensity with background
            result[i, j] = magnitude_squared * i0 + background[j]
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def intensity_calculation_scalar_bg_cython(cnp.ndarray[CTYPE_t, ndim=2] form_factor,
                                          cnp.ndarray[DTYPE_t, ndim=2] debye_waller,
                                          DTYPE_t i0,
                                          DTYPE_t background):
    """
    Cythonized intensity calculation with scalar background.
    
    Parameters:
    -----------
    form_factor : ndarray[complex128, ndim=2]
        Complex form factor array
    debye_waller : ndarray[float64, ndim=2]
        Debye-Waller factor array
    i0 : float
        Intensity scaling factor
    background : float
        Scalar background value
        
    Returns:
    --------
    ndarray[float64, ndim=2]
        Calculated intensity
    """
    cdef int rows = form_factor.shape[0]
    cdef int cols = form_factor.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] result = np.zeros((rows, cols), dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t real_part, imag_part, magnitude_squared
    cdef CTYPE_t form_val
    
    for i in range(rows):
        for j in range(cols):
            # Apply Debye-Waller factor to form factor
            form_val = form_factor[i, j] * debye_waller[i, j]
            
            # Calculate magnitude squared
            real_part = form_val.real
            imag_part = form_val.imag
            magnitude_squared = real_part * real_part + imag_part * imag_part
            
            # Calculate intensity with background
            result[i, j] = magnitude_squared * i0 + background
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def convert_cartesian_cylindrical_cython(cnp.ndarray[DTYPE_t, ndim=2] qx,
                                        cnp.ndarray[DTYPE_t, ndim=2] qy):
    """
    Cythonized conversion from Cartesian to cylindrical coordinates.
    
    Parameters:
    -----------
    qx : ndarray[float64, ndim=2]
        X-component of scattering vector
    qy : ndarray[float64, ndim=2]
        Y-component of scattering vector
        
    Returns:
    --------
    tuple
        (qr, alpha) arrays
    """
    cdef int rows = qx.shape[0]
    cdef int cols = qx.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] qr = np.zeros((rows, cols), dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=2] alpha = np.zeros((rows, cols), dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t qx_val, qy_val, alpha_val
    
    for i in range(rows):
        for j in range(cols):
            qx_val = qx[i, j]
            qy_val = qy[i, j]
            
            # Handle division by zero
            if qx_val != 0.0:
                alpha_val = atan2(qy_val, qx_val)
                qr[i, j] = qx_val / cos(alpha_val)
            else:
                alpha_val = 0.0
                qr[i, j] = 0.0
            
            alpha[i, j] = alpha_val
    
    return qr, alpha