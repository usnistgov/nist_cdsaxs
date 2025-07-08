# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp, log, sqrt, pow, isfinite, fabs, sin, cos, atan2
from libc.stdlib cimport malloc, free

# Import scipy.special.jv as a Python function
# Note: For production, you might want to use a C implementation of Bessel functions
from scipy.special import jv

# Declare numpy array types
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t CTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def cone_fourier_transform_cython(cnp.ndarray[DTYPE_t, ndim=2] par,
                                 int layers,
                                 cnp.ndarray[DTYPE_t, ndim=2] qr,
                                 cnp.ndarray[DTYPE_t, ndim=2] qz,
                                 cnp.ndarray[cnp.int32_t, ndim=1] discretization,
                                 cnp.ndarray[DTYPE_t, ndim=1] sld_values):
    """
    Cythonized Fourier transform for a cone in cylindrical coordinates with SLD support.
    
    Parameters:
    -----------
    par : ndarray[float64, ndim=2]
        Parameter array with shape (layers+1, 2) containing radius and height
    layers : int
        Number of layers
    qr : ndarray[float64, ndim=2]
        Radial component of scattering vector
    qz : ndarray[float64, ndim=2]
        Z-component of scattering vector
    discretization : ndarray[int32, ndim=1]
        Number of discretization steps for each layer
    sld_values : ndarray[float64, ndim=1]
        SLD values for each layer
        
    Returns:
    --------
    ndarray[complex128, ndim=2]
        Complex form factor array
    """
    cdef int rows = qr.shape[0]
    cdef int cols = qr.shape[1]
    cdef cnp.ndarray[CTYPE_t, ndim=2] form = np.zeros((rows, cols), dtype=np.complex128)
    
    cdef DTYPE_t h1 = 0.0
    cdef DTYPE_t h2 = 0.0
    cdef int i, ii, row, col, z_steps
    cdef DTYPE_t stepsize, r1, r2, slope
    cdef DTYPE_t ri1, ri2, qr_val, qz_val, sld_val
    cdef DTYPE_t bessel1, bessel2, pi_factor, phase1, phase2
    cdef DTYPE_t cos_val1, sin_val1, cos_val2, sin_val2
    cdef CTYPE_t fa, fb, contrib
    cdef cnp.ndarray[DTYPE_t, ndim=1] z
    
    pi_factor = 2.0 * np.pi
    
    for i in range(layers):
        h2 = h2 + par[i, 1]
        stepsize = par[i, 1] / discretization[i]
        
        if i > 0:
            h1 = h1 + par[i-1, 1]
        
        # Create z array for this layer
        z = np.arange(h1, h2 + 0.01, stepsize)
        
        r1 = par[i, 0]
        r2 = par[i+1, 0]
        sld_val = sld_values[i] if i < len(sld_values) else 1.0
        
        # Avoid division by zero
        if fabs(r1 - r2) < 1e-6:
            r1 = r1 + 1e-6
        
        slope = (h2 - h1) / (r2 - r1)
        
        for ii in range(len(z) - 1):
            ri1 = (z[ii] - h1) / slope + r1
            ri2 = (z[ii+1] - h1) / slope + r1
            
            for row in range(rows):
                for col in range(cols):
                    qr_val = qr[row, col]
                    qz_val = qz[row, col]
                    
                    if qr_val > 0:
                        # Calculate Bessel function contributions
                        # Using Python jv function - in production, consider C implementation
                        bessel1 = jv(1, qr_val * ri1)
                        bessel2 = jv(1, qr_val * ri2)
                        
                        # Calculate phase terms
                        phase1 = qz_val * z[ii]
                        phase2 = qz_val * z[ii+1]
                        
                        # Create complex exponential terms using cos and sin
                        cos_val1 = cos(phase1)
                        sin_val1 = sin(phase1)
                        cos_val2 = cos(phase2)
                        sin_val2 = sin(phase2)
                        
                        # Calculate contributions
                        fa = (pi_factor * ri1 / qr_val) * bessel1 * (cos_val1 + 1j * sin_val1)
                        fb = (pi_factor * ri2 / qr_val) * bessel2 * (cos_val2 + 1j * sin_val2)
                        
                        contrib = stepsize * (fb + fa) / 2.0 * sld_val
                        form[row, col] = form[row, col] + contrib
    
    return form

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
    Cythonized simulation for single material cylinder.
    
    Parameters:
    -----------
    par : ndarray[float64, ndim=2]
        Parameter array
    layers : int
        Number of layers
    qr : ndarray[float64, ndim=2]
        Radial component of scattering vector
    qz : ndarray[float64, ndim=2]
        Z-component of scattering vector
    dw : float
        Debye-Waller factor
    i0 : float
        Intensity scaling factor
    background : float
        Background intensity
    discretization : ndarray[int32, ndim=1]
        Discretization for each layer
    sld_values : ndarray[float64, ndim=1]
        SLD values for each layer
        
    Returns:
    --------
    ndarray[float64, ndim=2]
        Simulated intensity
    """
    # Calculate form factor
    cdef cnp.ndarray[CTYPE_t, ndim=2] form = cone_fourier_transform_cython(
        par, layers, qr, qz, discretization, sld_values
    )
    
    # Calculate intensity
    cdef int rows = qr.shape[0]
    cdef int cols = qr.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] sim_int = np.zeros((rows, cols), dtype=np.float64)
    
    cdef int i, j
    cdef DTYPE_t q_squared, dw_squared = dw * dw
    cdef CTYPE_t form_val
    cdef DTYPE_t real_part, imag_part, magnitude_squared, m_factor
    
    for i in range(rows):
        for j in range(cols):
            # Calculate Debye-Waller factor
            q_squared = qr[i, j] * qr[i, j] + qz[i, j] * qz[i, j]
            m_factor = sqrt(exp(-q_squared * dw_squared))
            
            # Apply Debye-Waller factor to form factor
            form_val = form[i, j] * m_factor
            
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
    Cythonized goodness of fit calculation for cylinder optimization.
    
    Parameters:
    -----------
    sim_par : ndarray[float64, ndim=1]
        1D array containing all simulation parameters (layer dimensions, I0, DW, Bk)
    layers : int
        Number of layers
    intensity : ndarray[float64, ndim=2]
        Measured intensity data
    qr : ndarray[float64, ndim=2]
        Radial component of scattering vector
    qz : ndarray[float64, ndim=2]
        Z-component of scattering vector
    discretization : ndarray[int32, ndim=1]
        Discretization for each layer
    sld_values : ndarray[float64, ndim=1]
        SLD values for each layer
        
    Returns:
    --------
    float
        Chi-square value representing goodness of fit
    """
    # Reshape parameters
    cdef cnp.ndarray[DTYPE_t, ndim=2] pars = np.zeros((layers + 1, 2), dtype=np.float64)
    cdef int param_end = (layers + 1) * 2
    
    # Extract PAR array
    cdef int i, j, idx = 0
    for i in range(layers + 1):
        for j in range(2):
            pars[i, j] = sim_par[idx]
            idx += 1
    
    # Extract global parameters
    cdef DTYPE_t i0 = sim_par[param_end]
    cdef DTYPE_t dw = sim_par[param_end + 1] 
    cdef DTYPE_t bk = sim_par[param_end + 2]
    
    # Calculate form factor
    cdef cnp.ndarray[CTYPE_t, ndim=2] f1 = cone_fourier_transform_cython(
        pars, layers, qr, qz, discretization, sld_values
    )
    
    # Calculate intensity and goodness of fit
    cdef int rows = qr.shape[0]
    cdef int cols = qr.shape[1]
    
    cdef DTYPE_t q_squared, dw_squared = dw * dw
    cdef CTYPE_t form_val
    cdef DTYPE_t real_part, imag_part, magnitude_squared, m_factor
    cdef DTYPE_t chi2 = 0.0, log_diff, sim_val, exp_val
    
    for i in range(rows):
        for j in range(cols):
            # Calculate Debye-Waller factor
            q_squared = qr[i, j] * qr[i, j] + qz[i, j] * qz[i, j]
            m_factor = sqrt(exp(-q_squared * dw_squared))
            
            # Apply Debye-Waller factor to form factor
            form_val = f1[i, j] * m_factor
            
            # Calculate magnitude squared
            real_part = form_val.real
            imag_part = form_val.imag
            magnitude_squared = real_part * real_part + imag_part * imag_part
            
            # Calculate intensity
            sim_val = magnitude_squared * i0 + bk
            exp_val = intensity[i, j]
            
            # Calculate chi-square contribution
            if sim_val > 0 and exp_val > 0 and isfinite(sim_val) and isfinite(exp_val):
                log_diff = log(exp_val) - log(sim_val)
                if isfinite(log_diff):
                    chi2 += fabs(log_diff)
    
    return chi2

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
            if fabs(qx_val) > 1e-12:
                alpha_val = atan2(qy_val, qx_val)
                qr[i, j] = qx_val / cos(alpha_val)
            else:
                alpha_val = 0.0
                qr[i, j] = 0.0
            
            alpha[i, j] = alpha_val
    
    return qr, alpha