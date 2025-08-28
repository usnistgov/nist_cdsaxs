# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp, log, sqrt, pow, isfinite, fabs, sin, cos
from libc.stdlib cimport malloc, free

# Declare numpy array types
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.complex128_t CTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sym_coord_assign_cython(cnp.ndarray[DTYPE_t, ndim=2] par,
                           int layers,
                           cnp.ndarray[DTYPE_t, ndim=1] sld_values):
    """
    Cythonized coordinate assignment for symmetric trapezoids with SLD support.
    
    Parameters:
    -----------
    par : ndarray[float64, ndim=2]
        Parameter array with shape (layers+1, 2) containing width and height
    layers : int
        Number of layers
    sld_values : ndarray[float64, ndim=1]
        SLD values for each layer
        
    Returns:
    --------
    ndarray[float64, ndim=3]
        Coordinate array with shape (layers+1, 5, 1)
    """
    cdef cnp.ndarray[DTYPE_t, ndim=3] coord = np.zeros((layers + 1, 5, 1), dtype=np.float64)
    cdef int i
    cdef DTYPE_t width, height, prev_width
    
    for i in range(layers + 1):
        width = par[i, 0]
        
        if i == 0:
            # Bottom layer
            coord[i, 0, 0] = 0.0  # Left edge
            coord[i, 1, 0] = width  # Right edge
            coord[i, 2, 0] = par[i, 1]  # Height
            coord[i, 3, 0] = 0.0  # Reserved
            coord[i, 4, 0] = sld_values[i] if i < len(sld_values) else 0.0  # SLD
        else:
            # Upper layers
            prev_width = par[i-1, 0]
            coord[i, 0, 0] = coord[i-1, 0, 0] + 0.5 * (prev_width - width)
            coord[i, 1, 0] = coord[i, 0, 0] + width
            coord[i, 2, 0] = par[i, 1] if i < layers else 0.0
            coord[i, 3, 0] = 0.0
            
            # SLD assignment: layer i gets sld_values[i] if available
            if i < layers and i < len(sld_values):
                coord[i, 4, 0] = sld_values[i]
            else:
                coord[i, 4, 0] = 0.0  # Top vertex has no SLD
    
    return coord

@cython.boundscheck(False)
@cython.wraparound(False)
def free_form_trapezoid_cython(cnp.ndarray[DTYPE_t, ndim=3] coord,
                              int layers,
                              cnp.ndarray[DTYPE_t, ndim=2] qx,
                              cnp.ndarray[DTYPE_t, ndim=2] qz):
    """
    Cythonized form factor calculation for free-form trapezoids.
    
    Parameters:
    -----------
    coord : ndarray[float64, ndim=3]
        Coordinate array with shape (layers+1, 5, 1)
    layers : int
        Number of layers
    qx : ndarray[float64, ndim=2]
        X-component of scattering vector
    qz : ndarray[float64, ndim=2]
        Z-component of scattering vector
        
    Returns:
    --------
    ndarray[complex128, ndim=2]
        Complex form factor array
    """
    cdef int rows = qx.shape[0]
    cdef int cols = qx.shape[1]
    cdef cnp.ndarray[CTYPE_t, ndim=2] form = np.zeros((rows, cols), dtype=np.complex128)
    
    cdef int i, row, col
    cdef DTYPE_t h1 = coord[0, 3, 0]
    cdef DTYPE_t h2 = h1
    cdef DTYPE_t x1, x2, x3, x4, sl, sr
    cdef DTYPE_t qx_val, qz_val, sld_val
    cdef CTYPE_t a1, a2, term
    cdef DTYPE_t real_part, imag_part
    cdef DTYPE_t phase1, phase2, phase3, phase4
    cdef DTYPE_t cos_val, sin_val
    
    for i in range(layers):
        h2 = h2 + coord[i, 2, 0]
        if i > 0:
            h1 = h1 + coord[i-1, 2, 0]
        
        x1 = coord[i, 0, 0]
        x4 = coord[i, 1, 0]
        x2 = coord[i+1, 0, 0]
        x3 = coord[i+1, 1, 0]
        sld_val = coord[i, 4, 0]
        
        # Avoid division by zero
        if fabs(x2 - x1) < 1e-6:
            x2 = x1 - 1e-6
        if fabs(x4 - x3) < 1e-6:
            x4 = x3 - 1e-6
        
        sl = coord[i, 2, 0] / (x2 - x1)
        sr = -coord[i, 2, 0] / (x4 - x3)
        
        for row in range(rows):
            for col in range(cols):
                qx_val = qx[row, col]
                qz_val = qz[row, col]
                
                if qx_val != 0.0:
                    # Calculate A1 term
                    real_part = qx_val * (h1 - sr * x4) / sr
                    phase1 = -h2 * (qx_val / sr + qz_val)
                    phase2 = -h1 * (qx_val / sr + qz_val)
                    
                    cos_val = cos(real_part)
                    sin_val = sin(real_part)
                    a1 = ((cos_val + 1j * sin_val) / (qx_val / sr + qz_val)) * (
                        (cos(phase1) + 1j * sin(phase1)) - (cos(phase2) + 1j * sin(phase2))
                    )
                    
                    # Calculate A2 term
                    real_part = qx_val * (h1 - sl * x1) / sl
                    phase3 = -h2 * (qx_val / sl + qz_val)
                    phase4 = -h1 * (qx_val / sl + qz_val)
                    
                    cos_val = cos(real_part)
                    sin_val = sin(real_part)
                    a2 = ((cos_val + 1j * sin_val) / (qx_val / sl + qz_val)) * (
                        (cos(phase3) + 1j * sin(phase3)) - (cos(phase4) + 1j * sin(phase4))
                    )
                    
                    term = (1j / qx_val) * (a1 - a2) * sld_val
                    form[row, col] = form[row, col] + term
    
    return form

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_trap_sm_cython(cnp.ndarray[DTYPE_t, ndim=2] par,
                      int layers,
                      cnp.ndarray[DTYPE_t, ndim=2] qx,
                      cnp.ndarray[DTYPE_t, ndim=2] qz,
                      DTYPE_t dw,
                      DTYPE_t i0,
                      cnp.ndarray[DTYPE_t, ndim=1] sld_values,
                      cnp.ndarray[DTYPE_t, ndim=1] background):
    """
    Cythonized simulation for single material trapezoid with optional array background.
    
    Parameters:
    -----------
    par : ndarray[float64, ndim=2]
        Parameter array
    layers : int
        Number of layers
    qx : ndarray[float64, ndim=2]
        X-component of scattering vector
    qz : ndarray[float64, ndim=2]
        Z-component of scattering vector
    dw : float
        Debye-Waller factor
    i0 : float
        Intensity scaling factor
    sld_values : ndarray[float64, ndim=1]
        SLD values for each layer
    background : ndarray[float64, ndim=1] or None
        Background array (one value per column), can be None
        
    Returns:
    --------
    ndarray[float64, ndim=2]
        Simulated intensity
    """
    # Generate coordinates with SLD
    cdef cnp.ndarray[DTYPE_t, ndim=3] coord = sym_coord_assign_cython(par, layers, sld_values)
    
    # Calculate form factor
    cdef cnp.ndarray[CTYPE_t, ndim=2] form = free_form_trapezoid_cython(coord, layers, qx, qz)
    
    # Calculate Debye-Waller factor and intensity
    cdef int rows = qx.shape[0]
    cdef int cols = qx.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] sim_int = np.zeros((rows, cols), dtype=np.float64)
    
    cdef int i, j
    cdef DTYPE_t q_squared, dw_squared = dw * dw
    cdef CTYPE_t form_val
    cdef DTYPE_t real_part, imag_part, magnitude_squared, m_factor
    cdef DTYPE_t bg_val
    cdef bint has_background = background is not None
    
    for i in range(rows):
        for j in range(cols):
            # Calculate Debye-Waller factor
            q_squared = qx[i, j] * qx[i, j] + qz[i, j] * qz[i, j]
            m_factor = sqrt(exp(-q_squared * dw_squared))
            
            # Apply Debye-Waller factor to form factor
            form_val = form[i, j] * m_factor
            
            # Calculate magnitude squared
            real_part = form_val.real
            imag_part = form_val.imag
            magnitude_squared = real_part * real_part + imag_part * imag_part
            
            # Calculate intensity with background
            if has_background:
                bg_val = background[j]
            else:
                bg_val = 0.0
                
            sim_int[i, j] = magnitude_squared * i0 + bg_val
    
    return sim_int

@cython.boundscheck(False)
@cython.wraparound(False)
def sim_trap_gf_cython(cnp.ndarray[DTYPE_t, ndim=1] optimization_values,
                      cnp.ndarray[DTYPE_t, ndim=2] par_template,
                      int layers,
                      cnp.ndarray[DTYPE_t, ndim=2] qx,
                      cnp.ndarray[DTYPE_t, ndim=2] qz,
                      cnp.ndarray[DTYPE_t, ndim=2] intensity,
                      DTYPE_t dw,
                      DTYPE_t i0,
                      cnp.ndarray[DTYPE_t, ndim=1] sld_values,
                      cnp.ndarray[DTYPE_t, ndim=1] background):
    """
    Cythonized goodness of fit calculation for trapezoid optimization.
    
    Parameters:
    -----------
    optimization_values : ndarray[float64, ndim=1]
        Array of optimization parameter values
    par_template : ndarray[float64, ndim=2]
        Template PAR array to be updated
    layers : int
        Number of layers
    qx : ndarray[float64, ndim=2]
        X-component of scattering vector
    qz : ndarray[float64, ndim=2]
        Z-component of scattering vector
    intensity : ndarray[float64, ndim=2]
        Measured intensity data
    dw : float
        Debye-Waller factor
    i0 : float
        Intensity scaling factor
    sld_values : ndarray[float64, ndim=1]
        SLD values for each layer
    background : ndarray[float64, ndim=1] or None
        Background array, can be None
        
    Returns:
    --------
    float
        Goodness of fit value
    """
    # Note: This is a simplified version. In practice, you'd need to map
    # optimization_values to the appropriate PAR positions based on param_names
    # For now, assuming optimization_values directly maps to PAR flattened
    
    cdef cnp.ndarray[DTYPE_t, ndim=2] par = par_template.copy()
    cdef int param_idx = 0
    
    # Update PAR with optimization values (simplified mapping)
    cdef int i, j
    for i in range(par.shape[0]):
        for j in range(par.shape[1]):
            if param_idx < len(optimization_values):
                par[i, j] = optimization_values[param_idx]
                param_idx += 1
    
    # Simulate intensity
    cdef cnp.ndarray[DTYPE_t, ndim=2] sim_int = sim_trap_sm_cython(
        par, layers, qx, qz, dw, i0, sld_values, background
    )
    
    # Calculate goodness of fit
    cdef DTYPE_t gf = 0.0
    cdef int rows = intensity.shape[0]
    cdef int cols = intensity.shape[1]
    cdef DTYPE_t log_diff, sim_val, exp_val
    
    for i in range(rows):
        for j in range(cols):
            sim_val = sim_int[i, j]
            exp_val = intensity[i, j]
            
            if sim_val > 0 and exp_val > 0 and isfinite(sim_val) and isfinite(exp_val):
                log_diff = log(exp_val) - log(sim_val)
                if isfinite(log_diff):
                    gf += fabs(log_diff)
    
    return gf