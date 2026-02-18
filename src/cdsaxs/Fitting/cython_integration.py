"""
Integration module for Cython-accelerated CDSAXS functions with fallback to original implementations.
"""

import numpy as np
import warnings
import time
from functools import wraps

# Global flag to control Cython usage
USE_CYTHON = True
CYTHON_AVAILABLE = False

# Try to import Cython modules
try:
    from cdsaxs_cython import base_cython, trapezoid_cython, cylinder_cython
    CYTHON_AVAILABLE = True
    print("Cython modules loaded successfully - accelerated functions available")
except ImportError as e:
    print(f"Cython modules not available: {e}")
    print("Falling back to pure Python implementations")
    CYTHON_AVAILABLE = False
    
def enable_cython():
    """Enable Cython acceleration if available."""
    global USE_CYTHON
    if CYTHON_AVAILABLE:
        USE_CYTHON = True
        print("Cython acceleration enabled")
    else:
        print("Cython not available - cannot enable acceleration")

def disable_cython():
    """Disable Cython acceleration and use pure Python."""
    global USE_CYTHON
    USE_CYTHON = False
    print("Cython acceleration disabled - using pure Python")

def cython_fallback(func_name):
    """
    Decorator to handle fallback from Cython to pure Python implementations.
    
    Parameters:
    -----------
    func_name : str
        Name of the function for error reporting
    """
    def decorator(cython_func):
        @wraps(cython_func)
        def wrapper(*args, **kwargs):
            if not USE_CYTHON or not CYTHON_AVAILABLE:
                # Return None to signal fallback should be used
                return None
            
            try:
                return cython_func(*args, **kwargs)
            except Exception as e:
                warnings.warn(
                    f"Cython function {func_name} failed with error: {e}. "
                    f"Falling back to pure Python implementation."
                )
                return None
        return wrapper
    return decorator

def benchmark_function(cython_func, python_func, *args, **kwargs):
    """
    Benchmark Cython vs Python implementations.
    
    Parameters:
    -----------
    cython_func : callable
        Cython implementation
    python_func : callable
        Python implementation
    *args, **kwargs : arguments to pass to both functions
        
    Returns:
    --------
    dict
        Benchmark results
    """
    results = {
        'cython_available': CYTHON_AVAILABLE,
        'cython_time': None,
        'python_time': None,
        'speedup': None,
        'results_match': None
    }
    
    if not CYTHON_AVAILABLE:
        print("Cython not available for benchmarking")
        return results
    
    # Benchmark Python implementation
    start_time = time.time()
    python_result = python_func(*args, **kwargs)
    python_time = time.time() - start_time
    results['python_time'] = python_time
    
    # Benchmark Cython implementation
    try:
        start_time = time.time()
        cython_result = cython_func(*args, **kwargs)
        cython_time = time.time() - start_time
        results['cython_time'] = cython_time
        
        # Calculate speedup
        if cython_time > 0:
            results['speedup'] = python_time / cython_time
        
        # Check if results match (within tolerance)
        if isinstance(python_result, np.ndarray) and isinstance(cython_result, np.ndarray):
            if python_result.shape == cython_result.shape:
                # Handle complex arrays
                if np.iscomplexobj(python_result) or np.iscomplexobj(cython_result):
                    results['results_match'] = np.allclose(python_result, cython_result, rtol=1e-10, atol=1e-12)
                else:
                    results['results_match'] = np.allclose(python_result, cython_result, rtol=1e-12, atol=1e-14)
            else:
                results['results_match'] = False
        else:
            # For scalar results
            if isinstance(python_result, (int, float)) and isinstance(cython_result, (int, float)):
                results['results_match'] = abs(python_result - cython_result) < 1e-12
            else:
                results['results_match'] = False
                
    except Exception as e:
        print(f"Cython benchmark failed: {e}")
        results['cython_time'] = None
    
    return results

# Base functions with fallback
@cython_fallback("gf_calc_cython")
def gf_calc_accelerated(sim_int, intensity):
    """Accelerated goodness of fit calculation with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return base_cython.gf_calc_cython(sim_int, intensity)
    return None

@cython_fallback("debye_waller_factor_cython")
def debye_waller_factor_accelerated(qx, qz, dw):
    """Accelerated Debye-Waller factor calculation with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return base_cython.debye_waller_factor_cython(qx, qz, dw)
    return None

@cython_fallback("intensity_calculation_cython")
def intensity_calculation_accelerated(form_factor, debye_waller, i0, background):
    """Accelerated intensity calculation with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        if isinstance(background, np.ndarray):
            return base_cython.intensity_calculation_cython(form_factor, debye_waller, i0, background)
        else:
            return base_cython.intensity_calculation_scalar_bg_cython(form_factor, debye_waller, i0, background)
    return None

# Trapezoid functions with fallback
@cython_fallback("sym_coord_assign_cython")
def sym_coord_assign_accelerated(par, layers, sld_values):
    """Accelerated coordinate assignment with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return trapezoid_cython.sym_coord_assign_cython(par, layers, sld_values)
    return None

@cython_fallback("free_form_trapezoid_cython")
def free_form_trapezoid_accelerated(coord, layers, qx, qz):
    """Accelerated trapezoid form factor calculation with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return trapezoid_cython.free_form_trapezoid_cython(coord, layers, qx, qz)
    return None

@cython_fallback("sim_trap_sm_cython")
def sim_trap_sm_accelerated(par, layers, qx, qz, dw, i0, sld_values, background=None):
    """Accelerated trapezoid simulation with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return trapezoid_cython.sim_trap_sm_cython(par, layers, qx, qz, dw, i0, sld_values, background)
    return None

# Cylinder functions with fallback
@cython_fallback("cone_fourier_transform_cython")
def cone_fourier_transform_accelerated(par, layers, qr, qz, discretization, sld_values):
    """Accelerated cylinder Fourier transform with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return cylinder_cython.cone_fourier_transform_cython(par, layers, qr, qz, discretization, sld_values)
    return None

@cython_fallback("sim_cyl_sm_cython")
def sim_cyl_sm_accelerated(par, layers, qr, qz, dw, i0, background, discretization, sld_values, dw2=-1.0):
    """Accelerated cylinder simulation with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return cylinder_cython.sim_cyl_sm_cython(par, layers, qr, qz, dw, i0, background, discretization, sld_values, dw2)
    return None

@cython_fallback("sim_cyl_gf_cython")
def sim_cyl_gf_accelerated(sim_par, layers, intensity, qr, qz, discretization, sld_values):
    """Accelerated cylinder GF calculation with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return cylinder_cython.sim_cyl_gf_cython(sim_par, layers, intensity, qr, qz, discretization, sld_values)
    return None

@cython_fallback("convert_cartesian_cylindrical_cython")
def convert_cartesian_cylindrical_accelerated(qx, qy):
    """Accelerated coordinate conversion with fallback."""
    if CYTHON_AVAILABLE and USE_CYTHON:
        return cylinder_cython.convert_cartesian_cylindrical_cython(qx, qy)
    return None

def run_comprehensive_benchmark(model_instance, n_iterations=5):
    """
    Run comprehensive benchmarks comparing Cython vs Python implementations.
    
    Parameters:
    -----------
    model_instance : CDSAXS_Model
        Model instance with loaded data for benchmarking
    n_iterations : int
        Number of iterations for timing
        
    Returns:
    --------
    dict
        Comprehensive benchmark results
    """
    if not hasattr(model_instance, 'Intensity'):
        print("Model must have data loaded for benchmarking")
        return None
    
    print("Running comprehensive CDSAXS benchmark...")
    print(f"Geometry: {model_instance.geometry}")
    print(f"Layers: {model_instance.layers}")
    print(f"Data shape: {model_instance.Intensity.shape}")
    print(f"Iterations: {n_iterations}")
    print("-" * 60)
    
    results = {}
    
    # Test GF calculation
    print("Benchmarking GF calculation...")
    sim_int = model_instance.simulate_structure()
    if sim_int is not None:
        
        def python_gf():
            return model_instance.GF_calc(sim_int)
        
        def cython_gf():
            return gf_calc_accelerated(sim_int, model_instance.Intensity)
        
        # Run multiple iterations
        python_times = []
        cython_times = []
        
        for i in range(n_iterations):
            # Python timing
            start = time.time()
            python_result = python_gf()
            python_times.append(time.time() - start)
            
            # Cython timing
            if CYTHON_AVAILABLE:
                start = time.time()
                cython_result = cython_gf()
                if cython_result is not None:
                    cython_times.append(time.time() - start)
        
        if cython_times:
            avg_python = np.mean(python_times)
            avg_cython = np.mean(cython_times)
            speedup = avg_python / avg_cython
            
            results['gf_calculation'] = {
                'python_time': avg_python,
                'cython_time': avg_cython,
                'speedup': speedup,
                'results_match': abs(python_result - cython_result) < 1e-10 if cython_result is not None else False
            }
            
            print(f"  Python time: {avg_python:.6f}s")
            print(f"  Cython time: {avg_cython:.6f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Results match: {results['gf_calculation']['results_match']}")
        else:
            print("  Cython not available")
    
    # Test geometry-specific functions
    if model_instance.geometry == 'trapezoid':
        print("\nBenchmarking trapezoid functions...")
        _benchmark_trapezoid_functions(model_instance, n_iterations, results)
    
    elif model_instance.geometry == 'cylinder':
        print("\nBenchmarking cylinder functions...")
        _benchmark_cylinder_functions(model_instance, n_iterations, results)
    
    return results

def _benchmark_trapezoid_functions(model_instance, n_iterations, results):
    """Benchmark trapezoid-specific functions."""
    
    # Test coordinate assignment
    print("  Coordinate assignment...")
    if hasattr(model_instance, 'sld_values'):
        sld_values = model_instance.sld_values
    else:
        sld_values = np.ones(model_instance.layers)
    
    python_times = []
    cython_times = []
    
    for i in range(n_iterations):
        # Python timing
        start = time.time()
        python_coord = model_instance.SymCoordAssign(model_instance.PAR, model_instance.layers, sld_values)
        python_times.append(time.time() - start)
        
        # Cython timing
        if CYTHON_AVAILABLE:
            start = time.time()
            cython_coord = sym_coord_assign_accelerated(model_instance.PAR, model_instance.layers, sld_values)
            if cython_coord is not None:
                cython_times.append(time.time() - start)
    
    if cython_times:
        avg_python = np.mean(python_times)
        avg_cython = np.mean(cython_times)
        speedup = avg_python / avg_cython
        
        results['coord_assignment'] = {
            'python_time': avg_python,
            'cython_time': avg_cython,
            'speedup': speedup,
            'results_match': np.allclose(python_coord, cython_coord, rtol=1e-12) if cython_coord is not None else False
        }
        
        print(f"    Python time: {avg_python:.6f}s")
        print(f"    Cython time: {avg_cython:.6f}s")
        print(f"    Speedup: {speedup:.2f}x")
    
    # Test form factor calculation
    print("  Form factor calculation...")
    python_times = []
    cython_times = []
    
    coord = python_coord  # Use the coordinate result from above
    
    for i in range(n_iterations):
        # Python timing
        start = time.time()
        python_form = model_instance.FreeFormTrapezoid(coord, model_instance.layers, model_instance.Qx, model_instance.Qz)
        python_times.append(time.time() - start)
        
        # Cython timing
        if CYTHON_AVAILABLE:
            start = time.time()
            cython_form = free_form_trapezoid_accelerated(coord, model_instance.layers, model_instance.Qx, model_instance.Qz)
            if cython_form is not None:
                cython_times.append(time.time() - start)
    
    if cython_times:
        avg_python = np.mean(python_times)
        avg_cython = np.mean(cython_times)
        speedup = avg_python / avg_cython
        
        results['form_factor'] = {
            'python_time': avg_python,
            'cython_time': avg_cython,
            'speedup': speedup,
            'results_match': np.allclose(python_form, cython_form, rtol=1e-10) if cython_form is not None else False
        }
        
        print(f"    Python time: {avg_python:.6f}s")
        print(f"    Cython time: {avg_cython:.6f}s")
        print(f"    Speedup: {speedup:.2f}x")

def _benchmark_cylinder_functions(model_instance, n_iterations, results):
    """Benchmark cylinder-specific functions."""
    
    # Test coordinate conversion
    print("  Coordinate conversion...")
    python_times = []
    cython_times = []
    
    for i in range(n_iterations):
        # Python timing
        start = time.time()
        python_result = model_instance.convert_Cartesian_Cylindrical()
        python_times.append(time.time() - start)
        
        # Cython timing
        if CYTHON_AVAILABLE:
            start = time.time()
            cython_result = convert_cartesian_cylindrical_accelerated(model_instance.Qx, model_instance.Qy)
            if cython_result is not None:
                cython_times.append(time.time() - start)
    
    if cython_times:
        avg_python = np.mean(python_times)
        avg_cython = np.mean(cython_times)
        speedup = avg_python / avg_cython
        
        # Compare results
        qr_match = np.allclose(model_instance.Qr, cython_result[0], rtol=1e-12) if cython_result is not None else False
        alpha_match = np.allclose(model_instance.Alpha, cython_result[1], rtol=1e-12) if cython_result is not None else False
        
        results['coord_conversion'] = {
            'python_time': avg_python,
            'cython_time': avg_cython,
            'speedup': speedup,
            'results_match': qr_match and alpha_match
        }
        
        print(f"    Python time: {avg_python:.6f}s")
        print(f"    Cython time: {avg_cython:.6f}s")
        print(f"    Speedup: {speedup:.2f}x")
    
    # Test Fourier transform
    print("  Fourier transform...")
    if hasattr(model_instance, 'sld_values'):
        sld_values = model_instance.sld_values
    else:
        sld_values = np.ones(model_instance.layers)
    
    python_times = []
    cython_times = []
    
    for i in range(n_iterations):
        # Python timing
        start = time.time()
        python_form = model_instance.ConeFourierTransform(model_instance.discretization, sld_values)
        python_times.append(time.time() - start)
        
        # Cython timing
        if CYTHON_AVAILABLE:
            start = time.time()
            cython_form = cone_fourier_transform_accelerated(
                model_instance.PAR, model_instance.layers, model_instance.Qr, model_instance.Qz, 
                np.array(model_instance.discretization, dtype=np.int32), sld_values
            )
            if cython_form is not None:
                cython_times.append(time.time() - start)
    
    if cython_times:
        avg_python = np.mean(python_times)
        avg_cython = np.mean(cython_times)
        speedup = avg_python / avg_cython
        
        results['fourier_transform'] = {
            'python_time': avg_python,
            'cython_time': avg_cython,
            'speedup': speedup,
            'results_match': np.allclose(python_form, cython_form, rtol=1e-10) if cython_form is not None else False
        }
        
        print(f"    Python time: {avg_python:.6f}s")
        print(f"    Cython time: {avg_cython:.6f}s")
        print(f"    Speedup: {speedup:.2f}x")

def print_benchmark_summary(results):
    """Print a summary of benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if not results:
        print("No benchmark results available")
        return
    
    total_speedup = []
    all_match = True
    
    for function_name, result in results.items():
        if isinstance(result, dict) and 'speedup' in result:
            speedup = result['speedup']
            matches = result['results_match']
            
            print(f"{function_name.replace('_', ' ').title()}:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Results match: {matches}")
            
            total_speedup.append(speedup)
            if not matches:
                all_match = False
    
    if total_speedup:
        avg_speedup = np.mean(total_speedup)
        print(f"\nOverall average speedup: {avg_speedup:.2f}x")
        print(f"All results match: {all_match}")
        
        if avg_speedup > 1.5:
            print("✅ Significant performance improvement with Cython!")
        elif avg_speedup > 1.0:
            print("⚠️  Modest performance improvement with Cython")
        else:
            print("❌ Cython slower than Python (check implementation)")
    
    print("="*60)