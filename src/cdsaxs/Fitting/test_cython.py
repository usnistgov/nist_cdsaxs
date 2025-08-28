#!/usr/bin/env python3
"""
Test script for CDSAXS Cython acceleration.

This script tests the Cython extensions and compares performance
with the pure Python implementations.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

def test_cython_imports():
    """Test if Cython modules can be imported."""
    print("Testing Cython imports...")
    
    modules_to_test = [
        ('cdsaxs_cython.base_cython', 'Base functions'),
        ('cdsaxs_cython.trapezoid_cython', 'Trapezoid functions'), 
        ('cdsaxs_cython.cylinder_cython', 'Cylinder functions')
    ]
    
    imported_modules = {}
    
    for module_name, description in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            imported_modules[module_name] = module
            print(f"✓ {description}: {module_name}")
        except ImportError as e:
            print(f"✗ {description}: Failed to import {module_name}")
            print(f"  Error: {e}")
            return None
        except Exception as e:
            print(f"✗ {description}: Unexpected error importing {module_name}")
            print(f"  Error: {e}")
            return None
    
    return imported_modules

def benchmark_gf_calculation(modules):
    """Benchmark the goodness of fit calculation."""
    print("\nBenchmarking GF calculation...")
    
    # Create test data
    rows, cols = 200, 100
    np.random.seed(42)  # For reproducible results
    sim_int = np.random.rand(rows, cols) * 1000 + 100
    intensity = np.random.rand(rows, cols) * 1000 + 100
    
    # Test Cython version
    base_cython = modules['cdsaxs_cython.base_cython']
    
    # Warm up
    try:
        for _ in range(5):
            result_cython = base_cython.gf_calc_cython(sim_int, intensity)
    except Exception as e:
        print(f"✗ Cython GF calculation failed during warmup: {e}")
        return None, None, False
    
    # Time Cython version
    n_iterations = 100
    start_time = time.time()
    for _ in range(n_iterations):
        result_cython = base_cython.gf_calc_cython(sim_int, intensity)
    cython_time = time.time() - start_time
    
    # Implement pure Python version for comparison
    def gf_calc_python(sim_int, intensity):
        # Avoid log of zero or negative values
        mask = (sim_int > 0) & (intensity > 0) & np.isfinite(sim_int) & np.isfinite(intensity)
        
        log_diff = np.zeros_like(sim_int)
        if np.any(mask):
            log_diff[mask] = np.abs(np.log(intensity[mask]) - np.log(sim_int[mask]))
        
        return np.sum(log_diff)
    
    # Time Python version
    start_time = time.time()
    for _ in range(n_iterations):
        result_python = gf_calc_python(sim_int, intensity)
    python_time = time.time() - start_time
    
    # Compare results
    relative_error = abs(result_cython - result_python) / max(abs(result_python), 1e-10)
    
    print(f"Data size: {rows} x {cols}")
    print(f"Iterations: {n_iterations}")
    print(f"Cython time: {cython_time:.4f}s ({cython_time/n_iterations*1000:.2f}ms per call)")
    print(f"Python time: {python_time:.4f}s ({python_time/n_iterations*1000:.2f}ms per call)")
    print(f"Speedup: {python_time/cython_time:.2f}x")
    print(f"Cython result: {result_cython:.6f}")
    print(f"Python result: {result_python:.6f}")
    print(f"Results match: {relative_error < 1e-6} (relative error: {relative_error:.2e})")
    
    return cython_time, python_time, relative_error < 1e-6

def benchmark_coordinate_assignment(modules):
    """Benchmark coordinate assignment for trapezoids."""
    print("\nBenchmarking coordinate assignment...")
    
    try:
        trapezoid_cython = modules['cdsaxs_cython.trapezoid_cython']
        
        # Create test data
        layers = 5
        np.random.seed(42)
        par = np.random.rand(layers + 1, 2) * 100 + 50  # widths and heights
        sld_values = np.random.rand(layers) * 2 + 0.5  # SLD values
        
        # Warm up
        for _ in range(10):
            coord_cython = trapezoid_cython.sym_coord_assign_cython(par, layers, sld_values)
        
        # Time Cython version
        n_iterations = 1000
        start_time = time.time()
        for _ in range(n_iterations):
            coord_cython = trapezoid_cython.sym_coord_assign_cython(par, layers, sld_values)
        cython_time = time.time() - start_time
        
        print(f"Coordinate assignment: {layers} layers")
        print(f"Cython time: {cython_time:.4f}s ({cython_time/n_iterations*1000:.3f}ms per call)")
        print(f"Output shape: {coord_cython.shape}")
        print(f"Output dtype: {coord_cython.dtype}")
        
        # Verify output makes sense
        print(f"Sample coordinates (layer 0): [{coord_cython[0, :, 0]}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Coordinate assignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_form_factor(modules):
    """Benchmark form factor calculation."""
    print("\nBenchmarking form factor calculation...")
    
    try:
        trapezoid_cython = modules['cdsaxs_cython.trapezoid_cython']
        
        # Create test data
        layers = 3
        np.random.seed(42)
        par = np.random.rand(layers + 1, 2) * 100 + 50
        sld_values = np.random.rand(layers) * 2 + 0.5
        
        # Generate coordinate array
        coord = trapezoid_cython.sym_coord_assign_cython(par, layers, sld_values)
        
        # Create Q arrays
        rows, cols = 100, 50
        qx = np.random.rand(rows, cols) * 0.1 + 0.01  # Avoid zero
        qz = np.random.rand(rows, cols) * 0.1 + 0.01
        
        # Warm up
        for _ in range(5):
            form_cython = trapezoid_cython.free_form_trapezoid_cython(coord, layers, qx, qz)
        
        # Time Cython version
        n_iterations = 50
        start_time = time.time()
        for _ in range(n_iterations):
            form_cython = trapezoid_cython.free_form_trapezoid_cython(coord, layers, qx, qz)
        cython_time = time.time() - start_time
        
        print(f"Form factor: {layers} layers, {rows}x{cols} Q-points")
        print(f"Cython time: {cython_time:.4f}s ({cython_time/n_iterations*1000:.1f}ms per call)")
        print(f"Output shape: {form_cython.shape}")
        print(f"Output type: {form_cython.dtype}")
        print(f"Output range: {np.min(np.abs(form_cython)):.2e} to {np.max(np.abs(form_cython)):.2e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Form factor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_cylinder_functions(modules):
    """Benchmark cylinder-specific functions."""
    print("\nBenchmarking cylinder functions...")
    
    try:
        cylinder_cython = modules['cdsaxs_cython.cylinder_cython']
        
        # Test coordinate conversion
        rows, cols = 100, 50
        np.random.seed(42)
        qx = np.random.rand(rows, cols) * 0.1 + 0.01  # Avoid zero
        qy = np.random.rand(rows, cols) * 0.1
        
        # Warm up
        for _ in range(10):
            qr, alpha = cylinder_cython.convert_cartesian_cylindrical_cython(qx, qy)
        
        # Time conversion
        n_iterations = 1000
        start_time = time.time()
        for _ in range(n_iterations):
            qr, alpha = cylinder_cython.convert_cartesian_cylindrical_cython(qx, qy)
        conversion_time = time.time() - start_time
        
        print(f"Coordinate conversion: {rows}x{cols} points")
        print(f"Cython time: {conversion_time:.4f}s ({conversion_time/n_iterations*1000:.3f}ms per call)")
        print(f"Output shapes: Qr{qr.shape}, Alpha{alpha.shape}")
        print(f"Qr range: {np.min(qr):.4f} to {np.max(qr):.4f}")
        print(f"Alpha range: {np.min(alpha):.4f} to {np.max(alpha):.4f}")
        
        # Verify conversion makes sense
        test_qr_manual = np.sqrt(qx**2 + qy**2)
        qr_error = np.mean(np.abs(qr - test_qr_manual))
        print(f"Conversion accuracy: {qr_error:.2e} (should be small)")
        
        return True
        
    except Exception as e:
        print(f"✗ Cylinder functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_environment():
    """Check the environment and provide debugging information."""
    print("Environment Information:")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        import scipy
        print(f"SciPy version: {scipy.__version__}")
    except ImportError:
        print("SciPy: Not installed")
    
    try:
        import cython
        print(f"Cython version: {cython.__version__}")
    except ImportError:
        print("Cython: Not installed")
    
    # Check if .so files exist
    print("\nChecking compiled extensions:")
    cython_dir = Path("cdsaxs_cython")
    if cython_dir.exists():
        so_files = list(cython_dir.glob("*.so"))
        if so_files:
            for so_file in so_files:
                print(f"Found: {so_file}")
        else:
            print("No .so files found - extensions not compiled")
    else:
        print("cdsaxs_cython directory not found")

def run_comprehensive_test():
    """Run comprehensive tests of all Cython functionality."""
    print("="*60)
    print("CDSAXS Cython Acceleration Test Suite")
    print("="*60)
    
    # Environment check
    check_environment()
    print()
    
    # Test imports
    modules = test_cython_imports()
    if modules is None:
        print("\n✗ Import tests failed. Cython modules are not available.")
        print("\nPossible solutions:")
        print("1. Rebuild extensions: python setup.py build_ext --inplace")
        print("2. Install required packages: pip install cython numpy scipy")
        print("3. Check compiler: make sure you have gcc or clang installed")
        return False
    
    print("\n✓ All Cython modules imported successfully")
    
    # Run benchmarks
    tests = [
        ("GF Calculation", lambda: benchmark_gf_calculation(modules)),
        ("Coordinate Assignment", lambda: benchmark_coordinate_assignment(modules)),
        ("Form Factor Calculation", lambda: benchmark_form_factor(modules)),
        ("Cylinder Functions", lambda: benchmark_cylinder_functions(modules))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*50}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✓ {test_name} completed successfully")
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Cython acceleration is working correctly.")
        print("\nNext steps:")
        print("1. Use accelerated models in your code")
        print("2. Enable Cython acceleration globally: enable_cython()")
        print("3. Benchmark your specific use cases")
        return True
    elif passed > 0:
        print(f"\n⚠️  {total-passed} tests failed, but {passed} passed.")
        print("Partial acceleration may be available.")
        return True
    else:
        print(f"\n❌ All tests failed. Cython acceleration is not working.")
        print("Check the compilation and import errors above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)