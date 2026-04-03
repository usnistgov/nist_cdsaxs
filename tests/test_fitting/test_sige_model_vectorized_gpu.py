import copy
import math
from pathlib import Path

import numpy as np
import pytest

from cdsaxs.Fitting.SiGe_model_vectorized import SiGeModelArray_vectorized
from cdsaxs.Fitting.SiGe_model_vectorized_GPU import SiGeModelArray_vectorized_GPU
from cdsaxs.Fitting.optimization_sige_realistic import (
    build_realistic_sige_candidate_matrix,
    build_realistic_sige_model,
    compare_realistic_sige_gpu_de_timing,
    compare_realistic_sige_gpu_objective_throughput,
    run_realistic_sige_convergence_suite,
)

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


def _gpu_available():
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_sige_gpu_vectorized_matches_cpu_vectorized_batches():
    baseline = build_realistic_sige_model(model_cls=SiGeModelArray_vectorized)
    gpu = build_realistic_sige_model(
        model_cls=SiGeModelArray_vectorized_GPU,
        freeform_use_cupy=True,
    )
    batch = build_realistic_sige_candidate_matrix(baseline, batch_size=16)

    baseline_values = np.asarray(baseline._trapezoid_optimization_wrapper(batch), dtype=float)
    gpu_values = np.asarray(gpu._trapezoid_optimization_wrapper(batch), dtype=float)

    assert np.all(np.isfinite(gpu_values))
    assert np.allclose(gpu_values, baseline_values, rtol=0.0, atol=1e-8)
    assert gpu._vectorized_last_execution_path == "gpu_resident_4d"
    assert gpu._vectorized_last_gpu_exception is None


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_sige_gpu_vectorized_small_batch_falls_back_to_cpu():
    gpu = build_realistic_sige_model(
        model_cls=SiGeModelArray_vectorized_GPU,
        freeform_use_cupy=True,
    )
    batch = build_realistic_sige_candidate_matrix(gpu, batch_size=4)

    values = np.asarray(gpu._trapezoid_optimization_wrapper(batch), dtype=float)

    assert values.shape == (4,)
    assert np.all(np.isfinite(values))
    assert gpu._vectorized_last_execution_path == "gpu_small_batch_cpu_fallback"
    assert gpu._vectorized_last_gpu_exception is None


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_sige_gpu_vectorized_loop_path_remains_opt_in():
    baseline = build_realistic_sige_model(model_cls=SiGeModelArray_vectorized)
    gpu = build_realistic_sige_model(
        model_cls=SiGeModelArray_vectorized_GPU,
        freeform_use_cupy=True,
    )
    gpu._vectorized_gpu_layer_algorithm = "loop"
    batch = build_realistic_sige_candidate_matrix(baseline, batch_size=16)

    baseline_values = np.asarray(baseline._trapezoid_optimization_wrapper(batch), dtype=float)
    gpu_values = np.asarray(gpu._trapezoid_optimization_wrapper(batch), dtype=float)

    assert np.all(np.isfinite(gpu_values))
    assert np.allclose(gpu_values, baseline_values, rtol=0.0, atol=1e-8)
    assert gpu._vectorized_last_execution_path == "gpu_resident"
    assert gpu._vectorized_last_gpu_exception is None


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_sige_gpu_vectorized_batched_objective_does_not_mutate_model_state():
    gpu = build_realistic_sige_model(
        model_cls=SiGeModelArray_vectorized_GPU,
        freeform_use_cupy=True,
    )
    batch = build_realistic_sige_candidate_matrix(gpu, batch_size=16)

    model_params_before = copy.deepcopy(gpu.model_params)
    gf_before = float(gpu.GF)
    bic_before = float(gpu.BIC)

    values = np.asarray(gpu._trapezoid_optimization_wrapper(batch), dtype=float)

    assert values.shape == (16,)
    assert np.all(np.isfinite(values))
    assert gpu.model_params == model_params_before
    assert float(gpu.GF) == gf_before
    assert float(gpu.BIC) == bic_before


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_sige_gpu_objective_throughput_smoke_reports_parity():
    comparison = compare_realistic_sige_gpu_objective_throughput(
        batch_sizes=(8, 16),
        repeats=1,
        warmups=0,
    )

    assert comparison["workflow"]["candidate_model_class"] == "SiGeModelArray_vectorized_GPU"
    assert len(comparison["results"]) == 2
    for row in comparison["results"]:
        assert row["baseline_elapsed_seconds"] > 0
        assert row["candidate_elapsed_seconds"] > 0
        assert row["speedup"] > 0
        assert row["max_abs_gf_diff"] < 1e-7


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_sige_gpu_de_smoke_matches_cpu_vectorized_physics():
    comparison = compare_realistic_sige_gpu_de_timing(
        maxiter=1,
        popsize=2,
        tol=0.5,
        polish=False,
        seed=1234,
    )

    assert comparison["workflow"]["optimizer"] == "differential_evolution"
    assert comparison["baseline"]["elapsed_seconds"] > 0
    assert comparison["candidate"]["elapsed_seconds"] > 0
    assert math.isfinite(comparison["baseline"]["gf"])
    assert math.isfinite(comparison["candidate"]["gf"])
    assert comparison["absolute_gf_difference"] < 1e-6


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_sige_cpu_vs_gpu_convergence_suite_writes_report(tmp_path):
    report = run_realistic_sige_convergence_suite(
        seeds=(1234,),
        maxiters=(1,),
        popsize=2,
        tol=0.5,
        polish=False,
        output_dir=tmp_path,
        report_name="smoke",
    )

    assert len(report["results"]) == 1
    assert len(report["budget_summaries"]) == 1
    assert report["budget_summaries"][0]["median_speedup"] > 0
    assert Path(report["summary_json_path"]).is_file()
    assert Path(report["results_csv_path"]).is_file()
    assert Path(report["representative"]["fit_plot_path"]).is_file()
    assert Path(report["representative"]["summary_plot_path"]).is_file()
