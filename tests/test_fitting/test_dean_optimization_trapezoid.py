import math

import numpy as np
import pytest

from cdsaxs.Fitting.dean_optimization_trapezoid import (
    build_candidate_matrix,
    build_example_trapezoid_model,
    compare_objective_scaling,
    evaluate_batched_objective,
    evaluate_scalar_objective,
    get_objective_batch_profile,
    profile_objective_components,
    run_objective_trial,
    run_optimizer_trial,
)
from cdsaxs.Fitting.Trapezoid_model import TrapezoidModelArray
from cdsaxs.Fitting.Trapezoid_model_dean import TrapezoidModelArrayDean
from cdsaxs.Fitting.Trapezoid_model_dean_gpu import TrapezoidModelArrayDeanGPU, TrapezoidModelArrayDeanGPUFused

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


def test_build_example_trapezoid_model_loads_notebook_style_data():
    model = build_example_trapezoid_model()

    assert model.geometry == "trapezoid"
    assert model.model == "single_material"
    assert model.Intensity.ndim == 2
    assert model.Qx.shape == model.Intensity.shape
    assert model.Qz.shape == model.Intensity.shape
    assert model.numberpoints > 0


def test_scalar_and_batched_objective_paths_return_finite_values():
    model = build_example_trapezoid_model()

    scalar_value = evaluate_scalar_objective(model)
    assert math.isfinite(scalar_value)

    candidates = build_candidate_matrix(model, batch_size=5)
    batched_values = evaluate_batched_objective(model, candidates=candidates)

    assert batched_values.shape == (5,)
    assert np.all(np.isfinite(batched_values))


def test_objective_trial_collects_timing_metadata():
    metrics = run_objective_trial(batch_size=4, repeats=1)

    assert metrics["batch_size"] == 4
    assert metrics["repeats"] == 1
    assert metrics["scalar_elapsed_seconds"] > 0
    assert metrics["batched_elapsed_seconds"] > 0
    assert math.isfinite(metrics["scalar_value"])
    assert metrics["batched_values"].shape == (4,)


def test_dean_candidate_matches_baseline_objective_values():
    baseline = build_example_trapezoid_model(model_cls=TrapezoidModelArray)
    candidate = build_example_trapezoid_model(model_cls=TrapezoidModelArrayDean)

    baseline_scalar = evaluate_scalar_objective(baseline)
    candidate_scalar = evaluate_scalar_objective(candidate)
    assert np.isclose(baseline_scalar, candidate_scalar, rtol=1e-10, atol=1e-10)

    candidates = build_candidate_matrix(baseline, batch_size=4)
    baseline_batch = evaluate_batched_objective(baseline, candidates=candidates)
    candidate_batch = evaluate_batched_objective(candidate, candidates=candidates)
    assert np.allclose(baseline_batch, candidate_batch, rtol=1e-10, atol=1e-10)


def test_compare_objective_scaling_smoke_profile_runs():
    comparison = compare_objective_scaling(
        batch_sizes=get_objective_batch_profile("smoke"),
        repeats=2,
        warmups=1,
    )

    assert comparison["baseline"]["model_class"] == "TrapezoidModelArray"
    assert comparison["candidate"]["model_class"] == "TrapezoidModelArrayDean"
    assert len(comparison["speedups"]) == len(get_objective_batch_profile("smoke"))

    for row in comparison["speedups"]:
        assert row["baseline_seconds_per_call"] > 0
        assert row["candidate_seconds_per_call"] > 0
        assert row["speedup"] > 0


def test_lightweight_optimizer_trial_runs_for_scalar_and_vectorized_modes():
    scalar_run = run_optimizer_trial(vectorized=False, maxiter=1, popsize=4, tol=0.5, polish=False)
    vectorized_run = run_optimizer_trial(vectorized=True, maxiter=1, popsize=4, tol=0.5, polish=False)

    assert scalar_run["result"] is not None
    assert vectorized_run["result"] is not None
    assert scalar_run["elapsed_seconds"] > 0
    assert vectorized_run["elapsed_seconds"] > 0
    assert math.isfinite(scalar_run["gf"])
    assert math.isfinite(vectorized_run["gf"])


def test_profile_objective_components_reports_expected_sections():
    profile = profile_objective_components(model_cls=TrapezoidModelArrayDean, batch_size=16)

    assert profile["batch_size"] == 16
    assert profile["total_elapsed_seconds"] > 0
    names = {row["name"] for row in profile["components"]}
    assert names == {"SymCoordAssign", "FreeFormTrapezoid", "GF_calc"}


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
@pytest.mark.parametrize("gpu_cls", [TrapezoidModelArrayDeanGPU, TrapezoidModelArrayDeanGPUFused])
def test_dean_gpu_candidate_matches_baseline_objective_values(gpu_cls):
    baseline = build_example_trapezoid_model(model_cls=TrapezoidModelArray)
    gpu = build_example_trapezoid_model(model_cls=gpu_cls, freeform_use_cupy=True)

    baseline_scalar = evaluate_scalar_objective(baseline)
    gpu_scalar = evaluate_scalar_objective(gpu)
    assert np.isclose(baseline_scalar, gpu_scalar, rtol=1e-10, atol=1e-10)

    candidates = build_candidate_matrix(baseline, batch_size=8)
    baseline_batch = evaluate_batched_objective(baseline, candidates=candidates)
    gpu_batch = evaluate_batched_objective(gpu, candidates=candidates)
    assert np.allclose(baseline_batch, gpu_batch, rtol=1e-10, atol=1e-10)
