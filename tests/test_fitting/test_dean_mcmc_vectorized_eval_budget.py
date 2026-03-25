import math

import pytest

from cdsaxs.Fitting.Trapezoid_model import TrapezoidModelArray
from cdsaxs.Fitting.Trapezoid_model_dean_gpu import TrapezoidModelArrayDeanGPUFused
from cdsaxs.Fitting.dean_optimization_trapezoid_realistic import (
    build_realistic_trapezoid_model,
    run_realistic_mcmc_profile,
    run_realistic_vectorized_mcmc_eval_budget_profile,
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


def test_vectorized_mcmc_eval_budget_smoke_returns_expected_metrics():
    trial = run_realistic_vectorized_mcmc_eval_budget_profile(
        profile="notebook_4param",
        native_family="mcmc_smoke",
        model_cls=TrapezoidModelArray,
        walker_batch_sizes=(8, 16),
        repeats=1,
        warmups=0,
        candidate_seed=1234,
    )

    assert trial["family"] == "mcmc_vectorized_eval_budget"
    assert trial["workflow"]["workflow_api"] == "vectorized_mcmc_eval_budget"
    assert trial["workflow"]["target_gross_evaluations"] == 200
    assert [row["walker_batch_size"] for row in trial["results"]] == [8, 16]

    for row in trial["results"]:
        assert row["target_gross_evaluations"] == 200
        assert row["realized_gross_evaluations"] == 200
        assert row["batches_per_run"] > 0
        assert row["elapsed_seconds"] > 0
        assert row["seconds_per_run"] > 0
        assert row["seconds_per_evaluation"] > 0
        assert row["evaluations_per_second"] > 0
        assert math.isfinite(row["best_objective"])
        assert math.isfinite(row["mean_objective"])
        assert math.isfinite(row["best_gf"])
        assert math.isfinite(row["best_bic"])
        assert row["best_params"].shape == (trial["workflow"]["parameter_count"],)


def test_vectorized_mcmc_eval_budget_chunking_preserves_best_result():
    trial = run_realistic_vectorized_mcmc_eval_budget_profile(
        profile="notebook_4param",
        native_family="mcmc_smoke",
        model_cls=TrapezoidModelArray,
        walker_batch_sizes=(8, 25),
        repeats=1,
        warmups=0,
        candidate_seed=1234,
    )

    row_a, row_b = trial["results"]
    assert row_a["target_gross_evaluations"] == row_b["target_gross_evaluations"]
    assert row_a["realized_gross_evaluations"] == row_b["realized_gross_evaluations"]
    assert row_a["best_objective"] == pytest.approx(row_b["best_objective"], rel=1e-12, abs=1e-12)
    assert row_a["best_gf"] == pytest.approx(row_b["best_gf"], rel=1e-12, abs=1e-12)
    assert row_a["best_bic"] == pytest.approx(row_b["best_bic"], rel=1e-12, abs=1e-12)
    assert row_a["best_params"] == pytest.approx(row_b["best_params"], rel=1e-12, abs=1e-12)


def test_realistic_vectorized_mcmc_wrapper_matches_native_schema_and_solution():
    kwargs = {
        "n_walkers": 10,
        "n_steps": 20,
        "burn_in": 5,
        "thin": 2,
        "progress": False,
        "plot_results": False,
        "verbose": False,
        "seed": 1234,
    }

    baseline_model = build_realistic_trapezoid_model(
        profile="notebook_4param",
        model_cls=TrapezoidModelArray,
    )
    candidate_model = build_realistic_trapezoid_model(
        profile="notebook_4param",
        model_cls=TrapezoidModelArray,
    )

    baseline = baseline_model.CDSAXS_MCMC(**kwargs)
    candidate = candidate_model.CDSAXS_MCMC_GPU(**kwargs)

    assert candidate is not None
    assert candidate["chains"].shape == baseline["chains"].shape
    assert candidate["flat_chains"].shape == baseline["flat_chains"].shape
    assert candidate["param_names"] == baseline["param_names"]
    assert candidate["effective_samples"] == baseline["effective_samples"]
    assert candidate["mean_acceptance"] == pytest.approx(baseline["mean_acceptance"])
    assert candidate["best_log_prob"] == pytest.approx(baseline["best_log_prob"])
    assert candidate["best_params"] == pytest.approx(baseline["best_params"])
    assert candidate_model.GF == pytest.approx(baseline_model.GF)
    assert candidate_model.BIC == pytest.approx(baseline_model.BIC)


def test_realistic_mcmc_respects_burn_in_when_thin_is_one():
    kwargs = {
        "n_walkers": 10,
        "n_steps": 20,
        "burn_in": 5,
        "thin": 1,
        "progress": False,
        "plot_results": False,
        "verbose": False,
        "seed": 1234,
    }
    expected_steps = kwargs["n_steps"] - kwargs["burn_in"]
    expected_samples = expected_steps * kwargs["n_walkers"]

    baseline_model = build_realistic_trapezoid_model(
        profile="notebook_4param",
        model_cls=TrapezoidModelArray,
    )
    candidate_model = build_realistic_trapezoid_model(
        profile="notebook_4param",
        model_cls=TrapezoidModelArray,
    )

    baseline = baseline_model.CDSAXS_MCMC(**kwargs)
    candidate = candidate_model.CDSAXS_MCMC_GPU(**kwargs)

    assert baseline["chains_burned"].shape[0] == expected_steps
    assert baseline["chains_final"].shape[0] == expected_steps
    assert baseline["log_prob_final"].shape[0] == expected_steps
    assert baseline["flat_chains"].shape[0] == expected_samples
    assert baseline["effective_samples"] == expected_samples

    assert candidate["chains_burned"].shape[0] == expected_steps
    assert candidate["chains_final"].shape[0] == expected_steps
    assert candidate["log_prob_final"].shape[0] == expected_steps
    assert candidate["flat_chains"].shape[0] == expected_samples
    assert candidate["effective_samples"] == expected_samples


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_realistic_native_mcmc_gpu_profile_matches_incumbent():
    kwargs = {
        "profile": "notebook_4param",
        "family": "mcmc_smoke",
        "n_walkers": 10,
        "n_steps": 20,
        "burn_in": 5,
        "thin": 2,
        "seed": 1234,
    }

    baseline = run_realistic_mcmc_profile(
        model_cls=TrapezoidModelArray,
        **kwargs,
    )
    candidate = run_realistic_mcmc_profile(
        model_cls=TrapezoidModelArrayDeanGPUFused,
        freeform_use_cupy=True,
        **kwargs,
    )

    base_row = baseline["results"][0]
    candidate_row = candidate["results"][0]
    assert baseline["mcmc_results"] is not None
    assert candidate["mcmc_results"] is not None
    assert candidate_row["effective_samples"] == base_row["effective_samples"]
    assert candidate_row["mean_acceptance"] == pytest.approx(base_row["mean_acceptance"])
    assert candidate_row["gf"] == pytest.approx(base_row["gf"])
    assert candidate_row["bic"] == pytest.approx(base_row["bic"])
    assert candidate["mcmc_results"]["best_log_prob"] == pytest.approx(
        baseline["mcmc_results"]["best_log_prob"]
    )
    assert candidate["mcmc_results"]["best_params"] == pytest.approx(
        baseline["mcmc_results"]["best_params"]
    )


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_realistic_vectorized_mcmc_wrapper_gpu_matches_incumbent():
    kwargs = {
        "n_walkers": 10,
        "n_steps": 20,
        "burn_in": 5,
        "thin": 2,
        "progress": False,
        "plot_results": False,
        "verbose": False,
        "seed": 1234,
    }

    baseline_model = build_realistic_trapezoid_model(
        profile="notebook_4param",
        model_cls=TrapezoidModelArray,
    )
    candidate_model = build_realistic_trapezoid_model(
        profile="notebook_4param",
        model_cls=TrapezoidModelArrayDeanGPUFused,
        freeform_use_cupy=True,
    )

    baseline = baseline_model.CDSAXS_MCMC_GPU(**kwargs)
    candidate = candidate_model.CDSAXS_MCMC_GPU(**kwargs)

    assert baseline is not None
    assert candidate is not None
    assert candidate["chains"].shape == baseline["chains"].shape
    assert candidate["effective_samples"] == baseline["effective_samples"]
    assert candidate["mean_acceptance"] == pytest.approx(baseline["mean_acceptance"])
    assert candidate["best_log_prob"] == pytest.approx(baseline["best_log_prob"])
    assert candidate["best_params"] == pytest.approx(baseline["best_params"], rel=1e-10, abs=1e-10)
    assert candidate_model.GF == pytest.approx(baseline_model.GF, rel=1e-10, abs=1e-10)
    assert candidate_model.BIC == pytest.approx(baseline_model.BIC, rel=1e-10, abs=1e-10)


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_vectorized_mcmc_eval_budget_gpu_matches_incumbent():
    kwargs = {
        "profile": "notebook_4param",
        "native_family": "mcmc_smoke",
        "walker_batch_sizes": (16,),
        "repeats": 1,
        "warmups": 0,
        "candidate_seed": 1234,
    }

    baseline = run_realistic_vectorized_mcmc_eval_budget_profile(
        model_cls=TrapezoidModelArray,
        **kwargs,
    )
    candidate = run_realistic_vectorized_mcmc_eval_budget_profile(
        model_cls=TrapezoidModelArrayDeanGPUFused,
        freeform_use_cupy=True,
        **kwargs,
    )

    base_row = baseline["results"][0]
    candidate_row = candidate["results"][0]
    assert candidate_row["realized_gross_evaluations"] == base_row["realized_gross_evaluations"]
    assert candidate_row["best_objective"] == pytest.approx(
        base_row["best_objective"], rel=1e-10, abs=1e-10
    )
    assert candidate_row["best_gf"] == pytest.approx(base_row["best_gf"], rel=1e-10, abs=1e-10)
    assert candidate_row["best_bic"] == pytest.approx(base_row["best_bic"], rel=1e-10, abs=1e-10)
    assert candidate_row["best_params"] == pytest.approx(
        base_row["best_params"], rel=1e-10, abs=1e-10
    )
