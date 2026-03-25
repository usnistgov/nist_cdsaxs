import contextlib
import io
import math

import numpy as np

from cdsaxs.Fitting.Trapezoid_model import TrapezoidModelArray
from cdsaxs.Fitting.Trapezoid_model_dean_gpu import TrapezoidModelArrayDeanGPUFused
from cdsaxs.Fitting.dean_optimization_trapezoid import build_example_trapezoid_model
from cdsaxs.Fitting.dean_optimization_trapezoid_realistic import (
    build_realistic_trapezoid_model,
    run_realistic_batch_initialize_smoke,
    run_realistic_dual_annealing_trial,
    run_realistic_mcmc_profile,
    run_realistic_sweep_replay_smoke,
)


@contextlib.contextmanager
def _suppress_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        yield


def test_realistic_layer_growth_preserves_gpu_class():
    model = build_realistic_trapezoid_model(
        profile="notebook_layer10_8param",
        model_cls=TrapezoidModelArrayDeanGPUFused,
    )

    assert isinstance(model, TrapezoidModelArrayDeanGPUFused)
    assert model.layers == 2
    assert len(model.model_params["slds"]) == model.layers


def test_parameter_sweep_2d_compatibility_entry_point_runs():
    model = build_example_trapezoid_model(model_cls=TrapezoidModelArray)

    with _suppress_output():
        results = model.parameter_sweep_2d(
            ("trap_0_width", "DW"),
            ((390.0, 400.0), (14.0, 15.0)),
            (2, 2),
            plot_results=False,
            verbose=False,
            optimization_kwargs={"maxiter": 1, "popsize": 2, "polish": False},
        )

    assert results["gf_matrix"].shape == (2, 2)
    assert results["bic_matrix"].shape == (2, 2)
    assert results["convergence_matrix"].shape == (2, 2)
    assert np.isfinite(results["gf_matrix"]).any()


def test_dual_annealing_workflow_smoke_runs():
    trial = run_realistic_dual_annealing_trial(
        profile="notebook_4param",
        model_cls=TrapezoidModelArray,
        maxiter=2,
        repeats=1,
    )

    row = trial["results"][0]
    assert trial["workflow"]["optimizer"] == "dual_annealing"
    assert row["elapsed_seconds"] > 0
    assert math.isfinite(row["gf"])
    assert math.isfinite(row["bic"])


def test_batch_initialize_smoke_returns_expected_schema():
    trial = run_realistic_batch_initialize_smoke(
        profile="notebook_4param",
        model_cls=TrapezoidModelArray,
        max_fits=1,
    )

    row = trial["results"][0]
    assert trial["family"] == "batch_initialize_and_fit_smoke"
    assert trial["workflow"]["workflow_api"] == "batch_initialize_and_fit"
    assert row["elapsed_seconds"] > 0
    assert row["n_total_fits"] == 1
    assert row["n_recorded_fits"] == 1
    assert row["n_converged_fits"] == 1
    assert math.isfinite(row["best_gf"])
    assert math.isfinite(row["best_bic"])


def test_realistic_mcmc_profile_smoke_returns_expected_metrics():
    trial = run_realistic_mcmc_profile(
        profile="notebook_4param",
        family="mcmc_smoke",
        model_cls=TrapezoidModelArray,
    )

    row = trial["results"][0]
    assert trial["workflow"]["workflow_api"] == "CDSAXS_MCMC"
    assert row["elapsed_seconds"] > 0
    assert row["effective_samples"] > 0
    assert row["mean_acceptance"] > 0
    assert math.isfinite(row["gf"])
    assert math.isfinite(row["bic"])


def test_sweep_replay_smoke_applies_best_parameters():
    trial = run_realistic_sweep_replay_smoke(
        profile="notebook_4param",
        model_cls=TrapezoidModelArray,
        sweep_kind="1d",
        run_optimization=False,
    )

    row = trial["results"][0]
    assert trial["workflow"]["workflow_api"] == "show_best_fit_results"
    assert row["elapsed_seconds"] > 0
    assert row["sweep_kind"] == "1d"
    assert math.isfinite(row["gf"])
    assert math.isfinite(row["bic"])


def test_stateful_sweep_then_mcmc_sequence_keeps_parameter_resolution_consistent():
    model = build_example_trapezoid_model(model_cls=TrapezoidModelArray)

    with _suppress_output():
        model.parameter_sweep_1d(
            sweep_param="trap_0_height",
            sweep_range=(690.0, 700.0),
            n_points=2,
            plot_results=False,
            verbose=False,
            optimization_kwargs={"maxiter": 1, "popsize": 2, "polish": False},
        )
        model.parameter_sweep_2d(
            ("trap_0_width", "DW"),
            ((390.0, 400.0), (14.0, 15.0)),
            (2, 2),
            plot_results=False,
            verbose=False,
            optimization_kwargs={"maxiter": 1, "popsize": 2, "polish": False},
        )
        results = model.CDSAXS_MCMC(
            n_walkers=10,
            n_steps=20,
            burn_in=5,
            thin=2,
            progress=False,
            plot_results=False,
            verbose=False,
        )

    assert results is not None
    assert results["mean_acceptance"] > 0
    assert math.isfinite(model.GF)
    assert math.isfinite(model.BIC)
