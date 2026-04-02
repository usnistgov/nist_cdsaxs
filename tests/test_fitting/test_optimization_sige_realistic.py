import math

import numpy as np
import pytest

from cdsaxs.Fitting.SiGe_model import SiGeModelArray
from cdsaxs.Fitting.SiGe_model_vectorized import SiGeModelArray_vectorized
from cdsaxs.Fitting.optimization_sige_realistic import (
    build_realistic_sige_model,
    compare_realistic_sige_de_timing,
    compare_realistic_sige_objective_throughput,
    describe_realistic_sige_profile,
    run_realistic_sige_de_timing,
)


def test_describe_realistic_sige_profile_matches_expected_shape():
    description = describe_realistic_sige_profile()

    assert description["geometry"] == "sige"
    assert description["label"] == "imec_ellipse_stack"
    assert description["design_layer_count"] == 11
    assert description["expanded_trapezoid_count"] == 26
    assert description["layers"] == 25
    assert description["parameter_count"] == 26
    assert description["ellipse_layer_count"] == 3
    assert description["curved_sidewall_presence"] is False
    assert description["intensity_shape"] == (121, 49)
    assert description["default_vector"].shape == (26,)


def test_build_realistic_sige_model_preserves_requested_class():
    model = build_realistic_sige_model(model_cls=SiGeModelArray_vectorized)

    assert isinstance(model, SiGeModelArray_vectorized)
    assert model.layers == 25
    assert len(model.model_params["trapezoids"]) == 26
    assert len(model.model_params["optimization"]) == 26
    assert len(model.model_params["design_trapezoids"]) == 11


def test_realistic_sige_objective_throughput_comparison_runs_and_matches_scalar():
    comparison = compare_realistic_sige_objective_throughput(
        baseline_cls=SiGeModelArray,
        candidate_cls=SiGeModelArray_vectorized,
        batch_sizes=(1, 4),
        repeats=1,
        warmups=0,
    )

    assert comparison["workflow"]["workflow_api"] == "objective_throughput"
    assert len(comparison["results"]) == 2

    for row in comparison["results"]:
        assert row["baseline_elapsed_seconds"] > 0
        assert row["candidate_elapsed_seconds"] > 0
        assert row["baseline_candidates_per_second"] > 0
        assert row["candidate_candidates_per_second"] > 0
        assert row["speedup"] > 0
        assert row["max_abs_gf_diff"] == pytest.approx(0.0, abs=1e-12)


def test_realistic_sige_vectorized_de_smoke_improves_from_initial_state():
    trial = run_realistic_sige_de_timing(
        model_cls=SiGeModelArray_vectorized,
        vectorized=True,
        maxiter=1,
        popsize=2,
        tol=0.5,
        polish=False,
        seed=1234,
    )

    assert trial["vectorized"] is True
    assert trial["elapsed_seconds"] > 0
    assert math.isfinite(trial["gf"])
    assert math.isfinite(trial["bic"])
    assert abs(trial["objective_delta"]) > 1e-9


def test_realistic_sige_vectorized_de_matches_scalar_de_physics():
    comparison = compare_realistic_sige_de_timing(
        baseline_cls=SiGeModelArray,
        candidate_cls=SiGeModelArray_vectorized,
        maxiter=1,
        popsize=2,
        tol=0.5,
        polish=False,
        seed=1234,
    )

    assert comparison["workflow"]["optimizer"] == "differential_evolution"
    assert comparison["baseline"]["elapsed_seconds"] > 0
    assert comparison["candidate"]["elapsed_seconds"] > 0
    assert np.isfinite(comparison["baseline"]["gf"])
    assert np.isfinite(comparison["candidate"]["gf"])
    assert comparison["absolute_gf_difference"] == pytest.approx(0.0, abs=1e-9)
