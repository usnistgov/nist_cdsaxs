import math

import numpy as np
import pytest

from cdsaxs.Fitting.Trapezoid_model import TrapezoidModelArray
from cdsaxs.Fitting.Trapezoid_model_dean_gpu import TrapezoidModelArrayDeanGPUFused
from cdsaxs.Fitting.dean_optimization_trapezoid_realistic import (
    build_realistic_trapezoid_model,
    describe_realistic_trapezoid_profile,
    get_realistic_population_sizes,
    get_realistic_trapezoid_profile_spec,
    run_realistic_candidate_budget_trial,
    run_realistic_optimizer_profile,
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


@pytest.mark.parametrize(
    ("profile", "expected_params", "expected_layers", "expected_names"),
    [
        (
            "notebook_4param",
            4,
            1,
            ("trap_0_width", "trap_0_height", "trap_1_width", "DW"),
        ),
        (
            "notebook_layer10_8param",
            8,
            2,
            (
                "trap_0_width",
                "trap_0_height",
                "trap_1_width",
                "trap_1_height",
                "trap_2_width",
                "DW",
                "I0",
                "Bk",
            ),
        ),
        (
            "notebook_layer90_8param",
            8,
            2,
            (
                "trap_0_width",
                "trap_0_height",
                "trap_1_width",
                "trap_1_height",
                "trap_2_width",
                "DW",
                "I0",
                "Bk",
            ),
        ),
        (
            "notebook_multilayer_12param",
            12,
            4,
            (
                "trap_0_width",
                "trap_0_height",
                "trap_1_width",
                "trap_1_height",
                "trap_2_width",
                "trap_2_height",
                "trap_3_width",
                "trap_3_height",
                "trap_4_width",
                "DW",
                "I0",
                "Bk",
            ),
        ),
    ],
)
def test_describe_realistic_trapezoid_profile_matches_expected_shape(
    profile,
    expected_params,
    expected_layers,
    expected_names,
):
    spec = get_realistic_trapezoid_profile_spec(profile)
    description = describe_realistic_trapezoid_profile(profile=profile)

    assert description["profile"] == profile
    assert description["label"] == spec["label"]
    assert description["parameter_count"] == expected_params
    assert description["layers"] == expected_layers
    assert description["layer_growth_path"]
    assert description["parameter_names"] == expected_names
    assert description["default_vector"].shape == (expected_params,)


def test_build_realistic_trapezoid_model_preserves_requested_class():
    model = build_realistic_trapezoid_model(
        profile="notebook_layer10_8param",
        model_cls=TrapezoidModelArrayDeanGPUFused,
    )

    assert isinstance(model, TrapezoidModelArrayDeanGPUFused)
    assert model.layers == 2
    assert len(model.model_params["optimization"]) == 8
    assert len(model.model_params["slds"]) == model.layers
    assert tuple(model.model_params["optimization"]) == (
        "trap_0_width",
        "trap_0_height",
        "trap_1_width",
        "trap_1_height",
        "trap_2_width",
        "DW",
        "I0",
        "Bk",
    )


def test_realistic_candidate_budget_trial_runs_on_small_population_sizes():
    trial = run_realistic_candidate_budget_trial(
        profile="notebook_layer10_8param",
        model_cls=TrapezoidModelArray,
        population_sizes=(2, 4),
        repeats=1,
        warmups=0,
    )

    assert trial["profile"] == "notebook_layer10_8param"
    assert trial["model_class"] == "TrapezoidModelArray"
    assert trial["workflow"]["workflow_api"] == "objective_batch"
    assert trial["workflow"]["layer_growth_path"] == "add_layer_at_percentage(10)"
    assert len(trial["results"]) == 2

    candidate_counts = [row["candidate_count"] for row in trial["results"]]
    assert candidate_counts == [16, 32]

    for row in trial["results"]:
        assert row["elapsed_seconds"] > 0
        assert row["seconds_per_call"] > 0
        assert row["seconds_per_candidate"] > 0
        assert row["candidates_per_second"] > 0
        assert math.isfinite(row["mean_objective"])


def test_realistic_optimizer_profile_runs_on_small_population_sizes():
    trial = run_realistic_optimizer_profile(
        profile="notebook_4param",
        family="one_generation",
        model_cls=TrapezoidModelArray,
        population_sizes=(2,),
        repeats=1,
        tol=0.5,
        polish=False,
        seed=1234,
    )

    assert trial["family"] == "one_generation"
    assert trial["model_class"] == "TrapezoidModelArray"
    assert trial["workflow"]["optimizer"] == "differential_evolution"
    assert trial["workflow"]["vectorized"] is True
    assert len(trial["results"]) == 1

    row = trial["results"][0]
    assert row["population_size"] == 2
    assert row["candidate_count_hint"] == 8
    assert row["maxiter"] == 1
    assert row["seconds_per_run"] > 0
    assert math.isfinite(row["gf"])
    assert math.isfinite(row["bic"])


def test_realistic_optimizer_profile_runs_on_layered_profile():
    trial = run_realistic_optimizer_profile(
        profile="notebook_layer10_8param",
        family="one_generation",
        model_cls=TrapezoidModelArray,
        population_sizes=(2,),
        repeats=1,
        tol=0.5,
        polish=False,
        seed=1234,
    )

    row = trial["results"][0]
    assert trial["workflow"]["height_percentage"] == 10
    assert row["population_size"] == 2
    assert row["candidate_count_hint"] == 16
    assert math.isfinite(row["gf"])
    assert math.isfinite(row["bic"])


def test_get_realistic_population_sizes_returns_expected_ladders():
    assert get_realistic_population_sizes("notebook_4param", family="candidate_budget") == (128, 256, 512, 1024)
    assert get_realistic_population_sizes("notebook_layer10_8param", family="one_generation") == (32, 64, 128)
    assert get_realistic_population_sizes("notebook_multilayer_12param", family="multi_generation") == (32, 64)


@pytest.mark.skipif(not _gpu_available(), reason="Requires CuPy with a visible CUDA GPU.")
def test_realistic_candidate_budget_trial_runs_on_fused_gpu_path():
    trial = run_realistic_candidate_budget_trial(
        profile="notebook_4param",
        model_cls=TrapezoidModelArrayDeanGPUFused,
        population_sizes=(4,),
        repeats=1,
        warmups=0,
        freeform_use_cupy=True,
    )

    row = trial["results"][0]
    assert row["candidate_count"] == 16
    assert row["elapsed_seconds"] > 0
    assert np.isfinite(row["mean_objective"])
