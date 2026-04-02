import copy
import math

import numpy as np

from cdsaxs.Fitting.SiGe_model import SiGeModelArray
from cdsaxs.Fitting.SiGe_model_vectorized import SiGeModelArray_vectorized
from cdsaxs.Fitting.optimization_sige_realistic import (
    build_realistic_sige_candidate_matrix,
    build_realistic_sige_model,
    evaluate_realistic_sige_scalar_objective,
    get_sige_parameter_names_and_defaults,
)


def test_vectorized_sige_wrapper_accepts_scalar_and_batched_shapes():
    model = build_realistic_sige_model(model_cls=SiGeModelArray_vectorized)
    _, defaults = get_sige_parameter_names_and_defaults(model)
    batch = build_realistic_sige_candidate_matrix(model, batch_size=4)

    scalar_value = model._trapezoid_optimization_wrapper(defaults)
    row_value = model._trapezoid_optimization_wrapper(defaults[None, :])
    col_value = model._trapezoid_optimization_wrapper(defaults[:, None])
    batch_sd = model._trapezoid_optimization_wrapper(batch)
    batch_ds = model._trapezoid_optimization_wrapper(batch.T)

    assert math.isfinite(float(scalar_value))
    assert np.asarray(row_value).shape == (1,)
    assert np.asarray(col_value).shape == (1,)
    assert np.asarray(batch_sd).shape == (4,)
    assert np.asarray(batch_ds).shape == (4,)
    assert np.all(np.isfinite(np.asarray(batch_sd)))
    assert np.all(np.isfinite(np.asarray(batch_ds)))


def test_vectorized_sige_matches_scalar_oracle_candidate_by_candidate():
    baseline = build_realistic_sige_model(model_cls=SiGeModelArray)
    candidate = build_realistic_sige_model(model_cls=SiGeModelArray_vectorized)
    batch = build_realistic_sige_candidate_matrix(baseline, batch_size=4)

    baseline_values = np.asarray(
        [evaluate_realistic_sige_scalar_objective(baseline, row) for row in batch],
        dtype=float,
    )
    candidate_values = np.asarray(
        candidate._trapezoid_optimization_wrapper(batch),
        dtype=float,
    )
    candidate_values_transposed = np.asarray(
        candidate._trapezoid_optimization_wrapper(batch.T),
        dtype=float,
    )

    assert np.allclose(candidate_values, baseline_values, rtol=0.0, atol=1e-12)
    assert np.allclose(candidate_values_transposed, baseline_values, rtol=0.0, atol=1e-12)


def test_vectorized_sige_batched_objective_does_not_mutate_model_state():
    model = build_realistic_sige_model(model_cls=SiGeModelArray_vectorized)
    batch = build_realistic_sige_candidate_matrix(model, batch_size=3)

    model_params_before = copy.deepcopy(model.model_params)
    gf_before = float(model.GF)
    bic_before = float(model.BIC)

    values = np.asarray(model._trapezoid_optimization_wrapper(batch), dtype=float)

    assert values.shape == (3,)
    assert np.all(np.isfinite(values))
    assert model.model_params == model_params_before
    assert float(model.GF) == gf_before
    assert float(model.BIC) == bic_before


def test_vectorized_sige_realistic_batch_path_does_not_call_scalar_oracle(monkeypatch):
    model = build_realistic_sige_model(model_cls=SiGeModelArray_vectorized)
    batch = build_realistic_sige_candidate_matrix(model, batch_size=4)

    def _raise_scalar_oracle(*args, **kwargs):
        raise AssertionError("scalar oracle should not be used for realistic 2D batched evaluation")

    monkeypatch.setattr(model, "_evaluate_scalar_oracle", _raise_scalar_oracle)

    values = np.asarray(model._trapezoid_optimization_wrapper(batch), dtype=float)
    assert values.shape == (4,)
    assert np.all(np.isfinite(values))
