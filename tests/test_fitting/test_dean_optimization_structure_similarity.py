import pytest

from cdsaxs.Fitting.dean_optimization_trapezoid import run_structure_similarity_trial
from cdsaxs.Fitting.Trapezoid_model_dean import TrapezoidModelArrayDean
from cdsaxs.Fitting.Trapezoid_model_dean_gpu import TrapezoidModelArrayDeanGPUFused

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
def test_trapezoid_structure_similarity_regression_candidate():
    result = run_structure_similarity_trial(
        baseline_cls=TrapezoidModelArrayDean,
        candidate_cls=TrapezoidModelArrayDeanGPUFused,
        baseline_freeform_use_cupy=False,
        candidate_freeform_use_cupy=True,
        optimizer_kwargs={
            "maxiter": 2,
            "popsize": 8,
            "tol": 0.5,
            "polish": False,
            "seed": 1234,
            "vectorized": True,
        },
    )

    assert result["candidate_elapsed_seconds"] < result["baseline_elapsed_seconds"]
    assert result["candidate"]["gf"] == pytest.approx(result["baseline"]["gf"])
    assert result["candidate"]["bic"] == pytest.approx(result["baseline"]["bic"])
    assert result["candidate"]["widths"] == pytest.approx(result["baseline"]["widths"])
    assert result["candidate"]["heights"] == pytest.approx(result["baseline"]["heights"])
