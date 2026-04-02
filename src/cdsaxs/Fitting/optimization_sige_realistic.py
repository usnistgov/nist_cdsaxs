from __future__ import annotations

import contextlib
import copy
import io
import time
import warnings
from pathlib import Path

import numpy as np

from .SiGe_model import SiGeModelArray
from .SiGe_model_vectorized import SiGeModelArray_vectorized
from .SiGe_model_vectorized_GPU import SiGeModelArray_vectorized_GPU


REALISTIC_SIGE_DATA_FILENAME = "SMI_imec_nsh_d24b_D260126_T104543_reduced_results.csv"
REALISTIC_SIGE_DATA_PATH = Path("/resdata/DeLongchamp/scratchdisk/nist_cdsaxs_sige") / REALISTIC_SIGE_DATA_FILENAME

REALISTIC_SIGE_MODEL_PARAMS = {
    "layers": 10,
    "trapezoids": [
        {"width": 1939.0282, "width_bounds": (1900, 1960), "height": 61.8643, "height_bounds": (50, 90), "twidth": None},
        {"width": 1734.1772, "width_bounds": (1700, 1760), "height": 194.6538, "height_bounds": (150, 250), "twidth": None},
        {"width": 1587.8525, "width_bounds": (1550, 1620), "height": 873.8879, "height_bounds": (800, 900), "twidth": None},
        {"width": 1396.7978, "width_bounds": (1350, 1440), "height": 94.0138, "height_bounds": (80, 120), "twidth": 1396.7978},
        {
            "width": 800.0,
            "width_bounds": (500, 1390),
            "height": 80,
            "height_bounds": (75, 95),
            "twidth": None,
            "Layer_Type": "Ellipse",
            "depth": 105,
            "depth_bounds": (5, 150),
            "num_layers": 6,
        },
        {"width": 1383.5896, "width_bounds": (1350, 1420), "height": 97.0890, "height_bounds": (80, 120), "twidth": None},
        {
            "width": 800.0,
            "width_bounds": (500, 1390),
            "height": 80,
            "height_bounds": (75, 95),
            "twidth": None,
            "Layer_Type": "Ellipse",
            "depth": 105,
            "depth_bounds": (5, 150),
            "num_layers": 6,
        },
        {"width": 1362.9142, "width_bounds": (1320, 1380), "height": 105.9347, "height_bounds": (90, 120), "twidth": None},
        {
            "width": 800.0,
            "width_bounds": (500, 1390),
            "height": 80,
            "height_bounds": (75, 95),
            "twidth": None,
            "Layer_Type": "Ellipse",
            "depth": 105,
            "depth_bounds": (5, 150),
            "num_layers": 6,
        },
        {"width": 1319.6926, "width_bounds": (1300, 1340), "height": 51.1966, "height_bounds": (40, 70), "twidth": None},
        {"width": 1335.9318, "width_bounds": (1300, 1370), "height": 62.8201, "height_bounds": (50, 90), "twidth": 1296.12, "twidth_bounds": (1000, 1300)},
    ],
    "DW": 10.58,
    "I0": 0.0000004,
    "Bk": 0.5,
    "slds": [1, 1, 1, 1, 1.28, 1, 1.28, 1, 1.28, 1, 1, 1],
    "constraints": [
        {"lhs": "trap_6_depth", "op": "==", "rhs": "trap_4_depth"},
        {"lhs": "trap_8_depth", "op": "==", "rhs": "trap_4_depth"},
        {"lhs": "trap_6_height", "op": "==", "rhs": "trap_4_height"},
        {"lhs": "trap_8_height", "op": "==", "rhs": "trap_4_height"},
        {"lhs": "trap_4_width", "op": "<=", "rhs": "trap_5_width"},
        {"lhs": "trap_6_width", "op": "<=", "rhs": "trap_7_width"},
        {"lhs": "trap_8_width", "op": "<=", "rhs": "trap_9_width"},
        {"lhs": "trap_3_twidth", "op": "==", "rhs": "trap_3_width", "offset": -10.0},
        {"lhs": "trap_5_twidth", "op": "==", "rhs": "trap_5_width", "offset": -10.0},
        {"lhs": "trap_7_twidth", "op": "==", "rhs": "trap_7_width", "offset": -10.0},
    ],
}


@contextlib.contextmanager
def _suppress_workflow_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield


def get_realistic_sige_data_path(data_path: str | Path | None = None) -> Path:
    """Resolve the realistic imec SiGe reduced dataset path."""
    if data_path is not None:
        return Path(data_path).expanduser().resolve()
    return REALISTIC_SIGE_DATA_PATH


def get_realistic_sige_model_params() -> dict:
    """Return a deep copy of the realistic imec SiGe model params."""
    return copy.deepcopy(REALISTIC_SIGE_MODEL_PARAMS)


def build_realistic_sige_model(
    *,
    data_path: str | Path | None = None,
    model_cls=SiGeModelArray,
    freeform_use_cupy: bool = False,
):
    """Build the realistic imec SiGe workflow through the public package API."""
    model = model_cls(
        model="single_material",
        layers=REALISTIC_SIGE_MODEL_PARAMS["layers"],
        model_params=get_realistic_sige_model_params(),
    )
    model.importCDSAXS_GUI(str(get_realistic_sige_data_path(data_path)))
    model.initialize_optimization_params()
    model._freeform_use_cupy = bool(freeform_use_cupy)
    return model


def get_sige_parameter_names_and_defaults(model) -> tuple[list[str], np.ndarray]:
    """Return the optimization parameter order and default vector for the realistic SiGe surface."""
    optimization = model.model_params["optimization"]
    param_names = list(optimization.keys())
    defaults = np.array([optimization[name]["default"] for name in param_names], dtype=float)
    return param_names, defaults


def build_realistic_sige_candidate_matrix(
    model,
    *,
    batch_size: int = 8,
    amplitude: float = 0.02,
) -> np.ndarray:
    """Build a deterministic batch of realistic SiGe candidates near the default vector."""
    param_names, defaults = get_sige_parameter_names_and_defaults(model)
    batch_size = max(1, int(batch_size))
    candidates = np.tile(defaults, (batch_size, 1))
    phases = np.linspace(-1.0, 1.0, batch_size, dtype=float)

    for idx, param_name in enumerate(param_names):
        bounds = model.model_params["optimization"][param_name]
        lower = float(bounds["min"])
        upper = float(bounds["max"])
        span = upper - lower
        offset = phases * span * amplitude * (idx + 1) / len(param_names)
        candidates[:, idx] = np.clip(candidates[:, idx] + offset, lower, upper)

    return candidates


def evaluate_realistic_sige_scalar_objective(model, vector: np.ndarray | None = None) -> float:
    """Evaluate the realistic SiGe objective for one candidate."""
    param_names, defaults = get_sige_parameter_names_and_defaults(model)
    model.param_names = param_names
    candidate = np.asarray(vector if vector is not None else defaults, dtype=float)
    value = model._trapezoid_optimization_wrapper(candidate)
    return float(value)


def evaluate_realistic_sige_batched_objective(
    model,
    *,
    candidates: np.ndarray | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Evaluate the realistic SiGe objective for a batch of candidates."""
    param_names, _ = get_sige_parameter_names_and_defaults(model)
    model.param_names = param_names
    batch = candidates if candidates is not None else build_realistic_sige_candidate_matrix(model, batch_size=batch_size)
    values = model._trapezoid_optimization_wrapper(np.asarray(batch, dtype=float))
    return np.asarray(values, dtype=float)


def describe_realistic_sige_profile(
    *,
    data_path: str | Path | None = None,
    model_cls=SiGeModelArray,
    freeform_use_cupy: bool = False,
) -> dict:
    """Return a compact description of the realistic imec SiGe optimization surface."""
    model = build_realistic_sige_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    param_names, defaults = get_sige_parameter_names_and_defaults(model)

    return {
        "geometry": "sige",
        "label": "imec_ellipse_stack",
        "model_class": model.__class__.__name__,
        "design_layer_count": int(len(model.model_params.get("design_trapezoids", model.model_params["trapezoids"]))),
        "expanded_trapezoid_count": int(len(model.model_params["trapezoids"])),
        "layers": int(model.layers),
        "parameter_count": int(len(param_names)),
        "parameter_names": tuple(param_names),
        "default_vector": np.asarray(defaults, dtype=float),
        "typed_layer_presence": True,
        "ellipse_layer_count": int(
            sum(
                1
                for trap in model.model_params.get("design_trapezoids", [])
                if str(trap.get("Layer_Type", "")).strip().lower() == "ellipse"
            )
        ),
        "curved_sidewall_presence": bool(
            any(
                str(trap.get("Layer_Type", "")).strip().lower() == "curved_sides"
                for trap in model.model_params.get("design_trapezoids", [])
            )
        ),
        "intensity_shape": tuple(int(v) for v in np.asarray(model.Intensity).shape),
        "data_path": str(get_realistic_sige_data_path(data_path)),
    }


def compare_realistic_sige_objective_throughput(
    *,
    baseline_cls=SiGeModelArray,
    candidate_cls=SiGeModelArray_vectorized,
    baseline_freeform_use_cupy: bool = False,
    candidate_freeform_use_cupy: bool = False,
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16, 32),
    repeats: int = 1,
    warmups: int = 0,
    amplitude: float = 0.02,
    data_path: str | Path | None = None,
) -> dict:
    """
    Compare realistic SiGe scalar objective throughput against one vectorized candidate.

    Baseline work is always measured candidate-by-candidate with the incumbent scalar
    SiGe objective. Candidate work is measured as one batched call per batch.
    """
    baseline = build_realistic_sige_model(
        data_path=data_path,
        model_cls=baseline_cls,
        freeform_use_cupy=baseline_freeform_use_cupy,
    )
    candidate = build_realistic_sige_model(
        data_path=data_path,
        model_cls=candidate_cls,
        freeform_use_cupy=candidate_freeform_use_cupy,
    )
    param_names, _ = get_sige_parameter_names_and_defaults(baseline)

    workflow = describe_realistic_sige_profile(
        data_path=data_path,
        model_cls=baseline_cls,
        freeform_use_cupy=baseline_freeform_use_cupy,
    )
    workflow.update(
        {
            "workflow_api": "objective_throughput",
            "baseline_model_class": baseline.__class__.__name__,
            "candidate_model_class": candidate.__class__.__name__,
            "baseline_freeform_use_cupy": bool(baseline_freeform_use_cupy),
            "candidate_freeform_use_cupy": bool(candidate_freeform_use_cupy),
            "batch_sizes": tuple(int(v) for v in batch_sizes),
            "repeats": int(repeats),
            "warmups": int(warmups),
            "candidate_amplitude": float(amplitude),
        }
    )

    results = []
    for batch_size in tuple(int(size) for size in batch_sizes):
        candidates = build_realistic_sige_candidate_matrix(
            baseline,
            batch_size=batch_size,
            amplitude=amplitude,
        )

        baseline_values = None
        for _ in range(max(0, warmups)):
            baseline_values = np.asarray(
                [
                    evaluate_realistic_sige_scalar_objective(baseline, row)
                    for row in candidates
                ],
                dtype=float,
            )
            evaluate_realistic_sige_batched_objective(candidate, candidates=candidates)

        baseline_start = time.perf_counter()
        for _ in range(max(1, repeats)):
            baseline_values = np.asarray(
                [
                    evaluate_realistic_sige_scalar_objective(baseline, row)
                    for row in candidates
                ],
                dtype=float,
            )
        baseline_elapsed = time.perf_counter() - baseline_start

        candidate_start = time.perf_counter()
        candidate_values = None
        for _ in range(max(1, repeats)):
            candidate_values = evaluate_realistic_sige_batched_objective(candidate, candidates=candidates)
        candidate_elapsed = time.perf_counter() - candidate_start

        max_abs_diff = float(np.max(np.abs(candidate_values - baseline_values)))
        total_candidates = int(batch_size * max(1, repeats))

        results.append(
            {
                "batch_size": int(batch_size),
                "total_candidates": total_candidates,
                "baseline_elapsed_seconds": baseline_elapsed,
                "candidate_elapsed_seconds": candidate_elapsed,
                "baseline_candidates_per_second": total_candidates / baseline_elapsed,
                "candidate_candidates_per_second": total_candidates / candidate_elapsed,
                "speedup": baseline_elapsed / candidate_elapsed,
                "max_abs_gf_diff": max_abs_diff,
                "baseline_mean_objective": float(np.mean(baseline_values)),
                "candidate_mean_objective": float(np.mean(candidate_values)),
            }
        )

    return {
        "workflow": workflow,
        "results": results,
    }


def run_realistic_sige_de_timing(
    *,
    model_cls=SiGeModelArray,
    vectorized: bool = False,
    freeform_use_cupy: bool = False,
    data_path: str | Path | None = None,
    maxiter: int = 2,
    popsize: int = 4,
    tol: float = 0.0,
    polish: bool = False,
    seed: int = 1234,
    updating: str = "deferred",
) -> dict:
    """Run one fixed-configuration realistic SiGe DE timing trial."""
    model = build_realistic_sige_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    param_names, _ = get_sige_parameter_names_and_defaults(model)
    initial_gf = float(getattr(model, "GF", np.inf))
    n_params = len(param_names)

    with _suppress_workflow_output():
        start = time.perf_counter()
        result = model.CDSAXS_Optimize(
            optimizer="differential_evolution",
            use_callbacks=False,
            verbose=False,
            vectorized=bool(vectorized),
            plot_results=False,
            plot_structure=False,
            plot_grid=False,
            plot_combined=False,
            maxiter=int(maxiter),
            popsize=int(popsize),
            tol=float(tol),
            polish=bool(polish),
            seed=int(seed),
            updating=str(updating),
            freeform_use_cupy=bool(freeform_use_cupy),
        )
        elapsed = time.perf_counter() - start

    optimization_result = getattr(model, "optimization_result", None)
    best_fun = None
    if optimization_result is not None and hasattr(optimization_result, "fun"):
        best_fun = float(optimization_result.fun)
    elif hasattr(result, "fun"):
        best_fun = float(result.fun)

    return {
        "model_class": model.__class__.__name__,
        "vectorized": bool(vectorized),
        "freeform_use_cupy": bool(freeform_use_cupy),
        "elapsed_seconds": elapsed,
        "initial_gf": initial_gf,
        "gf": float(getattr(model, "GF", np.inf)),
        "bic": float(getattr(model, "BIC", np.inf)),
        "objective_delta": float(getattr(model, "GF", np.inf) - initial_gf),
        "parameter_count": int(n_params),
        "population_size": int(popsize),
        "candidate_count_hint": int(popsize * n_params),
        "estimated_objective_evaluations": int((maxiter + 1) * popsize * n_params),
        "maxiter": int(maxiter),
        "tol": float(tol),
        "polish": bool(polish),
        "seed": int(seed),
        "updating": str(updating),
        "result": result,
        "optimization_result": optimization_result,
        "best_fun": best_fun,
    }


def compare_realistic_sige_de_timing(
    *,
    baseline_cls=SiGeModelArray,
    candidate_cls=SiGeModelArray_vectorized,
    baseline_freeform_use_cupy: bool = False,
    candidate_freeform_use_cupy: bool = False,
    data_path: str | Path | None = None,
    maxiter: int = 2,
    popsize: int = 4,
    tol: float = 0.0,
    polish: bool = False,
    seed: int = 1234,
    updating: str = "deferred",
) -> dict:
    """Compare incumbent scalar SiGe DE timing against the candidate vectorized path."""
    baseline = run_realistic_sige_de_timing(
        model_cls=baseline_cls,
        vectorized=False,
        freeform_use_cupy=baseline_freeform_use_cupy,
        data_path=data_path,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=polish,
        seed=seed,
        updating=updating,
    )
    candidate = run_realistic_sige_de_timing(
        model_cls=candidate_cls,
        vectorized=True,
        freeform_use_cupy=candidate_freeform_use_cupy,
        data_path=data_path,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=polish,
        seed=seed,
        updating=updating,
    )

    return {
        "workflow": {
            "geometry": "sige",
            "label": "imec_ellipse_stack",
            "optimizer": "differential_evolution",
            "maxiter": int(maxiter),
            "popsize": int(popsize),
            "tol": float(tol),
            "polish": bool(polish),
            "seed": int(seed),
            "updating": str(updating),
            "baseline_freeform_use_cupy": bool(baseline_freeform_use_cupy),
            "candidate_freeform_use_cupy": bool(candidate_freeform_use_cupy),
        },
        "baseline": baseline,
        "candidate": candidate,
        "speedup": baseline["elapsed_seconds"] / candidate["elapsed_seconds"],
        "absolute_gf_difference": float(abs(candidate["gf"] - baseline["gf"])),
    }


def compare_realistic_sige_gpu_objective_throughput(
    *,
    baseline_cls=SiGeModelArray_vectorized,
    candidate_cls=SiGeModelArray_vectorized_GPU,
    batch_sizes: tuple[int, ...] = (8, 16, 32),
    repeats: int = 1,
    warmups: int = 0,
    amplitude: float = 0.02,
    data_path: str | Path | None = None,
) -> dict:
    """Compare CPU-vectorized SiGe throughput against the GPU vectorized path."""
    return compare_realistic_sige_objective_throughput(
        baseline_cls=baseline_cls,
        candidate_cls=candidate_cls,
        baseline_freeform_use_cupy=False,
        candidate_freeform_use_cupy=True,
        batch_sizes=batch_sizes,
        repeats=repeats,
        warmups=warmups,
        amplitude=amplitude,
        data_path=data_path,
    )


def compare_realistic_sige_gpu_de_timing(
    *,
    baseline_cls=SiGeModelArray_vectorized,
    candidate_cls=SiGeModelArray_vectorized_GPU,
    data_path: str | Path | None = None,
    maxiter: int = 2,
    popsize: int = 4,
    tol: float = 0.0,
    polish: bool = False,
    seed: int = 1234,
    updating: str = "deferred",
) -> dict:
    """Compare CPU-vectorized SiGe DE timing against the GPU vectorized path."""
    return compare_realistic_sige_de_timing(
        baseline_cls=baseline_cls,
        candidate_cls=candidate_cls,
        baseline_freeform_use_cupy=False,
        candidate_freeform_use_cupy=True,
        data_path=data_path,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=polish,
        seed=seed,
        updating=updating,
    )
