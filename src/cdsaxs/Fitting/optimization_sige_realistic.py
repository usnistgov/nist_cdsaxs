from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .SiGe_model import SiGeModelArray
from .SiGe_model_vectorized import SiGeModelArray_vectorized
from .SiGe_model_vectorized_GPU import SiGeModelArray_vectorized_GPU


REALISTIC_SIGE_DATA_FILENAME = "SMI_imec_nsh_d24b_D260126_T104543_reduced_results.csv"
REALISTIC_SIGE_DATA_PATH = Path("/resdata/DeLongchamp/scratchdisk/nist_cdsaxs_sige") / REALISTIC_SIGE_DATA_FILENAME
REALISTIC_SIGE_REPORT_ROOT = Path(__file__).resolve().parents[3] / "dev_workspace" / "sige_convergence"

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


def get_realistic_sige_gpu_config(model=None) -> dict:
    """Return the effective GPU tuning config for the realistic SiGe GPU path."""
    return {
        "min_batch": int(getattr(model, "_vectorized_gpu_min_batch", 8)),
        "layer_algorithm": str(getattr(model, "_vectorized_gpu_layer_algorithm", "4d")),
        "layer_tile": int(getattr(model, "_vectorized_gpu_4d_layer_tile", 13)),
        "candidate_tile": int(getattr(model, "_vectorized_gpu_4d_candidate_tile", 32)),
        "rawkernel_threads": int(getattr(model, "_vectorized_gpu_rawkernel_threads", 128)),
    }


def apply_realistic_sige_gpu_config(model, gpu_config: dict | None = None) -> dict:
    """Apply optional GPU tuning config to a realistic SiGe model and return the effective values."""
    if gpu_config:
        if "min_batch" in gpu_config:
            model._vectorized_gpu_min_batch = int(gpu_config["min_batch"])
        if "layer_algorithm" in gpu_config:
            model._vectorized_gpu_layer_algorithm = str(gpu_config["layer_algorithm"])
        if "layer_tile" in gpu_config:
            model._vectorized_gpu_4d_layer_tile = int(gpu_config["layer_tile"])
        if "candidate_tile" in gpu_config:
            model._vectorized_gpu_4d_candidate_tile = int(gpu_config["candidate_tile"])
        if "rawkernel_threads" in gpu_config:
            model._vectorized_gpu_rawkernel_threads = int(gpu_config["rawkernel_threads"])

    return get_realistic_sige_gpu_config(model)


def build_realistic_sige_model(
    *,
    data_path: str | Path | None = None,
    model_cls=SiGeModelArray,
    freeform_use_cupy: bool = False,
    gpu_config: dict | None = None,
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
    apply_realistic_sige_gpu_config(model, gpu_config)
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
    gpu_config: dict | None = None,
) -> dict:
    """Return a compact description of the realistic imec SiGe optimization surface."""
    model = build_realistic_sige_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
        gpu_config=gpu_config,
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
        "gpu_config": get_realistic_sige_gpu_config(model),
    }


def compare_realistic_sige_objective_throughput(
    *,
    baseline_cls=SiGeModelArray,
    candidate_cls=SiGeModelArray_vectorized,
    baseline_freeform_use_cupy: bool = False,
    candidate_freeform_use_cupy: bool = False,
    baseline_gpu_config: dict | None = None,
    candidate_gpu_config: dict | None = None,
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
        gpu_config=baseline_gpu_config,
    )
    candidate = build_realistic_sige_model(
        data_path=data_path,
        model_cls=candidate_cls,
        freeform_use_cupy=candidate_freeform_use_cupy,
        gpu_config=candidate_gpu_config,
    )
    param_names, _ = get_sige_parameter_names_and_defaults(baseline)

    workflow = describe_realistic_sige_profile(
        data_path=data_path,
        model_cls=baseline_cls,
        freeform_use_cupy=baseline_freeform_use_cupy,
        gpu_config=baseline_gpu_config,
    )
    workflow.update(
        {
            "workflow_api": "objective_throughput",
            "baseline_model_class": baseline.__class__.__name__,
            "candidate_model_class": candidate.__class__.__name__,
            "baseline_freeform_use_cupy": bool(baseline_freeform_use_cupy),
            "candidate_freeform_use_cupy": bool(candidate_freeform_use_cupy),
            "baseline_gpu_config": get_realistic_sige_gpu_config(baseline),
            "candidate_gpu_config": get_realistic_sige_gpu_config(candidate),
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


def measure_realistic_sige_objective_throughput(
    *,
    model_cls=SiGeModelArray_vectorized_GPU,
    freeform_use_cupy: bool = True,
    gpu_config: dict | None = None,
    batch_size: int = 104,
    repeats: int = 1,
    warmups: int = 0,
    amplitude: float = 0.02,
    data_path: str | Path | None = None,
) -> dict:
    """Measure absolute candidate throughput for one realistic SiGe objective path."""
    model = build_realistic_sige_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
        gpu_config=gpu_config,
    )
    candidates = build_realistic_sige_candidate_matrix(
        model,
        batch_size=batch_size,
        amplitude=amplitude,
    )

    values = None
    for _ in range(max(0, warmups)):
        values = evaluate_realistic_sige_batched_objective(model, candidates=candidates)

    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        values = evaluate_realistic_sige_batched_objective(model, candidates=candidates)
    elapsed = time.perf_counter() - start
    total_candidates = int(batch_size * max(1, repeats))

    return {
        "model_class": model.__class__.__name__,
        "freeform_use_cupy": bool(freeform_use_cupy),
        "gpu_config": get_realistic_sige_gpu_config(model),
        "batch_size": int(batch_size),
        "repeats": int(repeats),
        "warmups": int(warmups),
        "elapsed_seconds": float(elapsed),
        "candidates_per_second": float(total_candidates / elapsed),
        "execution_path": getattr(model, "_vectorized_last_execution_path", None),
        "gpu_exception": getattr(model, "_vectorized_last_gpu_exception", None),
        "mean_objective": float(np.mean(np.asarray(values, dtype=float))),
    }


def run_realistic_sige_de_timing(
    *,
    model_cls=SiGeModelArray,
    vectorized: bool = False,
    freeform_use_cupy: bool = False,
    gpu_config: dict | None = None,
    data_path: str | Path | None = None,
    maxiter: int = 2,
    popsize: int = 4,
    tol: float = 0.0,
    polish: bool = False,
    seed: int = 1234,
    updating: str = "deferred",
    use_callbacks: bool = False,
    suppress_callback_plot: bool = True,
    return_model: bool = False,
) -> dict:
    """Run one fixed-configuration realistic SiGe DE timing trial."""
    model = build_realistic_sige_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
        gpu_config=gpu_config,
    )
    param_names, _ = get_sige_parameter_names_and_defaults(model)
    initial_gf = float(getattr(model, "GF", np.inf))
    n_params = len(param_names)
    callback_data = None
    callback_plotter = None
    if use_callbacks and suppress_callback_plot:
        callback_plotter = getattr(model, "_plot_callback_results", None)
        if callback_plotter is not None:
            model._plot_callback_results = lambda: None

    try:
        with _suppress_workflow_output():
            start = time.perf_counter()
            result = model.CDSAXS_Optimize(
                optimizer="differential_evolution",
                use_callbacks=bool(use_callbacks),
                callback_frequency=1,
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
    finally:
        if callback_plotter is not None:
            model._plot_callback_results = callback_plotter

    if use_callbacks:
        callback_data = copy.deepcopy(getattr(model, "_callback_data", {}))

    optimization_result = getattr(model, "optimization_result", None)
    best_fun = None
    if optimization_result is not None and hasattr(optimization_result, "fun"):
        best_fun = float(optimization_result.fun)
    elif hasattr(result, "fun"):
        best_fun = float(result.fun)

    trial = {
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
        "gpu_config": get_realistic_sige_gpu_config(model),
        "execution_path": getattr(model, "_vectorized_last_execution_path", None),
        "gpu_exception": getattr(model, "_vectorized_last_gpu_exception", None),
        "result": result,
        "optimization_result": optimization_result,
        "best_fun": best_fun,
        "best_vector": None if optimization_result is None or not hasattr(optimization_result, "x") else np.asarray(optimization_result.x, dtype=float),
        "callback_data": callback_data,
    }
    if return_model:
        trial["model"] = model
    return trial

def compare_realistic_sige_de_timing(
    *,
    baseline_cls=SiGeModelArray,
    candidate_cls=SiGeModelArray_vectorized,
    baseline_freeform_use_cupy: bool = False,
    candidate_freeform_use_cupy: bool = False,
    baseline_gpu_config: dict | None = None,
    candidate_gpu_config: dict | None = None,
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
        gpu_config=baseline_gpu_config,
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
        gpu_config=candidate_gpu_config,
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
            "baseline_gpu_config": baseline["gpu_config"],
            "candidate_gpu_config": candidate["gpu_config"],
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
    candidate_gpu_config: dict | None = None,
) -> dict:
    """Compare CPU-vectorized SiGe throughput against the GPU vectorized path."""
    return compare_realistic_sige_objective_throughput(
        baseline_cls=baseline_cls,
        candidate_cls=candidate_cls,
        baseline_freeform_use_cupy=False,
        candidate_freeform_use_cupy=True,
        candidate_gpu_config=candidate_gpu_config,
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
    candidate_gpu_config: dict | None = None,
) -> dict:
    """Compare CPU-vectorized SiGe DE timing against the GPU vectorized path."""
    return compare_realistic_sige_de_timing(
        baseline_cls=baseline_cls,
        candidate_cls=candidate_cls,
        baseline_freeform_use_cupy=False,
        candidate_freeform_use_cupy=True,
        candidate_gpu_config=candidate_gpu_config,
        data_path=data_path,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=polish,
        seed=seed,
        updating=updating,
    )


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items() if key not in {"model", "result", "optimization_result"}}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_realistic_sige_report_dir(output_dir: str | Path | None = None, report_name: str | None = None) -> Path:
    root = Path(output_dir) if output_dir is not None else REALISTIC_SIGE_REPORT_ROOT
    root = root.expanduser().resolve()
    name = report_name or f"current_defaults_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    report_dir = root / str(name)
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def _flatten_convergence_row(maxiter: int, seed: int, comparison: dict) -> dict:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    return {
        "maxiter": int(maxiter),
        "seed": int(seed),
        "cpu_elapsed_seconds": float(baseline["elapsed_seconds"]),
        "gpu_elapsed_seconds": float(candidate["elapsed_seconds"]),
        "speedup": float(comparison["speedup"]),
        "cpu_gf": float(baseline["gf"]),
        "gpu_gf": float(candidate["gf"]),
        "absolute_gf_difference": float(comparison["absolute_gf_difference"]),
        "cpu_bic": float(baseline["bic"]),
        "gpu_bic": float(candidate["bic"]),
        "gpu_execution_path": candidate.get("execution_path"),
        "gpu_exception": candidate.get("gpu_exception"),
    }


def _summarize_budget_rows(maxiter: int, rows: list[dict]) -> dict:
    cpu_elapsed = np.array([row["cpu_elapsed_seconds"] for row in rows], dtype=float)
    gpu_elapsed = np.array([row["gpu_elapsed_seconds"] for row in rows], dtype=float)
    speedup = np.array([row["speedup"] for row in rows], dtype=float)
    cpu_gf = np.array([row["cpu_gf"] for row in rows], dtype=float)
    gpu_gf = np.array([row["gpu_gf"] for row in rows], dtype=float)
    gf_diff = np.array([row["absolute_gf_difference"] for row in rows], dtype=float)
    return {
        "maxiter": int(maxiter),
        "seed_count": int(len(rows)),
        "median_cpu_elapsed_seconds": float(np.median(cpu_elapsed)),
        "median_gpu_elapsed_seconds": float(np.median(gpu_elapsed)),
        "median_speedup": float(np.median(speedup)),
        "best_speedup": float(np.max(speedup)),
        "median_cpu_gf": float(np.median(cpu_gf)),
        "median_gpu_gf": float(np.median(gpu_gf)),
        "best_cpu_gf": float(np.min(cpu_gf)),
        "best_gpu_gf": float(np.min(gpu_gf)),
        "max_abs_gf_difference": float(np.max(gf_diff)),
        "median_abs_gf_difference": float(np.median(gf_diff)),
    }


def _save_sige_fit_comparison_plot(cpu_trial: dict, gpu_trial: dict, output_path: Path) -> None:
    cpu_model = cpu_trial["model"]
    gpu_model = gpu_trial["model"]
    intensity = np.asarray(cpu_model.Intensity, dtype=float)
    qz = np.asarray(cpu_model.Qz, dtype=float)
    qx = np.asarray(cpu_model.Qx, dtype=float)
    cpu_simint = np.asarray(cpu_model.SimInt, dtype=float)
    gpu_simint = np.asarray(gpu_model.SimInt, dtype=float)

    finite_cols = np.where(np.any(np.isfinite(intensity), axis=0))[0]
    cut_indices = np.unique(np.linspace(0, len(finite_cols) - 1, 6, dtype=int))
    cut_cols = finite_cols[cut_indices]

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    vmin = np.nanpercentile(np.log10(np.clip(intensity, 1e-30, None)), 5)
    vmax = np.nanpercentile(np.log10(np.clip(intensity, 1e-30, None)), 95)
    panels = [
        ("Experimental log10(I)", intensity),
        (f"Scalar CPU fit\nGF={cpu_trial['gf']:.4f}, t={cpu_trial['elapsed_seconds']:.2f}s", cpu_simint),
        (f"GPU vectorized fit\nGF={gpu_trial['gf']:.4f}, t={gpu_trial['elapsed_seconds']:.2f}s", gpu_simint),
    ]

    for panel_index, (title, image) in enumerate(panels):
        ax = fig.add_subplot(grid[0, panel_index])
        im = ax.imshow(
            np.log10(np.clip(image, 1e-30, None)),
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("qx column")
        ax.set_ylabel("qz row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[1, :])
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(cut_cols)))
    for color, column in zip(colors, cut_cols):
        mask = np.isfinite(intensity[:, column]) & np.isfinite(qz[:, column])
        qz_col = qz[mask, column]
        exp_col = intensity[mask, column]
        cpu_col = cpu_simint[mask, column]
        gpu_col = gpu_simint[mask, column]
        qx_value = float(np.nanmedian(qx[:, column]))
        ax.plot(qz_col, exp_col, color=color, linewidth=2.2, alpha=0.9, label=f"exp qx={qx_value:g}")
        ax.plot(qz_col, cpu_col, color=color, linestyle="--", linewidth=1.2, alpha=0.9, label=f"cpu qx={qx_value:g}")
        ax.plot(qz_col, gpu_col, color=color, linestyle=":", linewidth=1.8, alpha=0.9, label=f"gpu qx={qx_value:g}")

    ax.set_yscale("log")
    ax.set_xlabel("qz")
    ax.set_ylabel("Intensity")
    ax.set_title("Representative cuts: experiment vs scalar CPU vs GPU vectorized")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.suptitle("Realistic SiGe convergence representative fit")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_sige_convergence_summary_plot(
    budget_summaries: list[dict],
    cpu_trial: dict,
    gpu_trial: dict,
    output_path: Path,
) -> None:
    maxiters = np.array([row["maxiter"] for row in budget_summaries], dtype=int)
    median_cpu_gf = np.array([row["median_cpu_gf"] for row in budget_summaries], dtype=float)
    median_gpu_gf = np.array([row["median_gpu_gf"] for row in budget_summaries], dtype=float)
    best_cpu_gf = np.array([row["best_cpu_gf"] for row in budget_summaries], dtype=float)
    best_gpu_gf = np.array([row["best_gpu_gf"] for row in budget_summaries], dtype=float)
    median_cpu_elapsed = np.array([row["median_cpu_elapsed_seconds"] for row in budget_summaries], dtype=float)
    median_gpu_elapsed = np.array([row["median_gpu_elapsed_seconds"] for row in budget_summaries], dtype=float)
    median_speedup = np.array([row["median_speedup"] for row in budget_summaries], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(maxiters, median_cpu_gf, "o-", label="CPU scalar")
    axes[0, 0].plot(maxiters, median_gpu_gf, "o-", label="GPU vectorized")
    axes[0, 0].set_title("Median Final GF vs DE Budget")
    axes[0, 0].set_xlabel("maxiter")
    axes[0, 0].set_ylabel("GF")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(maxiters, best_cpu_gf, "o-", label="CPU scalar")
    axes[0, 1].plot(maxiters, best_gpu_gf, "o-", label="GPU vectorized")
    axes[0, 1].set_title("Best Final GF vs DE Budget")
    axes[0, 1].set_xlabel("maxiter")
    axes[0, 1].set_ylabel("GF")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(maxiters, median_cpu_elapsed, "o-", label="CPU scalar")
    axes[1, 0].plot(maxiters, median_gpu_elapsed, "o-", label="GPU vectorized")
    axes[1, 0].set_title("Median Wall Time vs DE Budget")
    axes[1, 0].set_xlabel("maxiter")
    axes[1, 0].set_ylabel("seconds")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(maxiters, median_speedup, "o-", color="tab:green")
    axes[1, 1].set_title("Median GPU Speedup vs DE Budget")
    axes[1, 1].set_xlabel("maxiter")
    axes[1, 1].set_ylabel("CPU / GPU")
    axes[1, 1].grid(True, alpha=0.3)

    cpu_curve = np.asarray(cpu_trial["callback_data"]["best_objective"], dtype=float)
    gpu_curve = np.asarray(gpu_trial["callback_data"]["best_objective"], dtype=float)
    trajectory_ax = axes[0, 1].inset_axes([0.48, 0.5, 0.48, 0.42])
    trajectory_ax.plot(np.arange(1, cpu_curve.size + 1), cpu_curve, label="CPU scalar")
    trajectory_ax.plot(np.arange(1, gpu_curve.size + 1), gpu_curve, label="GPU vectorized")
    trajectory_ax.set_title("Representative best-so-far GF")
    trajectory_ax.set_xlabel("generation")
    trajectory_ax.set_ylabel("GF")
    trajectory_ax.grid(True, alpha=0.2)
    trajectory_ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_realistic_sige_convergence_suite(
    *,
    data_path: str | Path | None = None,
    seeds: tuple[int, ...] = (1234, 1235, 1236, 1237),
    maxiters: tuple[int, ...] = (3, 5, 10, 20),
    popsize: int = 4,
    tol: float = 0.0,
    polish: bool = False,
    updating: str = "deferred",
    gpu_config: dict | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
) -> dict:
    """Run a manual realistic SiGe convergence suite for scalar CPU versus GPU vectorized."""
    report_dir = _ensure_realistic_sige_report_dir(output_dir, report_name)
    workflow = {
        "geometry": "sige",
        "label": "imec_ellipse_stack",
        "comparison": "cpu_scalar_vs_gpu_vectorized",
        "optimizer": "differential_evolution",
        "seeds": tuple(int(seed) for seed in seeds),
        "maxiters": tuple(int(maxiter) for maxiter in maxiters),
        "popsize": int(popsize),
        "tol": float(tol),
        "polish": bool(polish),
        "updating": str(updating),
        "candidate_gpu_config": get_realistic_sige_gpu_config(build_realistic_sige_model(
            data_path=data_path,
            model_cls=SiGeModelArray_vectorized_GPU,
            freeform_use_cupy=True,
            gpu_config=gpu_config,
        )),
        "report_dir": str(report_dir),
    }

    comparisons = []
    rows = []
    representative_seed = None
    representative_budget = None
    representative_gpu_gf = np.inf
    largest_budget = max(int(value) for value in maxiters)

    for maxiter in tuple(int(value) for value in maxiters):
        for seed in tuple(int(value) for value in seeds):
            comparison = compare_realistic_sige_de_timing(
                baseline_cls=SiGeModelArray,
                candidate_cls=SiGeModelArray_vectorized_GPU,
                baseline_freeform_use_cupy=False,
                candidate_freeform_use_cupy=True,
                candidate_gpu_config=gpu_config,
                data_path=data_path,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                polish=polish,
                seed=seed,
                updating=updating,
            )
            comparisons.append({"maxiter": maxiter, "seed": seed, "comparison": comparison})
            row = _flatten_convergence_row(maxiter, seed, comparison)
            rows.append(row)
            if maxiter == largest_budget and row["gpu_gf"] < representative_gpu_gf:
                representative_seed = seed
                representative_budget = maxiter
                representative_gpu_gf = row["gpu_gf"]

    budget_summaries = [
        _summarize_budget_rows(maxiter, [row for row in rows if row["maxiter"] == int(maxiter)])
        for maxiter in tuple(int(value) for value in maxiters)
    ]

    if representative_seed is None or representative_budget is None:
        raise RuntimeError("No representative realistic SiGe convergence run was selected")

    representative_cpu = run_realistic_sige_de_timing(
        model_cls=SiGeModelArray,
        vectorized=False,
        freeform_use_cupy=False,
        data_path=data_path,
        maxiter=representative_budget,
        popsize=popsize,
        tol=tol,
        polish=polish,
        seed=representative_seed,
        updating=updating,
        use_callbacks=True,
        return_model=True,
    )
    representative_gpu = run_realistic_sige_de_timing(
        model_cls=SiGeModelArray_vectorized_GPU,
        vectorized=True,
        freeform_use_cupy=True,
        gpu_config=gpu_config,
        data_path=data_path,
        maxiter=representative_budget,
        popsize=popsize,
        tol=tol,
        polish=polish,
        seed=representative_seed,
        updating=updating,
        use_callbacks=True,
        return_model=True,
    )

    fit_plot_path = report_dir / f"representative_fit_seed{representative_seed}_iter{representative_budget}_pop{int(popsize)}.png"
    summary_plot_path = report_dir / f"convergence_summary_pop{int(popsize)}.png"
    _save_sige_fit_comparison_plot(representative_cpu, representative_gpu, fit_plot_path)
    _save_sige_convergence_summary_plot(budget_summaries, representative_cpu, representative_gpu, summary_plot_path)

    summary = {
        "workflow": workflow,
        "budget_summaries": budget_summaries,
        "representative": {
            "seed": int(representative_seed),
            "maxiter": int(representative_budget),
            "cpu_gf": float(representative_cpu["gf"]),
            "gpu_gf": float(representative_gpu["gf"]),
            "absolute_gf_difference": float(abs(representative_cpu["gf"] - representative_gpu["gf"])),
            "cpu_elapsed_seconds": float(representative_cpu["elapsed_seconds"]),
            "gpu_elapsed_seconds": float(representative_gpu["elapsed_seconds"]),
            "speedup": float(representative_cpu["elapsed_seconds"] / representative_gpu["elapsed_seconds"]),
            "gpu_execution_path": representative_gpu.get("execution_path"),
            "fit_plot_path": str(fit_plot_path),
            "summary_plot_path": str(summary_plot_path),
        },
    }

    csv_path = report_dir / "convergence_runs.csv"
    json_path = report_dir / "convergence_summary.json"
    _write_rows_csv(csv_path, rows)
    json_path.write_text(json.dumps(_json_ready({"summary": summary, "runs": rows}), indent=2))

    return {
        "report_dir": str(report_dir),
        "workflow": workflow,
        "results": rows,
        "budget_summaries": budget_summaries,
        "representative": summary["representative"],
        "summary_json_path": str(json_path),
        "results_csv_path": str(csv_path),
    }


def run_realistic_sige_gpu_tuning_sweep(
    *,
    data_path: str | Path | None = None,
    thread_options: tuple[int, ...] = (64, 128, 256, 512),
    include_loop_reference: bool = True,
    popsize: int = 4,
    maxiter: int = 3,
    tol: float = 0.0,
    polish: bool = False,
    seed: int = 1234,
    updating: str = "deferred",
    repeats: int = 2,
    warmups: int = 1,
    scalar_reference: dict | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
) -> dict:
    """Sweep realistic GPU tuning candidates and pick the fastest current default for DE-shaped work."""
    report_dir = _ensure_realistic_sige_report_dir(output_dir, report_name)
    baseline = build_realistic_sige_model(data_path=data_path, model_cls=SiGeModelArray)
    parameter_count = len(get_sige_parameter_names_and_defaults(baseline)[0])
    practical_batch = int(popsize * parameter_count)

    candidates: list[dict] = []
    if include_loop_reference:
        candidates.append({"label": "loop", "layer_algorithm": "loop"})
    candidates.extend(
        {
            "label": f"4d_threads_{int(threads)}",
            "layer_algorithm": "4d",
            "rawkernel_threads": int(threads),
        }
        for threads in thread_options
    )

    rows = []
    for candidate_gpu_config in candidates:
        throughput = measure_realistic_sige_objective_throughput(
            model_cls=SiGeModelArray_vectorized_GPU,
            freeform_use_cupy=True,
            gpu_config=candidate_gpu_config,
            batch_size=practical_batch,
            repeats=repeats,
            warmups=warmups,
            data_path=data_path,
        )
        gpu_trial = run_realistic_sige_de_timing(
            model_cls=SiGeModelArray_vectorized_GPU,
            vectorized=True,
            freeform_use_cupy=True,
            gpu_config=candidate_gpu_config,
            data_path=data_path,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            seed=seed,
            updating=updating,
        )
        scalar_elapsed = None
        scalar_gf = None
        de_speedup = None
        abs_gf_difference = None
        if scalar_reference is not None:
            scalar_elapsed = float(scalar_reference.get("elapsed_seconds"))
            scalar_gf = float(scalar_reference.get("gf"))
            de_speedup = float(scalar_elapsed / gpu_trial["elapsed_seconds"])
            abs_gf_difference = float(abs(scalar_gf - gpu_trial["gf"]))
        rows.append(
            {
                "label": str(candidate_gpu_config["label"]),
                "layer_algorithm": str(candidate_gpu_config["layer_algorithm"]),
                "rawkernel_threads": int(candidate_gpu_config.get("rawkernel_threads", 0)),
                "objective_batch_size": int(practical_batch),
                "gpu_candidates_per_second": float(throughput["candidates_per_second"]),
                "objective_execution_path": throughput.get("execution_path"),
                "gpu_elapsed_seconds": float(gpu_trial["elapsed_seconds"]),
                "de_speedup": de_speedup,
                "gpu_execution_path": gpu_trial.get("execution_path"),
                "scalar_elapsed_seconds": scalar_elapsed,
                "scalar_gf": scalar_gf,
                "gpu_gf": float(gpu_trial["gf"]),
                "absolute_gf_difference": abs_gf_difference,
            }
        )

    best_row = min(rows, key=lambda row: (row["gpu_elapsed_seconds"], -row["gpu_candidates_per_second"]))
    csv_path = report_dir / "gpu_tuning_sweep.csv"
    json_path = report_dir / "gpu_tuning_sweep.json"
    _write_rows_csv(csv_path, rows)
    json_path.write_text(json.dumps(_json_ready({"best": best_row, "rows": rows}), indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [row["label"] for row in rows]
    axes[0].bar(labels, [row["gpu_candidates_per_second"] for row in rows], color="tab:blue")
    axes[0].set_title("GPU objective throughput")
    axes[0].set_ylabel("candidates / second")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(labels, [row["gpu_elapsed_seconds"] for row in rows], color="tab:green")
    axes[1].set_title("GPU DE wall time")
    axes[1].set_ylabel("seconds")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    tuning_plot_path = report_dir / "gpu_tuning_sweep.png"
    fig.savefig(tuning_plot_path, dpi=180)
    plt.close(fig)

    return {
        "report_dir": str(report_dir),
        "objective_batch_size": int(practical_batch),
        "best": best_row,
        "rows": rows,
        "summary_json_path": str(json_path),
        "results_csv_path": str(csv_path),
        "plot_path": str(tuning_plot_path),
    }
