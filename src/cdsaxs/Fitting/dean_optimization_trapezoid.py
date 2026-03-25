from __future__ import annotations

import copy
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .Trapezoid_model import TrapezoidModel, TrapezoidModelArray
from .Trapezoid_model_dean import TrapezoidModelArrayDean, TrapezoidModelDean
from .Trapezoid_model_dean_gpu import (
    TrapezoidModelArrayDeanGPU,
    TrapezoidModelArrayDeanGPUFused,
    TrapezoidModelDeanGPU,
    TrapezoidModelDeanGPUFused,
)


DEFAULT_TRAPEZOID_MODEL_PARAMS = {
    "layers": 1,
    "trapezoids": [
        {"width": 400.0, "height": 700.0},
        {"width": 83.0, "height": 0.0},
    ],
    "DW": 15.0,
    "I0": 0.000001,
    "Bk": 0.5,
    "slds": [1.0],
    "optimization": {
        "trap_0_width": {"min": 300.0, "max": 1200.0, "default": 400.0},
        "trap_0_height": {"min": 500.0, "max": 800.0, "default": 700.0},
        "trap_1_width": {"min": 10.0, "max": 120.0, "default": 83.0},
        "DW": {"min": 1.0, "max": 50.0, "default": 15.0},
    },
}

EXAMPLE_DATA_FILENAME = "75s_30nm_200nmpitch_IvsQz.csv"


def get_example_data_path(data_path: str | Path | None = None) -> Path:
    """Resolve the trapezoid example data used in the DeRocher fitting notebooks."""
    if data_path is not None:
        return Path(data_path).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "examples" / "fitting_examples" / EXAMPLE_DATA_FILENAME


def build_example_trapezoid_model(
    data_path: str | Path | None = None,
    model_params: dict | None = None,
    model_cls=TrapezoidModelArray,
    freeform_use_cupy: bool = False,
) -> TrapezoidModelArray:
    """
    Build the notebook-style trapezoid example model using the current public API.
    """
    params = copy.deepcopy(model_params or DEFAULT_TRAPEZOID_MODEL_PARAMS)
    model = model_cls(
        model="single_material",
        layers=params["layers"],
        model_params=params,
    )
    model._freeform_use_cupy = bool(freeform_use_cupy)
    model.importCDSAXS_GUI(str(get_example_data_path(data_path)))
    return model


def get_parameter_names_and_defaults(model: TrapezoidModelArray) -> tuple[list[str], np.ndarray]:
    """Return the optimization parameter order and its default vector."""
    optimization = model.model_params["optimization"]
    param_names = list(optimization.keys())
    defaults = np.array([optimization[name]["default"] for name in param_names], dtype=float)
    return param_names, defaults


def build_candidate_matrix(
    model: TrapezoidModelArray,
    batch_size: int = 8,
    amplitude: float = 0.02,
) -> np.ndarray:
    """
    Build a deterministic batch of candidates near the default parameter vector.
    """
    param_names, defaults = get_parameter_names_and_defaults(model)
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


def evaluate_scalar_objective(
    model: TrapezoidModelArray,
    vector: np.ndarray | None = None,
) -> float:
    """Evaluate the trapezoid objective for a single candidate."""
    param_names, defaults = get_parameter_names_and_defaults(model)
    model.param_names = param_names
    candidate = np.asarray(vector if vector is not None else defaults, dtype=float)
    value = model._trapezoid_optimization_wrapper(candidate)
    return float(value)


def evaluate_batched_objective(
    model: TrapezoidModelArray,
    candidates: np.ndarray | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Evaluate the trapezoid objective for a batch of candidates."""
    param_names, _ = get_parameter_names_and_defaults(model)
    model.param_names = param_names
    batch = candidates if candidates is not None else build_candidate_matrix(model, batch_size=batch_size)
    values = model._trapezoid_optimization_wrapper(np.asarray(batch, dtype=float))
    return np.asarray(values, dtype=float)


def run_objective_trial(
    *,
    batch_size: int = 8,
    repeats: int = 1,
    data_path: str | Path | None = None,
    model_cls=TrapezoidModelArray,
    freeform_use_cupy: bool = False,
) -> dict:
    """
    Time both scalar and batched objective evaluation using the notebook-style example.
    """
    model = build_example_trapezoid_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    _, defaults = get_parameter_names_and_defaults(model)
    candidates = build_candidate_matrix(model, batch_size=batch_size)

    scalar_value = None
    scalar_start = time.perf_counter()
    for _ in range(max(1, repeats)):
        scalar_value = evaluate_scalar_objective(model, defaults)
    scalar_elapsed = time.perf_counter() - scalar_start

    batched_values = None
    batched_start = time.perf_counter()
    for _ in range(max(1, repeats)):
        batched_values = evaluate_batched_objective(model, candidates=candidates)
    batched_elapsed = time.perf_counter() - batched_start

    return {
        "batch_size": int(batch_size),
        "repeats": int(repeats),
        "scalar_elapsed_seconds": scalar_elapsed,
        "scalar_value": float(scalar_value),
        "batched_elapsed_seconds": batched_elapsed,
        "batched_values": np.asarray(batched_values, dtype=float),
    }


def run_optimizer_trial(
    *,
    model_cls=TrapezoidModelArray,
    vectorized: bool = False,
    workers: int | None = None,
    maxiter: int = 1,
    popsize: int = 4,
    tol: float = 0.5,
    polish: bool = False,
    seed: int | None = 1234,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
) -> dict:
    """
    Run a lightweight DE optimization trial suitable for repeated benchmarking.
    """
    model = build_example_trapezoid_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )

    optimizer_kwargs = {
        "maxiter": int(maxiter),
        "popsize": int(popsize),
        "tol": float(tol),
        "mutation": 1.0,
        "recombination": 0.5,
        "polish": bool(polish),
    }
    if seed is not None:
        optimizer_kwargs["seed"] = int(seed)
    if workers is not None and not vectorized:
        optimizer_kwargs["workers"] = int(workers)

    start = time.perf_counter()
    result = model.CDSAXS_Optimize(
        optimizer="differential_evolution",
        use_callbacks=False,
        verbose=False,
        vectorized=bool(vectorized),
        freeform_use_cupy=bool(freeform_use_cupy),
        plot_results=False,
        plot_structure=False,
        plot_grid=False,
        plot_combined=False,
        **optimizer_kwargs,
    )
    elapsed = time.perf_counter() - start

    return {
        "vectorized": bool(vectorized),
        "workers": workers,
        "elapsed_seconds": elapsed,
        "result": result,
        "gf": float(getattr(model, "GF", np.inf)),
        "bic": float(getattr(model, "BIC", np.inf)),
        "optimization_result": getattr(model, "optimization_result", None),
        "model": model,
    }


def get_objective_batch_profile(profile: str = "broadcast") -> tuple[int, ...]:
    """Return a named batch-size profile for objective timing experiments."""
    profiles = {
        "smoke": (1, 4, 16),
        "broadcast": (1, 4, 16, 64, 256, 1024),
        "gpu_candidate": (1, 16, 64, 256, 1024, 4096),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown batch profile: {profile}")
    return profiles[profile]


def run_objective_scaling_trial(
    *,
    model_cls=TrapezoidModelArray,
    batch_sizes: tuple[int, ...] | None = None,
    repeats: int = 5,
    warmups: int = 1,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
) -> dict:
    """
    Measure objective throughput across increasing batch sizes for one model class.
    """
    model = build_example_trapezoid_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    batch_sizes = batch_sizes or get_objective_batch_profile("broadcast")
    results = []

    for batch_size in batch_sizes:
        candidates = build_candidate_matrix(model, batch_size=batch_size)

        for _ in range(max(0, warmups)):
            evaluate_batched_objective(model, candidates=candidates)

        start = time.perf_counter()
        for _ in range(max(1, repeats)):
            values = evaluate_batched_objective(model, candidates=candidates)
        elapsed = time.perf_counter() - start

        results.append(
            {
                "batch_size": int(batch_size),
                "repeats": int(repeats),
                "elapsed_seconds": elapsed,
                "seconds_per_call": elapsed / max(1, repeats),
                "seconds_per_candidate": elapsed / max(1, repeats * batch_size),
                "mean_objective": float(np.mean(values)),
            }
        )

    return {
        "model_class": model_cls.__name__,
        "batch_sizes": tuple(int(size) for size in batch_sizes),
        "results": results,
    }


def compare_objective_scaling(
    *,
    baseline_cls=TrapezoidModelArray,
    candidate_cls=TrapezoidModelArrayDean,
    batch_sizes: tuple[int, ...] | None = None,
    repeats: int = 5,
    warmups: int = 1,
    data_path: str | Path | None = None,
    baseline_freeform_use_cupy: bool = False,
    candidate_freeform_use_cupy: bool = False,
) -> dict:
    """
    Compare objective scaling between the baseline trapezoid model and a candidate variant.
    """
    baseline = run_objective_scaling_trial(
        model_cls=baseline_cls,
        batch_sizes=batch_sizes,
        repeats=repeats,
        warmups=warmups,
        data_path=data_path,
        freeform_use_cupy=baseline_freeform_use_cupy,
    )
    candidate = run_objective_scaling_trial(
        model_cls=candidate_cls,
        batch_sizes=batch_sizes,
        repeats=repeats,
        warmups=warmups,
        data_path=data_path,
        freeform_use_cupy=candidate_freeform_use_cupy,
    )

    speedups = []
    for base_result, candidate_result in zip(baseline["results"], candidate["results"]):
        speedups.append(
            {
                "batch_size": base_result["batch_size"],
                "baseline_seconds_per_call": base_result["seconds_per_call"],
                "candidate_seconds_per_call": candidate_result["seconds_per_call"],
                "speedup": base_result["seconds_per_call"] / candidate_result["seconds_per_call"],
            }
        )

    return {
        "baseline": baseline,
        "candidate": candidate,
        "speedups": speedups,
    }


def extract_structure_signature(model) -> dict:
    """Extract a compact structure summary for later similarity regression."""
    widths = []
    heights = []
    for trap in model.model_params["trapezoids"]:
        widths.append(float(trap["width"]))
        heights.append(float(trap["height"]))

    if isinstance(model.Bk, np.ndarray):
        background = np.asarray(model.Bk, dtype=float)
    else:
        background = np.asarray([float(model.Bk)], dtype=float)

    return {
        "widths": np.asarray(widths, dtype=float),
        "heights": np.asarray(heights, dtype=float),
        "dw": float(model.DW),
        "i0": float(model.I0),
        "background_mean": float(np.mean(background)),
        "background_std": float(np.std(background)),
        "gf": float(getattr(model, "GF", np.inf)),
        "bic": float(getattr(model, "BIC", np.inf)),
    }


def profile_objective_components(
    *,
    model_cls=TrapezoidModelArrayDean,
    batch_size: int = 256,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
) -> dict:
    """
    Attribute one batched objective call across the dominant trapezoid subroutines.
    """
    model = build_example_trapezoid_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    model.param_names = list(model.model_params["optimization"].keys())
    candidates = build_candidate_matrix(model, batch_size=batch_size)

    timings = defaultdict(float)
    counts = defaultdict(int)
    originals = {}

    def install_wrapper(name):
        original = getattr(model, name)

        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            out = original(*args, **kwargs)
            timings[name] += time.perf_counter() - start
            counts[name] += 1
            return out

        originals[name] = original
        setattr(model, name, wrapped)

    for name in ("SymCoordAssign", "FreeFormTrapezoid", "GF_calc"):
        install_wrapper(name)

    total_start = time.perf_counter()
    values = model._trapezoid_optimization_wrapper(candidates)
    total_elapsed = time.perf_counter() - total_start

    for name, original in originals.items():
        setattr(model, name, original)

    component_rows = []
    for name in ("SymCoordAssign", "FreeFormTrapezoid", "GF_calc"):
        component_rows.append(
            {
                "name": name,
                "elapsed_seconds": timings[name],
                "calls": counts[name],
                "fraction_of_total": timings[name] / total_elapsed if total_elapsed else float("nan"),
            }
        )

    return {
        "model_class": model_cls.__name__,
        "batch_size": int(batch_size),
        "freeform_use_cupy": bool(freeform_use_cupy),
        "total_elapsed_seconds": total_elapsed,
        "components": component_rows,
        "unaccounted_seconds": total_elapsed - sum(timings.values()),
        "mean_objective": float(np.mean(values)),
    }


def run_structure_similarity_trial(
    *,
    baseline_cls=TrapezoidModelArray,
    candidate_cls=TrapezoidModelArrayDean,
    optimizer_kwargs: dict | None = None,
    data_path: str | Path | None = None,
    baseline_freeform_use_cupy: bool = False,
    candidate_freeform_use_cupy: bool = False,
) -> dict:
    """
    Run matching lightweight optimization trials and collect structure signatures.

    This is intentionally lightweight and should only become a gating regression once a
    candidate path has demonstrated meaningful speedup.
    """
    kwargs = {
        "maxiter": 2,
        "popsize": 4,
        "tol": 0.5,
        "polish": False,
        "seed": 1234,
        "vectorized": True,
    }
    if optimizer_kwargs:
        kwargs.update(optimizer_kwargs)

    baseline_run = run_optimizer_trial(
        model_cls=baseline_cls,
        data_path=data_path,
        freeform_use_cupy=baseline_freeform_use_cupy,
        **kwargs,
    )
    candidate_run = run_optimizer_trial(
        model_cls=candidate_cls,
        data_path=data_path,
        freeform_use_cupy=candidate_freeform_use_cupy,
        **kwargs,
    )

    return {
        "baseline": extract_structure_signature(baseline_run["model"]),
        "candidate": extract_structure_signature(candidate_run["model"]),
        "baseline_elapsed_seconds": baseline_run["elapsed_seconds"],
        "candidate_elapsed_seconds": candidate_run["elapsed_seconds"],
    }
