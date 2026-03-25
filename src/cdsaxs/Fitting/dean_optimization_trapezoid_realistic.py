from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np

from .Trapezoid_model import TrapezoidModelArray
from .Trapezoid_model_dean_gpu import TrapezoidModelArrayDeanGPUFused
from .dean_optimization_trapezoid import (
    build_candidate_matrix,
    build_example_trapezoid_model,
    evaluate_batched_objective,
    get_parameter_names_and_defaults,
)


REALISTIC_TRAPEZOID_PROFILE_SPECS = {
    "notebook_4param": {
        "label": "Notebook base trapezoid",
        "builder": "base",
        "expected_parameter_count": 4,
        "candidate_budget_population_sizes": (128, 256, 512, 1024),
        "one_generation_population_sizes": (128, 256, 512),
        "multi_generation_population_sizes": (256, 512),
    },
    "notebook_layer10_8param": {
        "label": "Notebook added-layer trapezoid at 10%",
        "builder": "single_layer",
        "height_percentage": 10,
        "expected_parameter_count": 8,
        "candidate_budget_population_sizes": (32, 64, 128, 256),
        "one_generation_population_sizes": (32, 64, 128),
        "multi_generation_population_sizes": (64, 128),
    },
    "notebook_layer90_8param": {
        "label": "Notebook added-layer trapezoid at 90%",
        "builder": "single_layer",
        "height_percentage": 90,
        "expected_parameter_count": 8,
        "candidate_budget_population_sizes": (32, 64, 128, 256),
        "one_generation_population_sizes": (32, 64, 128),
        "multi_generation_population_sizes": (64, 128),
    },
    "notebook_multilayer_12param": {
        "label": "Notebook multi-added-layer trapezoid",
        "builder": "multi_layer",
        "height_percentages": (10, 50, 90),
        "expected_parameter_count": 12,
        "candidate_budget_population_sizes": (16, 32, 64, 128),
        "one_generation_population_sizes": (16, 32, 64),
        "multi_generation_population_sizes": (32, 64),
    },
}


def get_realistic_trapezoid_profile_spec(profile: str = "notebook_4param") -> dict:
    """Return metadata for one notebook-parallel realistic trapezoid profile."""
    try:
        return REALISTIC_TRAPEZOID_PROFILE_SPECS[profile]
    except KeyError as exc:  # pragma: no cover - defensive branch
        known = ", ".join(sorted(REALISTIC_TRAPEZOID_PROFILE_SPECS))
        raise ValueError(f"Unknown realistic trapezoid profile: {profile}. Known profiles: {known}") from exc


def get_realistic_population_sizes(
    profile: str = "notebook_4param",
    family: str = "candidate_budget",
) -> tuple[int, ...]:
    """Return the population-size ladder for a benchmark family."""
    spec = get_realistic_trapezoid_profile_spec(profile)
    key = f"{family}_population_sizes"
    if key not in spec:
        raise ValueError(f"Unknown realistic benchmark family: {family}")
    return tuple(int(size) for size in spec[key])


def _derive_realistic_trapezoid_model_params(
    profile: str = "notebook_4param",
    data_path: str | Path | None = None,
) -> dict:
    """
    Build notebook-parallel trapezoid parameters using the current public model API.

    Layer-addition helpers currently return the base trapezoid class, so the realistic
    params are derived on a temporary baseline model and then re-instantiated on the
    requested optimization variant.
    """
    spec = get_realistic_trapezoid_profile_spec(profile)
    base_model = build_example_trapezoid_model(
        data_path=data_path,
        model_cls=TrapezoidModelArray,
    )

    builder = spec["builder"]
    if builder == "base":
        derived_model = base_model
    elif builder == "single_layer":
        derived_model = base_model.add_layer_at_percentage(int(spec["height_percentage"]))
    elif builder == "multi_layer":
        derived_model = base_model.add_multiple_layers(
            list(spec["height_percentages"]),
            sequential=True,
        )
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported realistic trapezoid builder: {builder}")

    params = copy.deepcopy(derived_model.model_params)
    layers = int(params.get("layers", max(1, len(params.get("trapezoids", ())) - 1)))
    slds = list(params.get("slds", []))

    if not slds:
        params["slds"] = [1.0] * layers
    elif len(slds) == 1 and layers > 1:
        params["slds"] = slds * layers

    return params


def build_realistic_trapezoid_model(
    *,
    profile: str = "notebook_4param",
    data_path: str | Path | None = None,
    model_cls=TrapezoidModelArrayDeanGPUFused,
    freeform_use_cupy: bool = False,
):
    """Build a notebook-derived trapezoid model on a selected fitting class."""
    params = _derive_realistic_trapezoid_model_params(profile=profile, data_path=data_path)
    return build_example_trapezoid_model(
        data_path=data_path,
        model_params=params,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )


def describe_realistic_trapezoid_profile(
    *,
    profile: str = "notebook_4param",
    data_path: str | Path | None = None,
) -> dict:
    """Return a compact description of a realistic trapezoid benchmark profile."""
    spec = get_realistic_trapezoid_profile_spec(profile)
    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=TrapezoidModelArray,
    )
    param_names, defaults = get_parameter_names_and_defaults(model)
    return {
        "profile": profile,
        "label": spec["label"],
        "layers": int(model.layers),
        "trapezoid_count": int(len(model.model_params["trapezoids"])),
        "parameter_count": int(len(param_names)),
        "parameter_names": tuple(param_names),
        "default_vector": np.asarray(defaults, dtype=float),
    }


def _candidate_count_from_population_size(model, population_size: int) -> int:
    param_names, _ = get_parameter_names_and_defaults(model)
    return int(len(param_names) * int(population_size))


def run_realistic_candidate_budget_trial(
    *,
    profile: str = "notebook_4param",
    model_cls=TrapezoidModelArrayDeanGPUFused,
    population_sizes: tuple[int, ...] | None = None,
    repeats: int = 3,
    warmups: int = 1,
    amplitude: float = 0.02,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
) -> dict:
    """
    Time deterministic batched objective calls across notebook-parallel population sizes.
    """
    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    param_names, _ = get_parameter_names_and_defaults(model)
    population_sizes = population_sizes or get_realistic_population_sizes(profile, family="candidate_budget")

    results = []
    for population_size in population_sizes:
        candidate_count = _candidate_count_from_population_size(model, population_size)
        candidates = build_candidate_matrix(
            model,
            batch_size=candidate_count,
            amplitude=amplitude,
        )

        for _ in range(max(0, warmups)):
            evaluate_batched_objective(model, candidates=candidates)

        start = time.perf_counter()
        for _ in range(max(1, repeats)):
            values = evaluate_batched_objective(model, candidates=candidates)
        elapsed = time.perf_counter() - start

        results.append(
            {
                "population_size": int(population_size),
                "candidate_count": int(candidate_count),
                "parameter_count": int(len(param_names)),
                "repeats": int(repeats),
                "elapsed_seconds": elapsed,
                "seconds_per_call": elapsed / max(1, repeats),
                "seconds_per_candidate": elapsed / max(1, repeats * candidate_count),
                "candidates_per_second": (max(1, repeats) * candidate_count) / elapsed if elapsed else float("inf"),
                "mean_objective": float(np.mean(values)),
            }
        )

    return {
        "profile": profile,
        "profile_spec": get_realistic_trapezoid_profile_spec(profile),
        "model_class": model_cls.__name__,
        "freeform_use_cupy": bool(freeform_use_cupy),
        "results": results,
    }


def _run_realistic_optimizer_trial(
    *,
    profile: str,
    model_cls,
    population_size: int,
    maxiter: int,
    tol: float,
    polish: bool,
    seed: int | None,
    data_path: str | Path | None,
    freeform_use_cupy: bool,
) -> dict:
    """Run one DE fit on a notebook-derived realistic trapezoid model."""
    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )

    optimizer_kwargs = {
        "maxiter": int(maxiter),
        "popsize": int(population_size),
        "tol": float(tol),
        "mutation": 1.0,
        "recombination": 0.5,
        "polish": bool(polish),
    }
    if seed is not None:
        optimizer_kwargs["seed"] = int(seed)

    start = time.perf_counter()
    result = model.CDSAXS_Optimize(
        optimizer="differential_evolution",
        use_callbacks=False,
        verbose=False,
        vectorized=True,
        freeform_use_cupy=bool(freeform_use_cupy),
        plot_results=False,
        plot_structure=False,
        plot_grid=False,
        plot_combined=False,
        **optimizer_kwargs,
    )
    elapsed = time.perf_counter() - start

    return {
        "elapsed_seconds": elapsed,
        "result": result,
        "gf": float(getattr(model, "GF", np.inf)),
        "bic": float(getattr(model, "BIC", np.inf)),
    }


def run_realistic_optimizer_profile(
    *,
    profile: str = "notebook_4param",
    family: str = "one_generation",
    model_cls=TrapezoidModelArrayDeanGPUFused,
    population_sizes: tuple[int, ...] | None = None,
    maxiter: int | None = None,
    tol: float = 0.5,
    polish: bool = False,
    seed: int | None = 1234,
    repeats: int = 1,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
) -> dict:
    """Run notebook-parallel DE timing trials for one realistic benchmark family."""
    if family not in {"one_generation", "multi_generation"}:
        raise ValueError(f"Unknown realistic optimizer family: {family}")

    profile_info = describe_realistic_trapezoid_profile(profile=profile, data_path=data_path)
    population_sizes = population_sizes or get_realistic_population_sizes(profile, family=family)
    optimizer_maxiter = int(maxiter if maxiter is not None else (1 if family == "one_generation" else 2))

    results = []
    for population_size in population_sizes:
        run = None
        start = time.perf_counter()
        for _ in range(max(1, repeats)):
            run = _run_realistic_optimizer_trial(
                profile=profile,
                model_cls=model_cls,
                maxiter=optimizer_maxiter,
                tol=float(tol),
                polish=bool(polish),
                seed=seed,
                population_size=int(population_size),
                data_path=data_path,
                freeform_use_cupy=freeform_use_cupy,
            )
        elapsed = time.perf_counter() - start
        mean_elapsed = elapsed / max(1, repeats)

        results.append(
            {
                "population_size": int(population_size),
                "candidate_count_hint": int(profile_info["parameter_count"] * int(population_size)),
                "parameter_count": int(profile_info["parameter_count"]),
                "maxiter": int(optimizer_maxiter),
                "repeats": int(repeats),
                "elapsed_seconds": elapsed,
                "seconds_per_run": mean_elapsed,
                "gf": float(run["gf"]),
                "bic": float(run["bic"]),
            }
        )

    return {
        "profile": profile,
        "profile_spec": get_realistic_trapezoid_profile_spec(profile),
        "family": family,
        "model_class": model_cls.__name__,
        "freeform_use_cupy": bool(freeform_use_cupy),
        "results": results,
    }


def run_realistic_gpu_scaling_matrix(
    *,
    profiles: tuple[str, ...] = (
        "notebook_4param",
        "notebook_layer10_8param",
        "notebook_layer90_8param",
        "notebook_multilayer_12param",
    ),
    data_path: str | Path | None = None,
    candidate_repeats: int = 3,
    candidate_warmups: int = 1,
    optimizer_tol: float = 0.5,
    multi_generation_tol: float | None = 0.0,
    optimizer_polish: bool = False,
    optimizer_seed: int | None = 1234,
) -> dict:
    """Run the current fused GPU path across the realistic notebook-derived benchmark matrix."""
    matrix = {"model_class": TrapezoidModelArrayDeanGPUFused.__name__, "profiles": []}

    for profile in profiles:
        matrix["profiles"].append(
            {
                "description": describe_realistic_trapezoid_profile(profile=profile, data_path=data_path),
                "candidate_budget": run_realistic_candidate_budget_trial(
                    profile=profile,
                    model_cls=TrapezoidModelArrayDeanGPUFused,
                    repeats=candidate_repeats,
                    warmups=candidate_warmups,
                    data_path=data_path,
                    freeform_use_cupy=True,
                ),
                "one_generation": run_realistic_optimizer_profile(
                    profile=profile,
                    family="one_generation",
                    model_cls=TrapezoidModelArrayDeanGPUFused,
                    tol=optimizer_tol,
                    polish=optimizer_polish,
                    seed=optimizer_seed,
                    data_path=data_path,
                    freeform_use_cupy=True,
                ),
                "multi_generation": run_realistic_optimizer_profile(
                    profile=profile,
                    family="multi_generation",
                    model_cls=TrapezoidModelArrayDeanGPUFused,
                    tol=optimizer_tol if multi_generation_tol is None else float(multi_generation_tol),
                    polish=optimizer_polish,
                    seed=optimizer_seed,
                    data_path=data_path,
                    freeform_use_cupy=True,
                ),
            }
        )

    return matrix
