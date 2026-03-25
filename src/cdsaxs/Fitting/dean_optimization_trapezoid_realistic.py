from __future__ import annotations

import contextlib
import io
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
        "sequential": True,
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


def _format_layer_growth_path(spec: dict) -> str:
    builder = spec["builder"]
    if builder == "base":
        return "base"
    if builder == "single_layer":
        return f"add_layer_at_percentage({int(spec['height_percentage'])})"
    if builder == "multi_layer":
        percentages = ", ".join(str(int(v)) for v in spec["height_percentages"])
        sequential = bool(spec.get("sequential", False))
        return f"add_multiple_layers([{percentages}], sequential={sequential})"
    raise ValueError(f"Unsupported realistic trapezoid builder: {builder}")


@contextlib.contextmanager
def _suppress_workflow_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        yield


def _apply_realistic_trapezoid_builder(model, profile: str):
    """Apply one notebook-parallel layer-growth workflow to the selected model."""
    spec = get_realistic_trapezoid_profile_spec(profile)
    builder = spec["builder"]

    if builder == "base":
        return model
    if builder == "single_layer":
        return model.add_layer_at_percentage(int(spec["height_percentage"]))
    if builder == "multi_layer":
        grown_model = model.add_multiple_layers(
            list(spec["height_percentages"]),
            sequential=bool(spec.get("sequential", False)),
        )
        if isinstance(grown_model, list):  # pragma: no cover - defensive branch
            raise TypeError("Expected sequential realistic builder to return one model, not a list")
        return grown_model

    raise ValueError(f"Unsupported realistic trapezoid builder: {builder}")


def _normalize_notebook_parallel_optimization_surface(model, profile: str, optimization_margin: float = 0.2):
    """
    Keep the benchmark parameter surface parallel to the notebook workflows.

    The real layer-growth APIs may expand the internal optimization surface to
    per-column `Bk_i` parameters once imported data has initialized array
    backgrounds, and trapezoid initialization may also add `sld_*` parameters.
    The notebook-derived realistic harness intentionally keeps the smaller
    geometry-plus-global-state ladders used by the notebook workflows:

    - base profile: geometry plus `DW`
    - grown profiles: geometry plus `DW`, `I0`, and scalar `Bk`
    """
    spec = get_realistic_trapezoid_profile_spec(profile)
    if spec["builder"] == "base":
        return model

    param_limits = {}
    trapezoids = model.model_params["trapezoids"]
    for idx, trap in enumerate(trapezoids):
        width = float(trap["width"])
        param_limits[f"trap_{idx}_width"] = {
            "min": width * (1 - optimization_margin),
            "max": width * (1 + optimization_margin),
            "default": width,
        }
        if idx < len(trapezoids) - 1:
            height = float(trap["height"])
            param_limits[f"trap_{idx}_height"] = {
                "min": height * (1 - optimization_margin),
                "max": height * (1 + optimization_margin),
                "default": height,
            }

    source_opt = {}
    if hasattr(model, "_source_model") and hasattr(model._source_model, "model_params"):
        source_opt = model._source_model.model_params.get("optimization", {})

    def _inherit_or_margin(name: str, value: float) -> dict:
        if name in source_opt:
            inherited = dict(source_opt[name])
            inherited["default"] = value
            return inherited
        return {
            "min": value * (1 - optimization_margin),
            "max": value * (1 + optimization_margin),
            "default": value,
        }

    param_limits["DW"] = _inherit_or_margin("DW", float(model.DW))
    param_limits["I0"] = _inherit_or_margin("I0", float(model.I0))
    bk_scalar = float(model.Bk[0]) if isinstance(model.Bk, np.ndarray) else float(model.Bk)
    param_limits["Bk"] = _inherit_or_margin("Bk", bk_scalar)

    # Assign the notebook-parallel surface directly so trapezoid-specific
    # initialization does not auto-append `sld_*` parameters.
    model.model_params["optimization"] = param_limits
    model.param_names = list(param_limits.keys())

    expected_parameter_count = int(spec["expected_parameter_count"])
    if len(param_limits) != expected_parameter_count:  # pragma: no cover - defensive branch
        raise RuntimeError(
            f"Realistic profile {profile} expected {expected_parameter_count} optimization parameters, "
            f"but normalized to {len(param_limits)}: {tuple(param_limits)}"
        )
    return model


def build_realistic_trapezoid_model(
    *,
    profile: str = "notebook_4param",
    data_path: str | Path | None = None,
    model_cls=TrapezoidModelArrayDeanGPUFused,
    freeform_use_cupy: bool = False,
):
    """Build a notebook-derived trapezoid model by calling the public workflow APIs."""
    model = build_example_trapezoid_model(
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    model = _apply_realistic_trapezoid_builder(model, profile=profile)
    model = _normalize_notebook_parallel_optimization_surface(model, profile=profile)
    model._freeform_use_cupy = bool(freeform_use_cupy)
    return model


def _build_realistic_workflow_metadata(
    model,
    *,
    profile: str,
    freeform_use_cupy: bool,
    **workflow_knobs,
) -> dict:
    spec = get_realistic_trapezoid_profile_spec(profile)
    param_names, defaults = get_parameter_names_and_defaults(model)

    metadata = {
        "geometry": "trapezoid",
        "profile": profile,
        "label": spec["label"],
        "builder": spec["builder"],
        "layer_growth_path": _format_layer_growth_path(spec),
        "height_percentage": spec.get("height_percentage"),
        "height_percentages": tuple(spec.get("height_percentages", ())),
        "sequential": bool(spec.get("sequential", False)),
        "typed_layer_presence": False,
        "constraint_presence": False,
        "model_class": model.__class__.__name__,
        "layers": int(model.layers),
        "trapezoid_count": int(len(model.model_params["trapezoids"])),
        "parameter_count": int(len(param_names)),
        "parameter_names": tuple(param_names),
        "default_vector": np.asarray(defaults, dtype=float),
        "freeform_use_cupy": bool(freeform_use_cupy),
    }

    for key, value in workflow_knobs.items():
        if value is not None:
            metadata[key] = value

    return metadata


def describe_realistic_trapezoid_profile(
    *,
    profile: str = "notebook_4param",
    data_path: str | Path | None = None,
) -> dict:
    """Return a compact description of a realistic trapezoid benchmark profile."""
    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=TrapezoidModelArray,
    )
    return _build_realistic_workflow_metadata(
        model,
        profile=profile,
        freeform_use_cupy=False,
    )


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
    workflow = _build_realistic_workflow_metadata(
        model,
        profile=profile,
        freeform_use_cupy=freeform_use_cupy,
        workflow_api="objective_batch",
        optimizer="objective_only",
        vectorized=True,
    )

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
        "workflow": workflow,
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

    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    workflow = _build_realistic_workflow_metadata(
        model,
        profile=profile,
        freeform_use_cupy=freeform_use_cupy,
        workflow_api="CDSAXS_Optimize",
        optimizer="differential_evolution",
        vectorized=True,
        workers=None,
        tol=float(tol),
        polish=bool(polish),
        seed=seed,
    )
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
                "candidate_count_hint": int(workflow["parameter_count"] * int(population_size)),
                "parameter_count": int(workflow["parameter_count"]),
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
        "workflow": workflow,
        "results": results,
    }


def run_realistic_dual_annealing_trial(
    *,
    profile: str = "notebook_4param",
    model_cls=TrapezoidModelArray,
    maxiter: int = 5,
    initial_temp: float = 5000.0,
    no_local_search: bool = True,
    repeats: int = 1,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
) -> dict:
    """Run a short notebook-parallel dual-annealing workflow."""
    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    workflow = _build_realistic_workflow_metadata(
        model,
        profile=profile,
        freeform_use_cupy=freeform_use_cupy,
        workflow_api="CDSAXS_Optimize",
        optimizer="dual_annealing",
        vectorized=False,
        workers=None,
        maxiter=int(maxiter),
        initial_temp=float(initial_temp),
        no_local_search=bool(no_local_search),
    )

    optimizer_kwargs = {
        "maxiter": int(maxiter),
        "initial_temp": float(initial_temp),
        "no_local_search": bool(no_local_search),
    }

    result = None
    start = time.perf_counter()
    for _ in range(max(1, repeats)):
        result = model.CDSAXS_Optimize(
            optimizer="dual_annealing",
            use_callbacks=False,
            verbose=False,
            freeform_use_cupy=bool(freeform_use_cupy),
            plot_results=False,
            plot_structure=False,
            plot_grid=False,
            plot_combined=False,
            **optimizer_kwargs,
        )
    elapsed = time.perf_counter() - start

    return {
        "profile": profile,
        "profile_spec": get_realistic_trapezoid_profile_spec(profile),
        "family": "dual_annealing",
        "model_class": model_cls.__name__,
        "freeform_use_cupy": bool(freeform_use_cupy),
        "workflow": workflow,
        "results": [
            {
                "repeats": int(repeats),
                "elapsed_seconds": elapsed,
                "seconds_per_run": elapsed / max(1, repeats),
                "gf": float(getattr(model, "GF", np.inf)),
                "bic": float(getattr(model, "BIC", np.inf)),
                "result": result,
            }
        ],
    }


def run_realistic_batch_initialize_smoke(
    *,
    profile: str = "notebook_4param",
    model_cls=TrapezoidModelArray,
    initialization_params: dict | None = None,
    n_points_per_param: int = 2,
    max_fits: int = 2,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
    optimization_kwargs: dict | None = None,
) -> dict:
    """Run a tiny batch-initialization workflow on a notebook-parallel model."""
    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    param_names, defaults = get_parameter_names_and_defaults(model)
    init_params = initialization_params or {
        param_names[0]: {
            "min": float(defaults[0] * 0.98),
            "max": float(defaults[0] * 1.02),
            "n_points": int(n_points_per_param),
        }
    }
    workflow = _build_realistic_workflow_metadata(
        model,
        profile=profile,
        freeform_use_cupy=freeform_use_cupy,
        workflow_api="batch_initialize_and_fit",
        optimizer="differential_evolution",
        vectorized=False,
        workers=None,
        initialization_params=tuple(init_params.keys()),
        n_points_per_param=int(n_points_per_param),
        max_fits=int(max_fits),
    )
    optimization_kwargs = optimization_kwargs or {
        "maxiter": 1,
        "popsize": 2,
        "polish": False,
        "plot_results": False,
        "verbose": False,
    }

    with _suppress_workflow_output():
        start = time.perf_counter()
        results = model.batch_initialize_and_fit(
            initialization_params=init_params,
            n_points_per_param=int(n_points_per_param),
            optimization_kwargs=optimization_kwargs,
            max_fits=int(max_fits),
            verbose=False,
            save_results=False,
        )
        elapsed = time.perf_counter() - start

    fits = list(results.get("fits", []))
    converged = [fit for fit in fits if fit.get("converged")]
    best_fit = fits[0] if fits else None

    return {
        "profile": profile,
        "profile_spec": get_realistic_trapezoid_profile_spec(profile),
        "family": "batch_initialize_and_fit_smoke",
        "model_class": model_cls.__name__,
        "freeform_use_cupy": bool(freeform_use_cupy),
        "workflow": workflow,
        "results": [
            {
                "elapsed_seconds": elapsed,
                "n_total_fits": int(results.get("n_total_fits", 0)),
                "n_recorded_fits": int(len(fits)),
                "n_converged_fits": int(len(converged)),
                "best_gf": float(best_fit["gf"]) if best_fit is not None else float("inf"),
                "best_bic": float(best_fit["bic"]) if best_fit is not None else float("inf"),
            }
        ],
        "batch_results": results,
    }


def run_realistic_mcmc_profile(
    *,
    profile: str = "notebook_4param",
    family: str = "mcmc_smoke",
    model_cls=TrapezoidModelArray,
    n_walkers: int | None = None,
    n_steps: int | None = None,
    burn_in: int | None = None,
    thin: int | None = None,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
) -> dict:
    """Run a short MCMC workflow on a notebook-parallel model."""
    if family not in {"mcmc_smoke", "mcmc_trial"}:
        raise ValueError(f"Unknown realistic MCMC family: {family}")

    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    param_count = int(len(get_parameter_names_and_defaults(model)[0]))
    default_walkers = max(2 * param_count + 2, 10 if family == "mcmc_smoke" else 18)
    default_steps = 20 if family == "mcmc_smoke" else 60
    default_burn_in = 5 if family == "mcmc_smoke" else 10
    default_thin = 2 if family == "mcmc_smoke" else 5

    walkers = int(n_walkers if n_walkers is not None else default_walkers)
    steps = int(n_steps if n_steps is not None else default_steps)
    burn = int(burn_in if burn_in is not None else default_burn_in)
    thin_value = int(thin if thin is not None else default_thin)

    workflow = _build_realistic_workflow_metadata(
        model,
        profile=profile,
        freeform_use_cupy=freeform_use_cupy,
        workflow_api="CDSAXS_MCMC",
        optimizer="mcmc",
        vectorized=False,
        workers=None,
        n_walkers=walkers,
        n_steps=steps,
        burn_in=burn,
        thin=thin_value,
    )

    with _suppress_workflow_output():
        start = time.perf_counter()
        mcmc_results = model.CDSAXS_MCMC(
            n_walkers=walkers,
            n_steps=steps,
            burn_in=burn,
            thin=thin_value,
            progress=False,
            plot_results=False,
            verbose=False,
        )
        elapsed = time.perf_counter() - start

    if mcmc_results is None:
        row = {
            "elapsed_seconds": elapsed,
            "effective_samples": 0,
            "mean_acceptance": float("nan"),
            "gf": float("inf"),
            "bic": float("inf"),
        }
    else:
        row = {
            "elapsed_seconds": elapsed,
            "effective_samples": int(mcmc_results["effective_samples"]),
            "mean_acceptance": float(mcmc_results["mean_acceptance"]),
            "gf": float(getattr(model, "GF", np.inf)),
            "bic": float(getattr(model, "BIC", np.inf)),
        }

    return {
        "profile": profile,
        "profile_spec": get_realistic_trapezoid_profile_spec(profile),
        "family": family,
        "model_class": model_cls.__name__,
        "freeform_use_cupy": bool(freeform_use_cupy),
        "workflow": workflow,
        "results": [row],
        "mcmc_results": mcmc_results,
    }


def run_realistic_sweep_replay_smoke(
    *,
    profile: str = "notebook_4param",
    sweep_kind: str = "1d",
    criterion: str = "GF",
    run_optimization: bool = False,
    model_cls=TrapezoidModelArray,
    data_path: str | Path | None = None,
    freeform_use_cupy: bool = False,
) -> dict:
    """Run a tiny sweep followed by `show_best_fit_results(...)` replay."""
    if sweep_kind not in {"1d", "2d"}:
        raise ValueError(f"Unknown realistic sweep kind: {sweep_kind}")

    model = build_realistic_trapezoid_model(
        profile=profile,
        data_path=data_path,
        model_cls=model_cls,
        freeform_use_cupy=freeform_use_cupy,
    )
    param_names, defaults = get_parameter_names_and_defaults(model)
    workflow = _build_realistic_workflow_metadata(
        model,
        profile=profile,
        freeform_use_cupy=freeform_use_cupy,
        workflow_api="show_best_fit_results",
        optimizer="differential_evolution" if run_optimization else "best_fit_replay_only",
        vectorized=False,
        workers=None,
        criterion=str(criterion),
        run_optimization=bool(run_optimization),
        sweep_kind=str(sweep_kind),
    )

    with _suppress_workflow_output():
        if sweep_kind == "1d":
            sweep_results = model.parameter_sweep_1d(
                sweep_param=param_names[0],
                sweep_range=(float(defaults[0] * 0.98), float(defaults[0] * 1.02)),
                n_points=3,
                plot_results=False,
                verbose=False,
                optimization_kwargs={"maxiter": 1, "popsize": 2, "polish": False},
            )
        else:
            sweep_results = model.parameter_sweep_2d(
                (param_names[0], param_names[-1]),
                (
                    (float(defaults[0] * 0.98), float(defaults[0] * 1.02)),
                    (float(defaults[-1] * 0.98), float(defaults[-1] * 1.02)),
                ),
                (2, 2),
                plot_results=False,
                verbose=False,
                optimization_kwargs={"maxiter": 1, "popsize": 2, "polish": False},
            )

        start = time.perf_counter()
        model.show_best_fit_results(
            sweep_results,
            criterion=str(criterion),
            run_optimization=bool(run_optimization),
        )
        elapsed = time.perf_counter() - start

    return {
        "profile": profile,
        "profile_spec": get_realistic_trapezoid_profile_spec(profile),
        "family": "sweep_replay_smoke",
        "model_class": model_cls.__name__,
        "freeform_use_cupy": bool(freeform_use_cupy),
        "workflow": workflow,
        "results": [
            {
                "elapsed_seconds": elapsed,
                "gf": float(getattr(model, "GF", np.inf)),
                "bic": float(getattr(model, "BIC", np.inf)),
                "sweep_kind": str(sweep_kind),
            }
        ],
        "sweep_results": sweep_results,
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
