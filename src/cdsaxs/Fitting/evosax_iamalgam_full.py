"""evosax iAMaLGaM_Full runner for SiGe JAX objectives."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from evosax.algorithms.distribution_based.iamalgam_full import iAMaLGaM_Full

from .JAX_polish import polish_sige_jax_solution
from .SiGe_jax_model import (
    make_objective_batch,
    prepare_sige_jax_problem,
    unit_to_physical_np,
)


@dataclass
class EvosaxIAMaLGaMFullResult:
    x_unit: np.ndarray
    x: np.ndarray
    fun: float
    param_names: tuple[str, ...]
    model_params: dict
    history: list[dict]
    compile_seconds: float
    run_seconds: float
    nfev: int
    initial_seeded_fun: float
    converged: bool = False
    stop_reason: str = "max generations reached"
    generations_completed: int = 0
    polish_enabled: bool = False
    polish_accepted: bool = False
    polish_seconds: float = 0.0
    polish_nfev: int = 0
    polish_njev: int = 0
    polish_fun_before: float = np.nan
    polish_fun_after: float = np.nan
    polish_message: str = ""


def _population_converged(fitness, tol, atol):
    if tol is None:
        return False, np.nan, np.nan, np.nan

    fitness = np.asarray(jax.device_get(fitness), dtype=float)
    if np.any(~np.isfinite(fitness)):
        return False, np.nan, np.nan, np.nan

    population_std = float(np.std(fitness))
    population_mean = float(np.mean(fitness))
    threshold = float(atol + tol * abs(population_mean))
    return population_std <= threshold, population_std, population_mean, threshold


def run_sige_evosax_iamalgam_full(
    model,
    population_size=128,
    generations=50,
    seed=0,
    std_init=1.0,
    verbose=True,
    print_every=1,
    apply_result=True,
    tol=None,
    atol=0.0,
    polish=False,
    polish_maxiter=200,
    polish_ftol=1e-9,
    polish_gtol=1e-5,
):
    """Run evosax iAMaLGaM_Full on a prepared SiGe model in unit-cube space.

    ``generations`` follows the SciPy/DE-style maxiter convention used by the
    demo script: sampled iAMaLGaM populations are evaluated for generation
    indices 0 through ``generations``.
    """
    if population_size < 2:
        raise ValueError("iAMaLGaM_Full requires population_size >= 2")
    if generations < 0:
        raise ValueError("generations must be >= 0")
    if std_init < 0:
        raise ValueError("std_init must be >= 0")
    if tol is not None and tol < 0:
        raise ValueError("tol must be >= 0 or None")
    if atol < 0:
        raise ValueError("atol must be >= 0")

    problem = prepare_sige_jax_problem(model)
    objective_batch = make_objective_batch(problem)

    n_params = len(problem.param_names)
    default_unit = np.clip(
        (problem.default - problem.lower) / (problem.upper - problem.lower),
        0.0,
        1.0,
    ).astype(np.float32)

    key = jax.random.key(seed)
    algorithm = iAMaLGaM_Full(
        population_size=population_size,
        solution=jnp.zeros(n_params, dtype=jnp.float32),
        std_schedule=optax.constant_schedule(float(std_init)),
    )
    params = algorithm.default_params

    t0 = time.perf_counter()
    seeded_batch = jnp.asarray(default_unit[None, :], dtype=jnp.float32)
    seeded_fitness = objective_batch(seeded_batch).block_until_ready()
    state = algorithm.init(
        key,
        jnp.asarray(default_unit, dtype=jnp.float32),
        params,
    )
    state.mean.block_until_ready()
    compile_seconds = time.perf_counter() - t0

    initial_seeded_fun = float(seeded_fitness[0])
    best_unit = np.asarray(default_unit, dtype=np.float32)
    best_fun = initial_seeded_fun
    if verbose:
        print(f"Parameters: {n_params}")
        print(f"Initial seeded GF: {initial_seeded_fun:.6f}")
        print(f"Initial iAMaLGaM_Full std: {float(std_init):.6g}")

    history = []
    converged = False
    stop_reason = "max generations reached"
    generations_completed = -1
    t1 = time.perf_counter()
    for generation in range(0, generations + 1):
        key, ask_key, tell_key = jax.random.split(key, 3)
        candidates, state = algorithm.ask(ask_key, state, params)
        candidates = jnp.clip(candidates, 0.0, 1.0)
        fitness = objective_batch(candidates)
        state, _ = algorithm.tell(tell_key, candidates, fitness, state, params)
        state.best_fitness.block_until_ready()

        state_best_fun = float(state.best_fitness)
        if state_best_fun < best_fun:
            best_fun = state_best_fun
            best_unit = np.asarray(jnp.clip(state.best_solution, 0.0, 1.0))

        (
            converged,
            population_std,
            population_mean,
            convergence_threshold,
        ) = _population_converged(fitness, tol, atol)
        generations_completed = generation
        history.append(
            {
                "generation": generation,
                "best_fitness": best_fun,
                "state_best_fitness": state_best_fun,
                "std": float(state.std),
                "c_mult": float(state.c_mult),
                "nis_counter": int(state.nis_counter),
                "population_std": population_std,
                "population_mean": population_mean,
                "convergence_threshold": convergence_threshold,
                "converged": converged,
            }
        )
        if verbose and (generation % print_every == 0 or generation == generations):
            print(
                f"Generation {generation:04d}: "
                f"best GF {best_fun:.6f}, c_mult {float(state.c_mult):.6g}"
            )
        if converged:
            stop_reason = (
                "population fitness std <= "
                f"atol + tol * abs(mean) ({population_std:.6g} <= "
                f"{convergence_threshold:.6g})"
            )
            if verbose:
                print(f"Stopping at generation {generation:04d}: {stop_reason}")
            break

    jax.block_until_ready(state.best_fitness)
    run_seconds = time.perf_counter() - t1

    best_unit = np.asarray(np.clip(best_unit, 0.0, 1.0), dtype=np.float32)
    best_x = unit_to_physical_np(best_unit, problem.lower, problem.upper)
    nfev = 1 + population_size * (generations_completed + 1)
    polish_accepted = False
    polish_seconds = 0.0
    polish_nfev = 0
    polish_njev = 0
    polish_fun_before = np.nan
    polish_fun_after = np.nan
    polish_message = ""

    if polish:
        if verbose:
            print("Polishing best JAX iAMaLGaM_Full solution with bounded L-BFGS-B")
        polish_result = polish_sige_jax_solution(
            problem,
            best_unit,
            best_fun,
            maxiter=polish_maxiter,
            ftol=polish_ftol,
            gtol=polish_gtol,
        )
        best_unit = polish_result.x_unit
        best_x = polish_result.x
        best_fun = polish_result.fun
        polish_accepted = polish_result.accepted
        polish_seconds = polish_result.seconds
        polish_nfev = polish_result.nfev
        polish_njev = polish_result.njev
        polish_fun_before = polish_result.fun_before
        polish_fun_after = polish_result.fun_after
        polish_message = polish_result.message
        run_seconds += polish_seconds
        nfev += polish_nfev
        if verbose:
            status = "accepted" if polish_accepted else "rejected"
            print(
                "JAX polish "
                f"{status}: {polish_fun_before:.6f} -> {polish_fun_after:.6f} "
                f"({polish_seconds:.3f} s, nfev={polish_nfev}, njev={polish_njev})"
            )

    updated_model_params = copy.deepcopy(problem.base_model_params)
    if apply_result:
        updated_model_params = model._update_model_with_optimization_result(
            SimpleNamespace(x=best_x),
            list(problem.param_names),
            copy.deepcopy(problem.base_model_params),
        )
        model.model_params = updated_model_params
        model.update_traditional_from_model_params()
        model.SimInt = model.simulate_structure()
        model.GF = model.GF_calc(model.SimInt)
        model.BIC = model.BIC_calc(model.GF)

    return EvosaxIAMaLGaMFullResult(
        x_unit=best_unit,
        x=best_x,
        fun=best_fun,
        param_names=problem.param_names,
        model_params=updated_model_params,
        history=history,
        compile_seconds=compile_seconds,
        run_seconds=run_seconds,
        nfev=nfev,
        initial_seeded_fun=initial_seeded_fun,
        converged=converged,
        stop_reason=stop_reason,
        generations_completed=generations_completed,
        polish_enabled=bool(polish),
        polish_accepted=polish_accepted,
        polish_seconds=polish_seconds,
        polish_nfev=polish_nfev,
        polish_njev=polish_njev,
        polish_fun_before=polish_fun_before,
        polish_fun_after=polish_fun_after,
        polish_message=polish_message,
    )
