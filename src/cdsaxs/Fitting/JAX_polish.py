"""Optional local polishing for JAX-backed SiGe optimization results."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .SiGe_jax_model import (
    _problem_to_jax_arrays,
    sige_objective_jax,
    unit_to_physical_jax,
    unit_to_physical_np,
)


@dataclass
class JAXPolishResult:
    x_unit: np.ndarray
    x: np.ndarray
    fun: float
    accepted: bool
    seconds: float
    nfev: int
    njev: int
    message: str
    fun_before: float
    fun_after: float


def polish_sige_jax_solution(
    problem,
    x_unit_start,
    fun_start,
    method="L-BFGS-B",
    maxiter=200,
    ftol=1e-9,
    gtol=1e-5,
):
    """Polish a normalized SiGe JAX solution with a bounded local optimizer.

    The local optimizer works in unit-cube coordinates so parameters with very
    different physical scales remain reasonably conditioned.
    """
    x_unit_start = np.clip(np.asarray(x_unit_start, dtype=np.float64), 0.0, 1.0)
    fun_start = float(fun_start)
    lower = jnp.asarray(problem.lower, dtype=jnp.float32)
    upper = jnp.asarray(problem.upper, dtype=jnp.float32)
    arrays = _problem_to_jax_arrays(problem)

    def objective_one(x_unit):
        x_phys = unit_to_physical_jax(x_unit, lower, upper)
        return sige_objective_jax(x_phys, arrays)

    value_and_grad = jax.jit(jax.value_and_grad(objective_one))

    def scipy_fun_and_jac(x_unit_np):
        value, grad = value_and_grad(jnp.asarray(x_unit_np, dtype=jnp.float32))
        return float(value), np.asarray(grad, dtype=np.float64)

    start = time.perf_counter()
    result = minimize(
        scipy_fun_and_jac,
        x_unit_start,
        method=method,
        jac=True,
        bounds=[(0.0, 1.0)] * len(x_unit_start),
        options={
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
        },
    )
    seconds = time.perf_counter() - start

    polished_unit = np.clip(np.asarray(result.x, dtype=np.float32), 0.0, 1.0)
    polished_fun = float(result.fun)
    accepted = bool(np.isfinite(polished_fun) and polished_fun < fun_start)
    final_unit = polished_unit if accepted else x_unit_start.astype(np.float32)
    final_fun = polished_fun if accepted else fun_start
    final_x = unit_to_physical_np(final_unit, problem.lower, problem.upper)

    return JAXPolishResult(
        x_unit=final_unit,
        x=final_x,
        fun=final_fun,
        accepted=accepted,
        seconds=seconds,
        nfev=int(getattr(result, "nfev", 0) or 0),
        njev=int(getattr(result, "njev", 0) or 0),
        message=str(getattr(result, "message", "")),
        fun_before=fun_start,
        fun_after=polished_fun,
    )
