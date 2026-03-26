# Trapezoid Optimization Status

## Purpose

This page records the current kept trapezoid optimization state and identifies what is reusable for future geometry work, especially SiGe.

## Status As Of 2026-03-26

Current practical baselines:

- realistic vectorized CPU baseline: `TrapezoidModelArray`
- current kept optimized GPU path: `TrapezoidModelArrayDeanGPUFused`
- scalar accelerated CPU reference for some small workloads: `AcceleratedTrapezoidModel`

Do not use as a performance baseline:

- the base `freeform_use_cupy=True` path inside `Trapezoid_model.py`

Rationale:

- it can look artificially fast in some candidate-budget runs
- it does not currently provide a reliable parity baseline
- the fused resident GPU path is the actual kept implementation for realistic GPU work

## Current Reusable Assets

The trapezoid work already established patterns that are reusable for other geometries.

Directly reusable patterns:

- active parameter-name resolution through `CDSAXS_Model._resolve_active_parameter_names(...)`
- optimizer-facing 1D and 2D objective semantics
- realistic harness structure for raw objective timing versus short optimizer timing
- dataset invariant caching such as `log(Intensity)` and `Qx**2 + Qz**2`
- cached parameter-plan compilation
- candidate-matrix normalization for `(S, D)` and `(D, S)` inputs
- GPU-resident dataset caches
- fused batched residual reduction on GPU

Best reference files:

- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py`

## What Is Not Reusable As-Is

These parts should not be copied blindly into SiGe:

- trapezoid-specific direct `PAR` mapping
- trapezoid `SymCoordAssign(...)` assumptions
- trapezoid coordinate schema and layer contribution math
- trapezoid GPU form-factor kernels
- trapezoid-specific parity tests that assume simple `PAR` updates are enough

Reason:

- SiGe has design-level parameter mapping, typed layers, constraints, and richer coordinate requirements
- the geometry preprocessing burden is materially different before form-factor evaluation even begins

## Reuse Guidance For SiGe

Treat trapezoid as the reference for optimization architecture, not as a drop-in SiGe implementation.

Recommended reuse boundary:

- reuse the optimizer contract
- reuse the caching patterns
- reuse the realistic harness shape
- reuse the CPU-versus-batch and CPU-versus-GPU test strategy
- do not force shared geometry kernels prematurely

## Naming Transition

Future trapezoid generalization work should follow the same neutral naming policy planned for SiGe:

- `_vectorized`
- `_vectorized_GPU`
- `_GPU`

Legacy `Dean` names should remain in place for existing code until there is an explicit migration plan.

## Resume Notes

If a future task depends on the current trapezoid implementation, establish these facts first:

1. `TrapezoidModelArray` is the realistic vectorized CPU baseline.
2. `TrapezoidModelArrayDeanGPUFused` is the kept optimized realistic GPU path.
3. The realistic harness and test suite already exist and should be reused as the model for future geometry harness work.
4. Any new general optimization abstractions should be extracted only after they serve SiGe or another second geometry cleanly.
