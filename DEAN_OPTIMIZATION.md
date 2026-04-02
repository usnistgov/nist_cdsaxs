# DEAN Optimization Action Plan

## Purpose

This file is now the landing page for optimization work on this branch.

Detailed geometry-specific plans live under `docs/optimization/`.
The dated experimental record remains in `DEAN_OPTIMIZATION_LEDGER.MD`.

## Documentation Map

- `docs/optimization/README.md`
  - doc index
  - naming rules
  - resume checklist
- `docs/optimization/trapezoid.md`
  - current trapezoid status
  - reusable optimization patterns
  - current kept CPU and GPU conclusions
- `docs/optimization/sige.md`
  - SiGe reuse appraisal
  - detailed CPU vectorization plan
  - follow-on GPU plan
  - SiGe-specific test and harness plan
- `DEAN_OPTIMIZATION_LEDGER.MD`
  - dated measurements
  - experiment outcomes
  - planning checkpoints that materially change execution

## Active Geometry Tracks

### Trapezoid

Current status:

- realistic vectorized CPU baseline remains `TrapezoidModelArray`
- current kept optimized GPU path remains `TrapezoidModelArrayDeanGPUFused`
- the base `freeform_use_cupy=True` path in `Trapezoid_model.py` is not a valid performance baseline

Current role in the roadmap:

- keep the validated trapezoid path as the reference implementation for vectorized optimizer semantics
- reuse its harness, caching, and GPU-resident reduction patterns where they generalize cleanly
- do not force SiGe to share trapezoid-specific geometry math

See `docs/optimization/trapezoid.md`.

### SiGe

Current status:

- realistic SiGe workflows now have a package-native `vectorized=True` ellipse-stack path in the repo
- the current SiGe batched objective path is functionally correct but still about parity-to-slower than the scalar CPU path
- a naive full layer-axis NumPy broadcast rewrite of `SiGeModel_vectorized._batched_form_factor(...)` was tried and rejected
- remaining CPU speedup options are now documented, but the next active milestone is GPU vectorization of the realistic SiGe batched path
- the full `(candidate, layer, qz, qx)` layer reduction should be revisited on GPU rather than on CPU
- GPU work is now the active next milestone for SiGe because the current CPU vectorized path is correct, profiled, and no longer a promising place for a naive 4D NumPy reduction

Current role in the roadmap:

- establish a realistic package-native SiGe harness
- refactor scalar SiGe objective logic into resumable helpers
- keep the documented CPU speedup options available for later follow-up if needed
- move the next active implementation pass to a GPU-vectorized SiGe objective
- then move the dominant batched form-factor and GF work onto GPU

See `docs/optimization/sige.md`.

## Current Branch Context

- repo: `/homes/deand/dev/nist_cdsaxs`
- working branch: `feature/dean_optimization`
- environment: `cdsaxs-dev`
- install mode: editable
- GPU package support used so far: `cupy-cuda12x`
- current non-doc repo fix already applied: `src/cdsaxs/Fitting/SRM_model.py` now prefers package-relative import of `CDSAXS_base_model`

## Naming Convention Going Forward

New optimization work should stop introducing `Dean` in public-facing names.

Use these suffixes instead:

- `_vectorized`
- `_vectorized_GPU`
- `_GPU`

Guidance:

- keep existing `Dean` modules and class names as historical artifacts until code migration is worth doing
- use neutral names for all new SiGe modules, helpers, tests, and future trapezoid generalizations
- preserve the existing public geometry aliases unless there is a deliberate compatibility migration

## Resume Checklist

If resuming from a fresh context:

1. Read `docs/optimization/sige.md` for the active SiGe implementation plan.
2. Read `docs/optimization/trapezoid.md` for the reusable patterns and current trapezoid baseline.
3. Use `DEAN_OPTIMIZATION_LEDGER.MD` only for dated outcomes, not for the current roadmap.
4. Work inside `cdsaxs-dev`.
5. Use the realistic SiGe notebook and reduced CSV as the reference workflow while keeping new harness code package-native.

## Non-Negotiable Principles

1. Preserve the fitting workflow shape.
   The optimized path should still look like:
   - construct model
   - import data
   - call `CDSAXS_Optimize(...)`
   - inspect `model.GF`, `model.BIC`, and optimized parameters

2. Preserve structural parallelism with the repo.
   Optimized variants should live beside the current geometry modules.
   Benchmark helpers should live inside `src/cdsaxs/Fitting/`.
   Tests should live inside `tests/test_fitting/`.

3. Keep user-facing naming parallel to current conventions.
   Use geometry-first names.
   Prefer the repo's public aliases like `TrapezoidModel` and `SiGeModel` in user-facing examples.
   Reserve implementation suffixes like `_vectorized` and `_vectorized_GPU` for explicit optimized variants.

4. Separate algorithm families.
   Objective microbenchmarks, DE timing, batch initialization, and MCMC should not be mixed into one number.

5. Separate incumbents by workload shape.
   There is not one universal incumbent anymore.

6. Keep scientific gating proportional to value.
   Validate speedup first.
   Tighten structure-similarity and stochastic regression only for candidates worth keeping.
