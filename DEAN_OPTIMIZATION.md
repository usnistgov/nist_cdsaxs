# DEAN Optimization Action Plan

## Goal

Improve fitting throughput in `nist_cdsaxs`, starting with trapezoid fitting, while preserving:

- Nearly identical user-facing APIs.
- Repo structure that parallels the current fitting package.
- Scientific comparability, especially structure similarity and fit-quality stability.

This plan assumes large-memory GPU systems are a target platform, but it stages work so each phase is measurable on CPU first.

## Current State

Artifacts added so far:

- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`

Artifacts now added for the first optimization pass:

- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`

Working branch and environment:

- branch: `feature/dean_optimization`
- mamba env: `cdsax-dev`
- install mode: editable
- GPU package support now added:
  - `cupy-cuda12x`

Guidance incorporated from the current round:

- preserve parallelism with the current repo layout and naming conventions
- grow the timing harness to study larger broadcast-style batches in a principled way
- add a structure-similarity regression scaffold, but only enforce it after a meaningful speedup candidate appears
- maintain a detailed experiment ledger separate from the roadmap
- proceed sequentially, and stop to reassess if an optimization only moves overhead around without improving realistic fitting runs

## Measured Readout So Far

Stage 1 has now been exercised once with the first Dean trapezoid variant in
`src/cdsaxs/Fitting/Trapezoid_model_dean.py`.

What was added:

- cached `log(Intensity)` for GF evaluation
- cached `Qx**2 + Qz**2` for Debye-Waller reuse
- cached parameter-name parsing into a reusable plan
- reduced temporary array and object churn in scalar and batched trapezoid objective paths
- benchmark helpers for:
  - named objective batch profiles
  - scaling comparisons across increasing batch sizes
  - lightweight structure-signature extraction

Measured outcome on the DeRocher V3 example:

- tiny DE, scalar objective path: about `1.14x` speedup
- tiny DE, vectorized objective path: about `1.16x` speedup
- objective-only batched timing:
  - peak win around batch size `64`: about `1.42x`
  - moderate win around batch size `256`: about `1.25x`
  - small win at batch size `1024` and `4096`: about `1.04x` to `1.08x`
- fixed-seed tiny structure trial produced identical widths, heights, `DW`, `GF`, and `BIC`

Interpretation:

- the first optimization is worth keeping as a cleaner baseline
- it is not yet the GPU-oriented breakthrough
- by batch size `1024+`, the expensive form-factor math dominates and the cached Python-side cleanup no longer moves the needle much
- this argues for a second-stage optimization that changes the math-kernel cost model or device residency, not just more Python cleanup

Additional environment readout:

- this workspace exposes three `Quadro RTX 8000` GPUs with `49152 MiB` each
- `cupy` is now installed in `cdsax-dev` via `cupy-cuda12x`
- the existing CuPy hooks in the code are not yet truly device-resident because the current form-factor path converts results back to NumPy before returning

## Latest Results

Two additional conclusions are now measured:

1. Component profiling of the current Dean CPU path
   - `FreeFormTrapezoid` accounts for about `85%` to `88%` of vectorized objective time at realistic batch sizes
   - `GF_calc` is only about `6%` to `8%`
   - `SymCoordAssign` is negligible in the current example

2. GPU status after trying the next candidates
   - the legacy-style CuPy hook in the baseline trapezoid model is not a keep candidate
   - it is not device-resident and, in the tested path, returns `inf` objective values because the old wrapper expects NumPy arrays
   - a new Dean GPU-resident trapezoid variant does produce a real win:
     - objective-only speedup at medium and large batch sizes: about `4.9x` to `8.2x`
     - vectorized tiny DE speedup: about `4.1x`
     - fixed-seed structure, `GF`, and `BIC` matched exactly in the lightweight regression
   - a fused refinement of that GPU path is now the leading branch:
     - objective speed is broadly similar and sometimes better
     - tiny vectorized DE improved again by about `1.29x` relative to the first GPU variant
     - fit parity remained exact in the tested regression

Interpretation:

- the GPU path is now justified for batched workflows
- the scalar path is still not a GPU win, so the accelerated path should remain explicitly batch-oriented
- the next optimization work should stay focused on vectorized / broadcasted fitting rather than scalar GPU use

## Current Read On The Code

- The main optimization entry point is `CDSAXS_Optimize` in `src/cdsaxs/Fitting/CDSAXS_base_model.py`.
- Trapezoid fitting already has a batched objective path for `scipy.optimize.differential_evolution(vectorized=True)`.
- Cylinder fitting is still mostly scalar and loop-heavy.
- The example trapezoid workflow in `examples/fitting_examples/CDSAXS_DeRocher_V3.ipynb` is the right baseline because it is small, already timed in the notebook, and uses the current public model-building pattern.
- Recorded notebook timings show that the current "GPU broadcast" path is not yet a win, so the first task is not "turn on GPU", it is "make the batched path principled and measurable".

## Working Principles

1. Preserve the public fitting workflow.
   New tools should still look like:
   - construct model
   - import data
   - call `CDSAXS_Optimize(...)`
   - inspect `model.GF`, `model.BIC`, and optimized parameters

2. Keep optimized variants structurally parallel to the existing package.
   Preferred placement:
   - optimized trapezoid variants beside `Trapezoid_model.py`
   - optimized cylinder variants beside `Cylinder_model.py`
   - shared benchmark helpers inside `src/cdsaxs/Fitting/`
   - test harnesses inside `tests/`

3. Optimize only against a fixed scientific benchmark.
   Each stage must check:
   - wall time
   - number of objective evaluations when available
   - final GF / BIC
   - parameter drift
   - derived structure similarity

4. Separate "hot-loop speed" from "optimizer behavior".
   We need both:
   - objective-level benchmarks
   - lightweight end-to-end optimizer benchmarks

5. Gate scientific-proof work appropriately.
   We will:
   - validate speedup first
   - only tighten structure-similarity proof once the candidate is worth keeping
   - avoid over-investing in stochastic regression machinery for candidates that do not materially improve speed

## Staged Plan

### Stage 0: Baseline And Measurement

Purpose:
- Establish a reproducible trapezoid benchmark that mirrors the example notebook.

Actions:
- Use the DeRocher V3 trapezoid example data and model setup.
- Add a small helper module in `src/cdsaxs/Fitting/` that builds the example model with current APIs.
- Add a `pytest` harness with:
  - scalar objective smoke test
  - batched objective smoke test
  - tiny end-to-end DE run with reduced `maxiter` / `popsize`
  - named batch-size profiles that separate smoke, broadcast, and GPU-candidate studies

Deliverables:
- `DEAN_OPTIMIZATION.md`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`

Acceptance:
- Baseline run succeeds in the new env.
- The harness finishes fast enough to run repeatedly during development.
- Benchmark helpers support both lightweight smoke profiles and larger broadcast-oriented profiles.
- Larger profiles remain outside routine CI expectations and are used as deliberate benchmark experiments.

### Stage 1: CPU Hot-Loop Cleanup

Purpose:
- Remove avoidable Python and allocation overhead before introducing new math kernels.

Actions:
- Precompute immutable dataset terms such as:
  - `log(Intensity)`
  - `Qx**2 + Qz**2`
- Reduce repeated dictionary copying and parameter-name parsing in trapezoid objective code.
- Reduce temporary array churn in batched trapezoid objective evaluation.
- Remove debugging prints from optimization hot paths.
- Keep the first optimization in a parallel Dean-style trapezoid variant rather than directly rewriting the baseline model.

Expected benefit:
- Better scalar speed.
- Better batched CPU speed.
- Cleaner reference implementation before GPU work.

Acceptance:
- Objective-level benchmark shows a measurable reduction in wall time.
- End-to-end tiny DE benchmark improves or stays flat with no scientific regressions.
- If no meaningful speedup is observed, stop and reconsider before continuing with later stages.

Current status:

- complete enough to keep
- measured win is real but modest
- not sufficient on its own to justify heavier scientific gating or broader refactors

### Stage 2: Trapezoid Batched CPU Path

Purpose:
- Make the current vectorized SciPy path a strong CPU baseline.

Actions:
- Treat candidate matrices as first-class inputs.
- Reuse work buffers where possible.
- Keep a single parameter-mapping implementation for scalar and batched paths.
- Validate shape handling and background handling on both scalar and batched inputs.
- Profile where the batched path stops scaling:
  - `SymCoordAssign`
  - `FreeFormTrapezoid`
  - Debye-Waller application
  - GF calculation
- If needed, split "broadcasting many candidates" from "computing one candidate efficiently" so both are benchmarked explicitly.

Expected benefit:
- Better scaling with `vectorized=True`.
- Clearer comparison point for future GPU work.

Acceptance:
- Batched objective benchmark beats repeated scalar calls at realistic batch sizes.
- Tiny DE vectorized run is competitive with or better than worker-based CPU runs.
- Larger batch-size timing profiles reveal whether more aggressive broadcasting is likely to pay off on GPU later.

### Stage 3: Trapezoid GPU-Resident Path

Purpose:
- Make large-population fitting benefit from large-memory GPUs without changing the public workflow.

Actions:
- Keep `Intensity`, `Qx`, `Qz`, and reusable buffers on device for the entire optimization.
- Avoid host-device round trips inside each objective call.
- Return only the objective vector to SciPy.
- If needed, add a parallel class or module variant in `src/cdsaxs/Fitting/` rather than creating a separate API family.
- Use the larger broadcast timing profiles from Stage 0/2 to choose realistic device batch sizes rather than guessing.
- Benchmark transfer cost separately from math throughput so the GPU path is only judged on device-resident work once setup is amortized.

Expected benefit:
- GPU path becomes viable for large trial populations instead of being slower than CPU due to transfer overhead.

Acceptance:
- For large enough populations, device-resident batched evaluation is faster than Stage 2 CPU batched evaluation.
- Final parameters and GF remain scientifically comparable.

### Stage 4: Trapezoid Compiled Kernels

Purpose:
- Decide whether Cython / compiled kernels should back the optimized trapezoid path.

Actions:
- Compare:
  - cleaned Python/NumPy batched path
  - current Cython-enabled path
  - GPU-resident batched path
- Only adopt compiled code where it changes the measured bottleneck.

Acceptance:
- Keep the smallest implementation that wins clearly.
- Do not keep extra compiled complexity if the measured gain is marginal.

### Stage 5: Cylinder And Workflow-Level Expansion

Purpose:
- Extend the same methodology after trapezoid is under control.

Actions:
- Add cylinder objective benchmark and tiny optimizer benchmark.
- Prioritize compiled or batched cylinder kernels because the current path is more loop-heavy.
- Revisit `parameter_sweep_*` and `batch_initialize_and_fit` so they can reuse the same benchmarked objective machinery.

Acceptance:
- Cylinder work follows the same benchmark discipline as trapezoid.

## Full Optimization Inventory

### High-Yield Candidates

- CPU hot-loop cleanup for trapezoid objective evaluation.
- Stronger batched CPU trapezoid path for `vectorized=True`.
- GPU-resident trapezoid candidate evaluation with reusable device buffers.
- Compiled trapezoid kernels only if they beat the cleaned batched CPU path.
- Batched or compiled cylinder objective path.

### Medium-Yield Candidates

- Route accelerated models into real workflows more directly so speedup paths are easier to exercise.
- Precompute more invariants once per dataset, including log-intensity and repeated Q-space terms.
- Remove unconditional debug output from optimizer hot paths.
- Add lightweight profiling helpers that attribute time to:
  - parameter mapping
  - coordinate construction
  - form-factor evaluation
  - GF calculation
- Reuse allocated candidate and work buffers across repeated objective calls when the optimizer shape is stable.
- Fuse simple elementwise operations where practical, especially:
  - Debye-Waller factor application
  - magnitude-square-plus-background construction
  - log-space GF comparison
- Avoid repeated transposes or layout fixes in vectorized paths by standardizing one internal candidate layout.
- Audit CuPy and Cython fallbacks so acceleration paths do not silently fall back in a way that hides performance regressions.
- Add offline benchmark scripts or pytest markers for large-batch runs on real GPU hosts.
- Use hybrid optimization strategies:
  - coarse global search
  - then local refinement
- Add coarse prescreening for parameter sweeps and batch initialization workflows before full optimization.
- Reuse optimization machinery across `parameter_sweep_*` and `batch_initialize_and_fit`.
- Add offline benchmark modes that study scaling versus population size and batch size.
- Improve optional dependency and environment support for acceleration-oriented paths.

## Priority Order To Try

This is the current execution order, combining the earlier roadmap with the deeper read on the bottlenecks and the available `48 GB` GPUs.

1. True device-resident trapezoid objective
   - keep `Intensity`, `Qx`, `Qz`, cached invariants, and large work arrays on GPU
   - return only the final objective vector to the CPU-side optimizer
   - use a Dean-style parallel variant beside the current trapezoid model

2. GPU-native batched `FreeFormTrapezoid`
   - redesign the batched trapezoid kernel around device arrays instead of wrapping the NumPy-oriented broadcast path
   - remove forced `.get()` behavior from the internal hot path

3. Persistent GPU workspaces
   - preallocate and reuse candidate, coordinate, Debye-Waller, simulated-intensity, and GF scratch arrays
   - target the large-batch flattening seen in the first benchmark pass

4. Fused post-form-factor batched path
   - fuse Debye-Waller application, magnitude-square, background addition, and log-space GF accumulation where practical
   - reduce temporary-array traffic on both CPU and GPU

5. Stronger batched CPU path
   - profile and optimize `SymCoordAssign`, `FreeFormTrapezoid`, and GF separately
   - standardize candidate layout and avoid repeated transpose / layout repair
   - keep this as the CPU baseline even if GPU work wins

6. Optimizer strategy aligned with large batches
   - keep the public API nearly identical
   - continue using the current SciPy-facing flow where possible
   - if needed later, introduce an internal ask/tell style or staged global-plus-local strategy that naturally emits large candidate batches

7. Compiled trapezoid kernels
   - compare only after the GPU-native and stronger batched CPU paths are measured
   - keep compiled complexity only if it wins clearly

8. Workflow-level propagation
   - route winning trapezoid paths into `parameter_sweep_*` and `batch_initialize_and_fit`
   - extend the same method to cylinder once trapezoid is stable

9. Structure-similarity gating and broader regression hardening
   - tighten only for candidates that materially improve speed
   - keep early-stage scientific checks lightweight but real

### Lower-Priority Or Dependent Work

- Device-specific tuning once a working GPU-resident path exists.
- Strong structure-similarity gating after a speedup candidate is identified.
- Broader workflow refactors that are not on the critical path to faster trapezoid fitting.

## What To Measure At Every Stage

### Performance

- Objective wall time:
  - scalar candidate
  - batched candidate list
- Objective throughput:
  - seconds per call
  - seconds per candidate
- objective scaling across batch profiles:
  - smoke
  - broadcast
  - GPU-candidate
- End-to-end optimizer wall time:
  - tiny DE run
  - optionally larger comparison run outside the default test suite
- Optional memory counters for large-batch experiments

### Scientific Stability

- Final GF
- Final BIC
- Final optimized parameter vector
- Derived geometry comparison
- Structure similarity summary

For structure similarity, we should not rely only on GF. We should compare the optimized geometry explicitly, especially widths and heights per layer, because a faster optimizer is not useful if it arrives at materially different structures under the same setup.

Policy:

- Before speedup exists:
  - keep structure-similarity scaffolding available
  - do not block progress on stochastic equivalence proof
- After speedup exists:
  - compare optimized structures with fixed seeds where possible
  - add tolerances for widths, heights, DW, and background summaries

## Proposed File Layout

### Added Now

- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`

### Likely Next Files

- `src/cdsaxs/Fitting/Cylinder_model_dean.py`
- `tests/test_fitting/test_dean_optimization_cylinder.py`
- `tests/test_fitting/test_dean_optimization_broadcast_scaling.py`

## Proposal For The Principled Benchmark Matrix

For the high-level options, I propose a fixed benchmark matrix:

1. Objective-only trapezoid benchmark
   - scalar current path
   - batched CPU current path
   - future Dean CPU path
   - future Dean GPU path
   - evaluate across named batch profiles, not just one population size

2. Tiny end-to-end trapezoid optimizer benchmark
   - same data
   - same initial parameters
   - reduced `maxiter`, `popsize`, `polish=False`
   - compare runtime, GF, parameter drift

3. Medium offline trapezoid benchmark
   - not part of default `pytest`
   - closer to notebook settings
   - compare scaling versus population size
   - explicitly include sizes where the first Dean candidate flattened out, so later work must beat the current plateau rather than just the smoke profile

4. Structure-similarity regression benchmark
   - compare optimized structures between baseline and candidate implementation
   - define acceptable tolerance per parameter and total GF drift
   - keep this scaffolded but non-gating until a significant speedup candidate exists

### Named Batch Profiles

- `smoke`: `1, 4, 16`
  - quick correctness and timing sanity checks
- `broadcast`: `1, 4, 16, 64, 256, 1024`
  - default offline CPU scaling profile
- `gpu_candidate`: `1, 16, 64, 256, 1024, 4096`
  - larger profile for testing whether a candidate plausibly benefits from massive broadcasting

### Current Next-Step Proposal

The next principled option should not be "more of the same caching." The first candidate already captured most of the easy Python overhead.

The best next target is:

1. Keep the Dean GPU-resident batched path as the active acceleration branch for trapezoid fitting.
   - current leading implementation: fused GPU post-form-factor variant
2. Improve that branch only where measurements still say it matters:
   - fused post-form-factor work
   - persistent GPU workspaces
   - larger-batch optimizer workflows
3. Avoid investing in scalar GPU routing unless a future workflow truly needs it.
4. Use structure-similarity regression as a real gate for subsequent GPU-path changes, because there is now a significant speedup worth protecting.

This should let us reject changes that are fast but scientifically sloppy, and also reject changes that are scientifically fine but only move time from compute into Python overhead.
