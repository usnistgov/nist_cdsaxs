# DEAN Optimization Ledger

## Purpose

This ledger captures each optimization experiment in enough detail to answer:

- What was changed?
- How was it measured?
- Did it improve speed?
- Was it kept, deferred, or rejected?

The ledger is intentionally more detailed than `DEAN_OPTIMIZATION.md`. The plan defines the roadmap. The ledger captures actual trials.

## Current Artifacts

Added so far:

- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_trapezoid_realistic.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`
- `tests/test_fitting/test_dean_workflow_semantics.py`

Environment used so far:

- branch: `feature/dean_optimization`
- mamba env: `cdsax-dev`
- editable install: `pip install -e .[dev]`
- extras installed for fitting/benchmark support:
  - `pytest-benchmark`
  - `cython`
  - `emcee`
  - `line_profiler`
  - `corner`
  - `cupy-cuda12x`
- local acceleration package installed into the env:
  - editable `cdsaxs_cython` from `src/cdsaxs/Fitting`
- GPU hardware visible in this workspace:
  - `Quadro RTX 8000`
  - `49152 MiB`
  - three devices reported by `nvidia-smi`

## Measurement Policy

Until a candidate shows meaningful speedup:

- Speed is the first gate.
- Structure similarity is scaffolded but not yet a hard gate.
- Default tests stay lightweight.
- Larger batch-size and broadcast-oriented timing runs are treated as benchmark experiments, not routine CI tests.

Once a candidate shows meaningful speedup:

- Add structure-similarity checks with fixed seeds where possible.
- Compare:
  - GF
  - BIC
  - parameter drift
  - derived structure signature

## Experiment Template

### Experiment ID

- date:
- area:
- candidate:
- status:

### Motivation

- 

### Files Touched

- 

### Benchmark Protocol

- 

### Results

- 

### Outcome

- keep / revise / reject / defer

### Notes

- 

## Experiment Entries

### 2026-03-25 / Stage 0 / Baseline Harness

- area: trapezoid benchmark scaffolding
- candidate: baseline harness only
- status: complete

Motivation:

- Establish a notebook-parallel benchmark before changing fitting internals.

Files touched:

- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`

Benchmark protocol:

- Load the DeRocher V3 trapezoid example data.
- Measure scalar objective, batched objective, and tiny DE scalar/vectorized runs.

Results:

- Harness imported and executed successfully in `cdsax-dev`.
- `pytest tests/test_fitting/test_dean_optimization_trapezoid.py -q` passed.

Outcome:

- keep

Notes:

- Current trapezoid math emits divide warnings in the existing code path.
- The package import also required `corner`, so the env was updated to match actual fitting imports.

### 2026-03-25 / Stage 1 / First Dean Trapezoid Candidate

- area: trapezoid objective hot loop
- candidate: `TrapezoidModelArrayDean`
- status: complete

Motivation:

- Try the lowest-risk CPU optimization first:
  - cached `log(Intensity)`
  - cached `Qx**2 + Qz**2`
  - reduced parameter parsing overhead
  - reduced temporary object churn in scalar and batched objective paths

Files touched:

- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`

Benchmark protocol:

- Use the same DeRocher V3 trapezoid example model.
- Compare baseline vs Dean candidate across increasing batch sizes using objective timing.
- Use named batch profiles:
  - `broadcast`: `1, 4, 16, 64, 256, 1024`
  - `gpu_candidate`: `1, 16, 64, 256, 1024, 4096`
- Run repeated objective trials with warmup.
- Follow with tiny fixed-seed DE comparisons in both scalar and vectorized modes.
- Run a lightweight structure-signature comparison after speedup is confirmed.

Results:

- Objective scaling, `broadcast` profile:
  - batch `1`: about `1.08x` speedup
  - batch `4`: about `1.06x` speedup
  - batch `16`: about `1.26x` speedup
  - batch `64`: about `1.42x` speedup
  - batch `256`: about `1.25x` speedup
  - batch `1024`: about `1.08x` speedup
- Objective scaling, `gpu_candidate` profile:
  - batch `1`: about `1.10x` speedup
  - batch `16`: about `1.03x` speedup
  - batch `64`: about `1.08x` speedup
  - batch `256`: about `1.12x` speedup
  - batch `1024`: about `1.04x` speedup
  - batch `4096`: about `1.04x` speedup
- Tiny DE, scalar objective:
  - baseline elapsed: `0.0555 s`
  - candidate elapsed: `0.0488 s`
  - speedup: about `1.14x`
  - `GF` and `BIC` matched exactly in this fixed-seed run
- Tiny DE, vectorized objective:
  - baseline elapsed: `0.0393 s`
  - candidate elapsed: `0.0339 s`
  - speedup: about `1.16x`
  - `GF` and `BIC` matched exactly in this fixed-seed run
- Lightweight structure-signature comparison:
  - widths matched exactly
  - heights matched exactly
  - `DW` matched exactly
  - `GF` matched exactly
  - `BIC` matched exactly

Outcome:

- keep, but do not treat as the main acceleration win

Notes:

- Structure-similarity regression is scaffolded but intentionally skipped until speedup is demonstrated.
- The first optimization is a useful cleanup pass and a valid new baseline.
- The gain tapers at very large CPU batch sizes, which suggests the dominant cost is moving into trapezoid math kernels rather than Python-side setup.
- This means the next candidate should be guided by profiling inside the vectorized path, not by adding more cache layers blindly.
- Existing optimizer hot-path debug output is still present in the baseline workflow and remains a medium-yield cleanup item.

### 2026-03-25 / Stage 2 / Vectorized Path Component Profile

- area: trapezoid profiling
- candidate: component attribution for current Dean CPU path
- status: complete

Motivation:

- Confirm which subroutine actually dominates once the first CPU cleanup pass is in place.

Files touched:

- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`

Benchmark protocol:

- Profile one vectorized objective call on the DeRocher V3 example.
- Time:
  - `SymCoordAssign`
  - `FreeFormTrapezoid`
  - `GF_calc`
- Inspect representative medium and large batch sizes, especially `1024`.

Results:

- At batch size `1024` for `TrapezoidModelArrayDean`:
  - total elapsed: about `0.758 s`
  - `FreeFormTrapezoid`: about `0.663 s`, or `87.5%`
  - `GF_calc`: about `0.046 s`, or `6.1%`
  - `SymCoordAssign`: negligible
- Similar ratios were seen in ad hoc profiling at other large batch sizes.

Outcome:

- keep as decision-making evidence

Notes:

- This confirmed that the next serious optimization should target the form-factor kernel path, not more parameter mapping cleanup.

### 2026-03-25 / Stage 3 / Legacy CuPy Hook Audit

- area: trapezoid GPU path audit
- candidate: existing `freeform_use_cupy=True` path in the baseline trapezoid model
- status: complete

Motivation:

- Determine whether the repo already had a usable CuPy-native fitting path before building a new Dean GPU variant.

Files touched:

- none

Benchmark protocol:

- Install `cupy-cuda12x` in `cdsax-dev`.
- Run the existing baseline trapezoid objective wrapper with:
  - `freeform_use_cupy=True`
  - batched candidates

Results:

- The existing baseline CuPy path did not produce a usable result in the tested batched objective path.
- `SimTrap_GF` reported `optimization_values must be a numpy array`.
- The wrapper returned an `inf` objective vector.
- Code inspection also confirmed that the old `FreeFormTrapezoid` CuPy path forces a `.get()` back to NumPy before returning, so it is not actually device-resident.

Outcome:

- reject

Notes:

- This was important to test because it means "turn on CuPy" is not the same as "use a GPU-native workflow" in this repo.

### 2026-03-25 / Stage 3 / Dean GPU-Resident Trapezoid Candidate

- area: trapezoid GPU acceleration
- candidate: `TrapezoidModelArrayDeanGPU`
- status: complete

Motivation:

- Build a real GPU-resident batched objective path that preserves the current public fitting workflow while moving the dominant trapezoid math onto the GPU.

Files touched:

- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `src/cdsaxs/Fitting/__init__.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`

Benchmark protocol:

- Use the same DeRocher V3 trapezoid example model.
- Compare `TrapezoidModelArrayDean` against `TrapezoidModelArrayDeanGPU` with `freeform_use_cupy=True`.
- Use warmups to amortize first-call GPU setup.
- Measure:
  - component profile at batch `1024`
  - objective scaling on `broadcast` and `gpu_candidate` profiles
  - tiny fixed-seed DE scalar and vectorized runs
  - fixed-seed structure similarity in vectorized mode
- Re-test after adding small-batch CPU fallback inside the GPU class.

Results:

- GPU component profile at batch `1024`:
  - total elapsed: about `0.138 s`
  - `FreeFormTrapezoid`: about `0.0468 s`
  - `GF_calc`: about `0.0455 s`
  - `SymCoordAssign`: negligible
- Objective scaling versus the Dean CPU path, after warmup:
  - batch `1`: slower, about `0.54x`
  - batch `4`: about `1.30x`
  - batch `16`: about `4.16x`
  - batch `64`: about `8.21x`
  - batch `256`: about `4.87x`
  - batch `1024`: about `6.50x`
- Earlier large-batch runs also showed:
  - batch `4096`: about `5.59x`
- Tiny DE, scalar objective:
  - GPU path remained slower, about `0.57x`
  - `GF` and `BIC` matched exactly
- Tiny DE, vectorized objective:
  - speedup about `4.07x`
  - `GF` and `BIC` matched exactly
- Fixed-seed vectorized structure similarity:
  - widths matched exactly
  - heights matched exactly
  - `DW` matched exactly
  - `GF` matched exactly
  - `BIC` matched exactly
- Focused validation:
  - `pytest tests/test_fitting/test_dean_optimization_trapezoid.py -q` passed with `8` tests
  - `pytest tests/test_fitting/test_dean_optimization_structure_similarity.py -q` passed with `1` test

Outcome:

- keep

Notes:

- The GPU candidate is a real acceleration path for vectorized and broadcast-oriented workloads.
- It is not a scalar-optimization win, so the public workflow should continue to steer GPU use toward vectorized candidate evaluation.
- The class now includes a small-batch CPU fallback because GPU launch and transfer overheads dominate tiny batches.

### 2026-03-25 / Stage 4 / Fused GPU Post-Form-Factor Candidate

- area: trapezoid GPU refinement
- candidate: `TrapezoidModelArrayDeanGPUFused`
- status: complete

Motivation:

- The first Dean GPU profile still showed roughly one third of objective time outside `FreeFormTrapezoid`, largely in post-form-factor array assembly and GF accumulation.
- Try a fused GPU residual path that reduces temporary-array traffic after the form factor is computed.

Files touched:

- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`
- `src/cdsaxs/Fitting/__init__.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`

Benchmark protocol:

- Compare `TrapezoidModelArrayDeanGPU` against `TrapezoidModelArrayDeanGPUFused` with `freeform_use_cupy=True`.
- Check batched objective parity on a representative candidate matrix.
- Compare objective timing at batch sizes:
  - `16`
  - `64`
  - `256`
  - `1024`
  - `4096`
- Compare tiny fixed-seed vectorized DE timing and final fit metrics.
- Re-compare the fused candidate directly against the Dean CPU baseline.

Results:

- Objective parity:
  - fused GPU candidate matched the first Dean GPU candidate exactly on the tested batch
- GPU-vs-GPU objective timing:
  - batch `16`: about `0.94x`
  - batch `64`: about `2.23x`
  - batch `256`: about `1.01x`
  - batch `1024`: about `1.03x`
  - batch `4096`: about `0.99x`
- Tiny DE, vectorized objective:
  - first Dean GPU elapsed: about `0.0126 s`
  - fused Dean GPU elapsed: about `0.00973 s`
  - speedup: about `1.29x`
  - `GF` and `BIC` matched exactly
- Fused GPU versus Dean CPU:
  - batch `16`: about `4.29x`
  - batch `64`: about `8.39x`
  - batch `256`: about `5.28x`
  - batch `1024`: about `6.54x`
  - batch `4096`: about `5.71x`
  - tiny vectorized DE speedup: about `3.75x`
  - fixed-seed structure regression passed with exact widths, heights, `DW`, `GF`, and `BIC`
- Focused validation:
  - `pytest tests/test_fitting/test_dean_optimization_trapezoid.py -q` passed with `9` tests
  - `pytest tests/test_fitting/test_dean_optimization_structure_similarity.py -q` passed with `1` test

Outcome:

- keep

Notes:

- The fused path is not uniformly faster for every batch size versus the first Dean GPU candidate.
- It is still worth keeping because it improves the tiny vectorized optimizer run, which is closer to the real fitting workflow than a single synthetic objective call.
- This fused GPU class is the current leading trapezoid acceleration branch.

### 2026-03-25 / Stage 5 / Cython Installation And Scalar Comparison

- area: trapezoid compiled CPU path
- candidate: existing `cdsaxs_cython` package and `AcceleratedTrapezoidModel`
- status: complete

Motivation:

- The repo already contained a Cython extension package, but it was not installed into the working env.
- Benchmark it in the regime where it is actually valid and compare it against the current vectorized CPU and GPU paths.

Files touched:

- `src/cdsaxs/Fitting/accelerated_models.py`
- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`

Environment actions:

- built local extensions with:
  - `python setup.py build_ext --inplace`
- installed editable local package with:
  - `pip install -e . --no-build-isolation`
  - from `src/cdsaxs/Fitting`

Benchmark protocol:

- Verify direct import of `cdsaxs_cython` and `CYTHON_AVAILABLE=True`.
- Run function-level benchmark using the existing `run_comprehensive_benchmark(...)`.
- Fix the accelerated wrapper bug that used `Intensity or self.Intensity` with NumPy arrays.
- Re-benchmark scalar-only usage:
  - scalar objective parity
  - repeated scalar candidate sweeps for:
    - incumbent
    - Cython accelerated model
    - Dean CPU
- Compare scalar Cython total time against:
  - Dean CPU batched objective
  - Dean fused GPU batched objective
  at matched candidate counts.
- Compare tiny fixed-seed DE runs:
  - incumbent scalar
  - Cython scalar
  - Dean CPU scalar
  - Dean CPU vectorized
  - Dean fused GPU vectorized

Results:

- Installation:
  - `cdsaxs_cython` now imports successfully from normal repo-root execution
  - `CYTHON_AVAILABLE=True`
- Function-level benchmark on the DeRocher example:
  - `GF_calc`: about `1.01x`
  - `SymCoordAssign`: about `1.30x`
  - `FreeFormTrapezoid`: about `1.00x`
- Scalar objective parity:
  - default scalar candidate matched exactly
  - tested batch of scalar candidates matched exactly
- Scalar objective throughput:
  - incumbent scalar per candidate: roughly `0.71 ms` to `1.20 ms`
  - Cython scalar per candidate: roughly `0.44 ms` to `0.52 ms`
  - Dean CPU scalar per candidate: roughly `0.60 ms` to `0.85 ms`
- Scalar Cython versus vectorized Dean paths at matched candidate counts:
  - versus Dean CPU batched:
    - batch `16`: about parity
    - batches `64` to `1024`: scalar Cython remained faster than Dean CPU batched
  - versus Dean fused GPU batched:
    - batch `16`: GPU about `3.15x` faster
    - batch `64`: GPU about `3.23x` faster
    - batch `256`: GPU about `3.80x` faster
    - batch `1024`: GPU about `4.00x` faster
- Tiny DE comparisons:
  - incumbent scalar: about `0.0490 s`, `GF=1719.1091`
  - Cython scalar: about `0.0310 s`, `GF=1719.1091`
  - Dean CPU scalar: about `0.0431 s`, `GF=1719.1091`
  - Dean CPU vectorized: about `0.0353 s`, `GF=1268.2043`
  - Dean fused GPU vectorized: about `0.00904 s`, `GF=1268.2043`

Outcome:

- keep, but only as a scalar CPU option

Notes:

- The current Cython path is not a real batched/vectorized optimization path.
- Existing batched/vectorized Cython attempts fall back because the compiled functions expect lower-dimensional inputs.
- That makes the current compiled code valuable for scalar CPU work, but not a competitor to the vectorized GPU branch.
- Dean CPU remains the batched CPU reference path.
- Dean fused GPU remains the leading vectorized path.

## 2026-03-25: Realistic GPU Harness Build And Notebook-Derived Scaling Baseline

Goal:

- Replace the tiny-only benchmark with a notebook-derived realistic harness that still finishes in seconds, not minutes.
- Measure the current fused GPU path across a realistic trapezoid profile ladder.
- Keep repo structure and public semantics parallel to the existing fitting code.

Files touched:

- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py`
- `src/cdsaxs/Fitting/__init__.py`
- `tests/test_fitting/test_dean_optimization_trapezoid_realistic.py`
- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`

Implementation notes:

- Initial version added realistic profile builders that derived notebook-parallel `model_params` on the baseline trapezoid model and then re-instantiated the requested fitting variant.
- That initial builder was later superseded by the workflow-native refactor recorded below.
- Implemented four routine realistic profiles:
  - `notebook_4param`
  - `notebook_layer10_8param`
  - `notebook_layer90_8param`
  - `notebook_multilayer_12param`
- Implemented benchmark families:
  - deterministic candidate-budget objective timing
  - one-generation vectorized DE timing
  - short multi-generation vectorized DE timing
- Defined candidate budget in a principled way:
  - `candidate_count = population_size * parameter_count`
- Added a fused-GPU matrix runner and saved one reference result set to:
  - `.dean_realistic_gpu_scaling.json`

Bug found and fixed:

- Layer-addition helpers left `slds` at length `1` even when the new model had multiple layers.
- That allowed some objective-only timing paths to run, but broke later end-to-end optimizer simulation with:
  - `SLD array length (1) must exactly match number of layers`
- Fixed in the initial harness by expanding the single-material `slds` list to one value per layer before re-instantiating the target model class.
- That repair later moved into the core layer-growth API so the harness no longer has to own it.

Benchmark protocol:

- Environment:
  - `mamba run -n cdsax-dev`
  - CuPy visible with `3` CUDA devices
- Candidate-budget timing:
  - `3` timed repeats
  - `1` warmup
  - fused GPU class
- One-generation DE:
  - `maxiter=1`
  - `tol=0.5`
  - `polish=False`
  - fixed seed `1234`
- Short multi-generation DE:
  - `maxiter=2`
  - `tol=0.0`
  - `polish=False`
  - fixed seed `1234`

Validation:

- `pytest tests/test_fitting/test_dean_optimization_trapezoid_realistic.py -q`
  - `10 passed`
- existing Dean optimization regression coverage still passed after the additions

Results:

- Candidate-budget throughput stayed in a fairly tight band across the realistic ladder:
  - `notebook_4param`: about `7.7k` to `9.1k` candidates/s
  - `notebook_layer10_8param`: about `7.8k` to `8.3k` candidates/s
  - `notebook_layer90_8param`: about `7.7k` to `8.3k` candidates/s
  - `notebook_multilayer_12param`: about `7.4k` to `10.5k` candidates/s
- Objective cost per candidate was roughly:
  - `95 us` to `136 us`
- One-generation DE timing:
  - `notebook_4param`: about `0.13 s` to `0.59 s`
  - `notebook_layer10_8param`: about `0.07 s` to `0.27 s`
  - `notebook_layer90_8param`: about `0.07 s` to `0.27 s`
  - `notebook_multilayer_12param`: about `0.06 s` to `0.23 s`
- Short multi-generation DE timing:
  - `notebook_4param`: about `0.40 s` to `0.91 s`
  - `notebook_layer10_8param`: about `0.21 s` to `0.42 s`
  - `notebook_layer90_8param`: about `0.21 s` to `0.42 s`
  - `notebook_multilayer_12param`: about `0.18 s` to `0.35 s`
- Best observed `GF` values in the current matrix:
  - `notebook_4param`
    - one-generation best: about `1267.60`
    - multi-generation best: about `1229.39`
  - `notebook_layer10_8param`
    - one-generation best: about `3938.07`
    - multi-generation best: about `3919.46`
  - `notebook_layer90_8param`
    - one-generation best: about `3948.27`
    - multi-generation best: about `3890.57`
  - `notebook_multilayer_12param`
    - one-generation best: about `4140.12`
    - multi-generation best: about `4140.12`

Outcome:

- keep

Why kept:

- The realistic harness now exists in the repo with parallel structure and naming.
- The current fused GPU path remains fast and stable across a more notebook-like scaling ladder.
- The harness is now strong enough to serve as the baseline for future GPU optimization attempts.

Open follow-on work:

- Add a larger offline notebook-derived profile after current GPU optimization ideas stop moving the 4/8/12-parameter ladder.
- Re-introduce scalar Cython comparison later when cross-device reporting matters again.
- Only add strict structure-similarity gating for future candidates that beat this realistic GPU baseline by a meaningful amount.

## 2026-03-25: Realistic GPU Versus Scalar Cython Incumbent

Goal:

- Benchmark the realistic 4/8/12-parameter trapezoid workloads against the current scalar Cython incumbent.
- Use the comparison to decide whether future optimization should focus on small-batch overheads or large-batch vectorized throughput.

Artifacts saved:

- `.dean_realistic_incumbent_scaling.json`
- `.dean_realistic_gpu_vs_incumbent.json`

Benchmark protocol:

- Incumbent model:
  - `AcceleratedTrapezoidModel`
- GPU model:
  - `TrapezoidModelArrayDeanGPUFused`
- Candidate-budget comparison:
  - same realistic profile ladder as the GPU baseline
  - deterministic candidate matrices
  - scalar incumbent evaluated one candidate at a time
  - fused GPU evaluated the whole candidate matrix vectorized
- Optimizer comparison:
  - same `popsize`, `maxiter`, seed, mutation, recombination, and polish settings
  - incumbent ran scalar DE
  - GPU ran vectorized DE
  - one-generation family used `maxiter=1`, `tol=0.5`
  - short multi-generation family used `maxiter=2`, `tol=0.0`

Results:

- Candidate-budget objective speedup of fused GPU over scalar Cython:
  - `notebook_4param`: about `3.6x` to `5.2x`
  - `notebook_layer10_8param`: about `6.3x` to `7.1x`
  - `notebook_layer90_8param`: about `7.0x` to `7.6x`
  - `notebook_multilayer_12param`: about `12.0x` to `17.2x`
- One-generation DE wall-time speedup:
  - `notebook_4param`: about `3.7x` to `4.1x`
  - `notebook_layer10_8param`: about `6.1x` to `6.5x`
  - `notebook_layer90_8param`: about `6.6x` to `7.0x`
  - `notebook_multilayer_12param`: about `10.5x` to `11.4x`
- Short multi-generation DE wall-time speedup:
  - `notebook_4param`: about `3.6x` to `3.9x`
  - `notebook_layer10_8param`: about `6.3x` to `6.4x`
  - `notebook_layer90_8param`: about `7.0x` to `7.1x`
  - `notebook_multilayer_12param`: about `10.9x` to `11.3x`

Fit-quality observations:

- `notebook_4param`:
  - scalar Cython and fused GPU were close under the short-DE budgets
  - neither path dominated completely on `GF`
- Layered realistic profiles:
  - fused GPU not only ran much faster, it also reached much lower `GF` under the same short budgets
  - scalar Cython was materially worse on the tested 8- and 12-parameter ladder
  - this is likely a combination of:
    - vectorized path efficiency
    - higher practical search budget per wall-clock second
    - optimizer-behavior differences between scalar and vectorized SciPy DE execution

Small-batch crossover observations:

- Very small candidate sets still benefit from the fused GPU path, but only modestly:
  - `notebook_4param`, `4` candidates: about `1.3x`
  - `notebook_4param`, `8` candidates: about `2.1x`
  - `notebook_layer10_8param`, `8` candidates: about `2.3x`
- Once the batch reaches the tens-of-candidates regime, vectorization becomes clearly dominant:
  - `notebook_4param`, `16` candidates: about `3.3x`
  - `notebook_layer10_8param`, `16` candidates: about `4.1x`
  - `notebook_layer10_8param`, `32` candidates: about `4.3x`

Outcome:

- keep fused GPU as the primary optimization target
- keep scalar Cython only as the incumbent CPU reference and fallback path

Optimization implication:

- Small-workload tuning priority:
  - launch overhead
  - host/device transfer overhead
  - small-batch CPU fallback threshold
- Large-workload tuning priority:
  - fused device-resident math
  - memory traffic reduction
  - occupancy and batch-shape tuning
  - avoiding scalar-path work entirely where possible

## 2026-03-25: Workflow Surface Stabilization Before Harness Refactor

Goal:

- make layer-growth and stateful workflow APIs usable without harness-only workarounds
- remove unconditional debug noise from routine optimizer and import paths
- clear the way for a workflow-native harness refactor

Files touched:

- `src/cdsaxs/Fitting/CDSAXS_base_model.py`
- `src/cdsaxs/Fitting/Trapezoid_model.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `src/cdsaxs/Fitting/SiGe_model.py`
- `src/cdsaxs/Fitting/Cylinder_model.py`
- `src/cdsaxs/Fitting/cython_integration.py`
- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`

Implementation notes:

- `_create_new_model_from_params(...)` now prefers the concrete runtime class before falling back to package-level factories.
- `_copy_model_data(...)` now propagates `_freeform_use_cupy` and re-runs `process_imported_data()` on the copied-data model so caches and baseline simulation state are rebuilt.
- `add_layer_at_percentage(...)` now normalizes `slds` to the new layer count instead of leaving that repair to the realistic harness.
- added a backward-compatible `parameter_sweep_2d(...)` entry point that delegates to `parameter_sweep_2d_clean(...)`.
- added `_resolve_active_parameter_names(...)` and updated the trapezoid, Dean trapezoid, SiGe, and cylinder wrappers to choose the parameter-name set that matches the active workflow state.
- removed unconditional debug prints from the optimizer hot path and removed import-time Cython status prints.

Validation:

- runtime smoke checks in `cdsax-dev` confirmed:
  - `TrapezoidModelArrayDeanGPUFused.add_layer_at_percentage(10)` returns `TrapezoidModelArrayDeanGPUFused`
  - the grown model now has `len(model.model_params['slds']) == model.layers`
  - `parameter_sweep_2d(...)` executes successfully on a tiny 2x2 smoke case
  - `batch_initialize_and_fit(...)` no longer emits the old optimizer debug spam in quiet mode
  - a short `CDSAXS_MCMC(...)` smoke run now proceeds with a nonzero mean acceptance fraction in the tested case

Outcome:

- keep

Notes:

- this supersedes the earlier harness-only `slds` repair as the long-term fix point
- the realistic trapezoid harness was later refactored to stop re-instantiating from cloned parameters

## 2026-03-25: Workflow-Native Trapezoid Harness Refactor

Goal:

- stop building realistic trapezoid profiles by cloning `model_params`
- drive the realistic harness through the same layer-growth APIs that real workflows use
- add the missing workflow-native trapezoid runners and structured metadata

Files touched:

- `src/cdsaxs/Fitting/CDSAXS_base_model.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py`
- `src/cdsaxs/Fitting/__init__.py`
- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`

Implementation notes:

- `build_realistic_trapezoid_model(...)` now constructs the requested model class first and then applies the profile through:
  - `add_layer_at_percentage(...)`
  - `add_multiple_layers(...)`
- profile descriptions now record workflow semantics such as:
  - `layer_growth_path`
  - `height_percentage`
  - `height_percentages`
  - `sequential`
  - `freeform_use_cupy`
- existing realistic candidate-budget and DE runners now emit structured workflow metadata.
- added trapezoid workflow-native runners for:
  - `run_realistic_dual_annealing_trial(...)`
  - `run_realistic_batch_initialize_smoke(...)`
  - `run_realistic_mcmc_profile(...)`
  - `run_realistic_sweep_replay_smoke(...)`
- `_run_scipy_optimizer(...)` now strips `vectorized` before non-DE optimizers so `dual_annealing` can run through `CDSAXS_Optimize(...)`.

Validation:

- smoke execution in `cdsax-dev` confirmed:
  - realistic layer-growth now preserves the requested class on the direct builder path
  - realistic DE coverage still runs
  - dual annealing returns finite `GF`
  - batch initialization smoke returns finite best-fit metrics
  - MCMC smoke returns a nonzero mean acceptance fraction in the tested case
  - sweep replay smoke successfully applies best-fit parameters through `show_best_fit_results(...)`

Outcome:

- keep

Notes:

- the harness now reflects current repo workflow semantics much more closely
- the test split follow-on described here was completed in the next entry

## 2026-03-25: Semantic Workflow Regression And Smoke Test Reorganization

Goal:

- split trapezoid fitting coverage by purpose: workflow semantics versus benchmark smoke
- lock the notebook-parallel realistic profiles to the intended `4`/`8`/`12` optimization ladders
- remove stale assumptions from existing regression coverage before resuming GPU optimization

Files touched:

- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py`
- `tests/test_fitting/test_dean_optimization_trapezoid_realistic.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`
- `tests/test_fitting/test_dean_workflow_semantics.py`
- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`

Implementation notes:

- realistic profile normalization now assigns the notebook-parallel optimization surface directly so layered profiles keep geometry plus `DW`, `I0`, and scalar `Bk` without auto-appending `sld_*`
- realistic profile tests now assert the exact parameter-name ladders for:
  - `notebook_4param`
  - `notebook_layer10_8param`
  - `notebook_layer90_8param`
  - `notebook_multilayer_12param`
- added `tests/test_fitting/test_dean_workflow_semantics.py` to separate semantic workflow regressions from benchmark smoke coverage
- the new semantic suite covers:
  - class preservation through layer growth
  - `parameter_sweep_2d(...)` compatibility
  - dual annealing smoke
  - batch initialization smoke schema
  - MCMC smoke metrics
  - sweep replay smoke
  - stateful sweep-to-MCMC parameter-name consistency
- the structure-similarity regression no longer assumes GPU must win wall time on a tiny run; it now checks timing sanity plus structural parity

Validation:

- `mamba run -n cdsax-dev python -m pytest tests/test_fitting/test_dean_optimization_trapezoid_realistic.py -q`
  - `10 passed`
- `mamba run -n cdsax-dev python -m pytest tests/test_fitting/test_dean_workflow_semantics.py -q`
  - `7 passed`
- `mamba run -n cdsax-dev python -m pytest tests/test_fitting/test_dean_optimization_structure_similarity.py -q`
  - `1 passed`
- `mamba run -n cdsax-dev python -m pytest tests/test_fitting -q`
  - `27 passed`

Outcome:

- keep

Notes:

- the harness and test prerequisite for resuming trapezoid GPU optimization is now complete
- stale documentation that still described the realistic builder as cloned-parameter re-instantiation or the test split as future work was updated in this step
