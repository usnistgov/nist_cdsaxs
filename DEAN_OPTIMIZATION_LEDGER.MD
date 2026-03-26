# DEAN Optimization Ledger

## Purpose

This ledger captures each optimization experiment in enough detail to answer:

- What was changed?
- How was it measured?
- Did it improve speed?
- Was it kept, deferred, or rejected?

The ledger is intentionally more detailed than `DEAN_OPTIMIZATION.md`. The plan defines the roadmap. The ledger captures actual trials.

Current planning docs now live at:

- `DEAN_OPTIMIZATION.md`
- `docs/optimization/README.md`
- `docs/optimization/trapezoid.md`
- `docs/optimization/sige.md`

## Current Artifacts

Added so far:

- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`
- `docs/optimization/README.md`
- `docs/optimization/trapezoid.md`
- `docs/optimization/sige.md`
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
- mamba env: `cdsaxs-dev`
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

### 2026-03-26 / Planning Refactor / Geometry Docs And SiGe Roadmap

- area: planning and documentation checkpoint
- candidate: geometry-specific planning docs plus resumable SiGe roadmap
- status: complete

Motivation:

- the root planning docs had become trapezoid-heavy and difficult to resume from
- SiGe needed a dedicated roadmap with enough detail to continue from a fresh context
- the documented env name was stale and needed to match the renamed `cdsaxs-dev` environment

Files touched:

- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`
- `docs/optimization/README.md`
- `docs/optimization/trapezoid.md`
- `docs/optimization/sige.md`

Benchmark protocol:

- none
- this checkpoint records planning structure, not a timing run

Results:

- converted `DEAN_OPTIMIZATION.md` into a landing page for active optimization work
- established `docs/optimization/` as the location for active geometry-specific plans
- recorded the current trapezoid reusable patterns and kept implementation status
- recorded the SiGe reuse appraisal
- recorded the detailed CPU-vectorized SiGe plan
- recorded the basic follow-on GPU SiGe plan
- corrected the documented env name to `cdsaxs-dev`

Outcome:

- keep

Notes:

- current active SiGe conclusion remains CPU vectorization first, GPU second
- existing `Dean` names remain historical for current trapezoid code, but new work should use neutral `_vectorized`, `_vectorized_GPU`, and `_GPU` naming
- the only non-doc repo code change already on branch at this checkpoint is the `SRM_model.py` import fix

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

- Harness imported and executed successfully in `cdsaxs-dev`.
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

- Install `cupy-cuda12x` in `cdsaxs-dev`.
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
  - `mamba run -n cdsaxs-dev`
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

- runtime smoke checks in `cdsaxs-dev` confirmed:
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

- smoke execution in `cdsaxs-dev` confirmed:
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

- `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_optimization_trapezoid_realistic.py -q`
  - `10 passed`
- `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_workflow_semantics.py -q`
  - `7 passed`
- `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_optimization_structure_similarity.py -q`
  - `1 passed`
- `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting -q`
  - `27 passed`

Outcome:

- keep

Notes:

- the harness and test prerequisite for resuming trapezoid GPU optimization is now complete
- stale documentation that still described the realistic builder as cloned-parameter re-instantiation or the test split as future work was updated in this step

## 2026-03-25: Prioritize Separate Speculative Vectorized MCMC Harness In Documentation

Goal:

- document a separate speculative vectorized MCMC harness before further trapezoid GPU optimization
- keep native notebook MCMC timing and validation distinct from optimization-only vectorized MCMC comparisons
- define the comparison rule for the planned harness in terms of similar gross model-evaluation budgets

Files touched:

- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`

Implementation notes:

- the action plan priority order now places a separate speculative vectorized trapezoid MCMC harness ahead of further trapezoid GPU tuning
- the main optimization document now distinguishes:
  - native notebook-parallel `CDSAXS_MCMC(...)` smoke and trial coverage
  - a planned speculative vectorized-MCMC evaluation-budget harness
- the planned comparison rule is now explicit:
  - choose a native MCMC budget target
  - approximate gross model evaluations as `n_walkers * n_steps`
  - execute a similar number of model evaluations through vectorized walker-matrix batches on CPU and GPU
  - log target budget, chunk size, and realized total evaluations
- stage-planning and immediate-next-work sections now say this harness should land before more trapezoid GPU optimization resumes

Validation:

- not run
  - documentation-only update

Outcome:

- keep

Notes:

- this update is intentionally planning-only; it does not yet implement the speculative vectorized MCMC harness
- stale wording that treated generic MCMC expansion as a lower-priority follow-on was updated here

## 2026-03-25: Implement Vectorized MCMC Evaluation-Budget Harness And Tests

Goal:

- implement the separate speculative vectorized MCMC harness that the optimization document had planned
- add regression coverage that keeps the harness purpose explicit and protects incumbent-versus-GPU parity
- document the new test and clarify that it is optimization-only, not a replacement for native notebook MCMC timing

Files touched:

- `DEAN_OPTIMIZATION.md`
- `environment_dev.yml`
- `pyproject.toml`
- `src/cdsaxs/Fitting/CDSAXS_base_model.py`
- `src/cdsaxs/Fitting/__init__.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py`
- `tests/test_fitting/test_dean_mcmc_vectorized_eval_budget.py`
- `tests/test_fitting/test_dean_optimization_trapezoid_realistic.py`

Implementation notes:

- `CDSAXS_MCMC(...)` now accepts an optional `seed` so short realistic MCMC runs can be compared deterministically across incumbent and GPU-backed model classes
- the realistic trapezoid harness now includes `run_realistic_vectorized_mcmc_eval_budget_profile(...)`
  - it matches the speculative comparison rule from the planning document
  - it derives a gross evaluation budget from a native MCMC baseline
  - it builds one deterministic candidate stream for that budget
  - it replays that stream in vectorized walker-batch chunks on CPU and GPU
- a dedicated regression file now covers the new harness purpose directly:
  - vectorized evaluation-budget smoke/schema coverage
  - chunk-size invariance for the same candidate stream
  - seeded incumbent-versus-GPU native MCMC parity on a short realistic profile
  - incumbent-versus-GPU parity for the vectorized evaluation-budget path
- realistic DE coverage now also includes a short incumbent-versus-GPU one-generation parity check in the realistic trapezoid test file
- the optimization document now treats the vectorized-MCMC family as implemented and documents its use boundary:
  - use it to answer vectorized throughput and parity questions
  - do not use it to claim notebook-native MCMC timing or optimizer-level convergence superiority

Validation:

- run in `/home/deand/mambaforge/envs/cdsaxs-dev/bin/python`
- `python -m pytest -q tests/test_fitting --maxfail=1`
  - `32 passed`
- additional realistic incumbent-versus-GPU DE/native-MCMC/vectorized-MCMC comparison runs completed in the same environment

Outcome:

- keep

Notes:

- before environment cleanup, the similarly named `cdsaxs_dev` environment on this machine was missing required scientific-test dependencies; the surviving project environment is now `cdsaxs-dev`
- the new vectorized-MCMC test is intentionally a harness guardrail for future optimization work, not a claim that native `CDSAXS_MCMC(...)` is already walker-vectorized

## 2026-03-25: Verify Dean GPU Path Execution Under Current CuPy And Reprioritize Optimization Work

Goal:

- verify that the current Dean GPU classes actually execute their intended GPU-resident path in the active `cdsaxs-dev` environment
- separate environment-compatibility issues from true performance issues
- use one more round of measurement to produce a prioritized GPU-only optimization plan

Files touched:

- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`

Benchmark protocol:

- environment verification:
  - confirm `cdsaxs-dev` is the usable env
  - confirm visible CUDA devices
  - inspect CuPy error-state support
- current-path verification:
  - compare:
    - `TrapezoidModelArrayDean` with `freeform_use_cupy=True`
    - `TrapezoidModelArrayDeanGPU`
    - `TrapezoidModelArrayDeanGPUFused`
  - use the existing DeRocher example objective batch at size `1024`
- analysis-only forcing run:
  - temporarily replace `cp.errstate(...)` with a no-op context manager in-process
  - re-measure the intended resident GPU path
  - compare against the Dean CPU-plus-CuPy hybrid path on realistic objective-batch cases:
    - `notebook_4param`, population `128`
    - `notebook_4param`, population `1024`
    - `notebook_layer10_8param`, population `128`
    - `notebook_multilayer_12param`, population `64`
- attribution run:
  - with the same analysis-only forcing setup, time:
    - batched parameter mapping
    - `SymCoordAssign`
    - host-to-device transfers
    - `_free_form_trapezoid_gpu(...)`
    - fused residual/reduction
    - device-to-host result copy
- small-batch screening:
  - compare the Dean CPU-plus-CuPy hybrid path against the analysis-only resident GPU path at batch sizes:
    - `4`
    - `8`
    - `16`
    - `64`
    - `256`
    - `1024`

Results:

- environment:
  - `cdsaxs-dev` was the correct env for this repo
  - `cupy==14.0.1`
  - `cupy.errstate` was absent
  - `cupyx.errstate` existed but raised `NotImplementedError`
- current-path verification at batch `1024`:
  - Dean CPU with CuPy hybrid: about `0.116 s`
  - Dean GPU current: about `0.119 s`
  - Dean GPU fused current: about `0.119 s`
  - interpretation:
    - the current Dean GPU classes were effectively matching the Dean CPU-plus-CuPy hybrid path in this env, which is consistent with silent fallback from the intended resident path
- analysis-only forced resident GPU path at batch `1024`:
  - Dean CPU with CuPy hybrid: about `0.116 s`
  - Dean GPU resident: about `0.00924 s`
  - Dean GPU fused resident: about `0.00867 s`
- realistic objective-batch comparisons, Dean CPU-plus-CuPy hybrid versus analysis-only Dean GPU fused resident:
  - `notebook_4param`, population `128`:
    - hybrid: about `0.0623 s`
    - resident GPU: about `0.00487 s`
    - speedup: about `12.8x`
  - `notebook_4param`, population `1024`:
    - hybrid: about `0.541 s`
    - resident GPU: about `0.0319 s`
    - speedup: about `17.0x`
  - `notebook_layer10_8param`, population `128`:
    - hybrid: about `0.120 s`
    - resident GPU: about `0.0134 s`
    - speedup: about `8.9x`
  - `notebook_multilayer_12param`, population `64`:
    - hybrid: about `0.0990 s`
    - resident GPU: about `0.0196 s`
    - speedup: about `5.0x`
- attribution with the analysis-only resident GPU path:
  - `notebook_4param`, population `128`:
    - total: about `0.00466 s`
    - `_free_form_trapezoid_gpu(...)`: about `81.8%`
    - post-form-factor residual/reduction: about `13.3%`
    - CPU mapping plus `SymCoordAssign` plus transfers: about `6.8%`
  - `notebook_4param`, population `1024`:
    - total: about `0.0272 s`
    - `_free_form_trapezoid_gpu(...)`: about `83.5%`
    - post-form-factor residual/reduction: about `13.2%`
    - CPU mapping plus `SymCoordAssign` plus transfers: about `3.2%`
  - `notebook_multilayer_12param`, population `64`:
    - total: about `0.0196 s`
    - `_free_form_trapezoid_gpu(...)`: about `94.5%`
    - post-form-factor residual/reduction: about `3.6%`
    - CPU mapping plus `SymCoordAssign` plus transfers: about `2.8%`
- small-batch screening under the analysis-only resident GPU path:
  - batch `4`: near parity with the hybrid path
  - batch `8`: modest resident-GPU win
  - batch `16` and above: clear resident-GPU win
  - batch `1024`:
    - hybrid: about `0.118 s`
    - resident GPU: about `0.00851 s`

Outcome:

- keep as planning and prioritization evidence

Notes:

- the first blocker for the next Dean GPU round is not a math-kernel optimization; it is restoring the intended resident path in the real CuPy environment
- after that blocker is removed, the dominant remaining target is `_free_form_trapezoid_gpu(...)`
- host-side mapping and transfer cleanup were shown to be much lower-yield than form-factor-kernel work on realistic workloads
- the no-op `errstate` replacement used here was analysis-only and should not be treated as the production fix without follow-on review

## 2026-03-25 / Stage 6 / Dean GPU Resident-Path Compatibility Restore

- area: trapezoid GPU execution-path restore
- candidate: CuPy-14-safe resident Dean GPU path with explicit fallback diagnostics
- status: complete

Motivation:

- the intended Dean GPU-resident path was not executing in the active `cdsaxs-dev` environment because `cupy==14.0.1` does not provide a working `cp.errstate(...)`
- before any additional GPU kernel tuning, the resident path had to be restored and made observable so silent CPU fallback could not masquerade as GPU timing

Files touched:

- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`

Implementation notes:

- added a one-time CuPy errstate capability probe and a safe `_gpu_errstate()` context manager
- when CuPy errstate support is unavailable, the resident path now proceeds without it and relies on the existing `nan_to_num(...)` cleanup instead of throwing and silently falling back
- added explicit execution-path diagnostics on the Dean GPU classes:
  - resident GPU success
  - scalar CPU fallback
  - small-batch CPU fallback
  - disabled GPU fallback
  - exception-triggered CPU fallback
- added tests that prove:
  - resident GPU execution is used for batched objectives above the GPU minimum batch
  - small batches still take the intended CPU fallback path

Benchmark protocol:

- focused validation:
  - `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_optimization_trapezoid.py -q`
  - `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_optimization_trapezoid_realistic.py -q`
  - `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_mcmc_vectorized_eval_budget.py -q`
- realistic DE timing comparisons against the Dean CPU incumbent:
  - `notebook_4param`, `one_generation`, population `128`, repeated runs
  - `notebook_multilayer_12param`, `one_generation`, population `64`, repeated runs
  - `notebook_4param`, `multi_generation`, population `256`, repeated runs
- realistic MCMC comparisons:
  - native `mcmc_smoke` on `notebook_4param`
  - speculative vectorized MCMC evaluation-budget runs on:
    - `notebook_4param`
    - `notebook_multilayer_12param`

Results:

- focused validation:
  - `tests/test_fitting/test_dean_optimization_trapezoid.py`
    - `12 passed`
  - `tests/test_fitting/test_dean_optimization_trapezoid_realistic.py`
    - `11 passed`
  - `tests/test_fitting/test_dean_mcmc_vectorized_eval_budget.py`
    - `4 passed`
- realistic DE timing, Dean CPU incumbent versus restored Dean GPU fused resident path:
  - `notebook_4param`, `one_generation`, population `128`:
    - CPU: about `0.702 s` per run
    - GPU: about `0.0647 s` per run
    - speedup: about `10.9x`
    - `GF` and `BIC` matched exactly
  - `notebook_multilayer_12param`, `one_generation`, population `64`:
    - CPU: about `3.77 s` per run
    - GPU: about `0.0806 s` per run
    - speedup: about `46.8x`
    - `GF` and `BIC` matched to floating-point noise
  - `notebook_4param`, `multi_generation`, population `256`:
    - CPU: about `1.41 s` per run
    - GPU: about `0.0419 s` per run
    - speedup: about `33.7x`
    - `GF` and `BIC` matched exactly
- realistic native MCMC smoke on `notebook_4param`:
  - CPU: about `0.185 s`
  - GPU-backed model: about `0.257 s`
  - speedup: about `0.72x`
  - effective samples, acceptance, `GF`, and `BIC` matched exactly
  - interpretation:
    - native notebook MCMC remains a scalar workflow-validation path, not a resident-GPU throughput win
- realistic vectorized MCMC evaluation-budget timing:
  - `notebook_4param`, walker batch `16`:
    - CPU: about `0.102 s`
    - GPU: about `0.0195 s`
    - speedup: about `5.2x`
    - best objective, `GF`, and `BIC` matched exactly
  - `notebook_4param`, walker batch `32`:
    - CPU: about `0.0972 s`
    - GPU: about `0.0105 s`
    - speedup: about `9.2x`
    - best objective, `GF`, and `BIC` matched exactly
  - `notebook_multilayer_12param`, walker batch `24`:
    - CPU: about `1.06 s`
    - GPU: about `0.0856 s`
    - speedup: about `12.3x`
    - best objective, `GF`, and `BIC` matched exactly
  - `notebook_multilayer_12param`, walker batch `48`:
    - CPU: about `1.06 s`
    - GPU: about `0.0432 s`
    - speedup: about `24.5x`
    - best objective, `GF`, and `BIC` matched exactly

Outcome:

- keep

Notes:

- this step restored the intended Dean GPU resident path on the current branch and converted the earlier analysis-only expectation into an actual measured win
- the restored path clearly improves realistic DE and vectorized-MCMC throughput while preserving result parity
- native notebook `CDSAXS_MCMC(...)` still does not become a GPU-time win from this step alone; it remains a parity and workflow-semantics validation path
- the next optimization step should target `_free_form_trapezoid_gpu(...)` directly, since the compatibility blocker is now removed

## 2026-03-25 / Stage 7 / Fused GPU Layer-Contribution Kernel

- area: trapezoid GPU form-factor kernel
- candidate: fuse the per-layer trapezoid contribution math into one CuPy fused kernel
- status: complete

Motivation:

- after restoring the resident path, `_free_form_trapezoid_gpu(...)` remained the dominant GPU-time cost
- the current loop still launched many elementwise operations per layer
- try reducing GPU launch overhead by fusing each layer contribution into one CuPy kernel while leaving the overall workflow and math intact

Files touched:

- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`

Implementation notes:

- added `_fused_trapezoid_layer_contribution(...)` alongside the existing fused residual helper
- replaced the explicit `a1`, `a2`, and `(1j / qx) * (a1 - a2) * sld` assembly in `_free_form_trapezoid_gpu(...)` with the fused helper in both:
  - batched GPU form-factor evaluation
  - scalar GPU form-factor evaluation

Benchmark protocol:

- validation:
  - `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_optimization_trapezoid.py -q`
  - `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_optimization_trapezoid_realistic.py -q`
  - `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting/test_dean_mcmc_vectorized_eval_budget.py -q`
  - `mamba run -n cdsaxs-dev python -m pytest tests/test_fitting -q`
- performance:
  - use the same realistic DE and MCMC timing families as Stage 6
  - first observation:
    - the very first timed GPU run in a fresh process included fused-kernel JIT cost and therefore understated the steady-state gain
  - gating comparison:
    - pre-warm representative GPU shapes with `run_realistic_candidate_budget_trial(...)`
    - then re-run realistic DE and vectorized-MCMC timing

Results:

- validation:
  - focused test modules still passed
  - `tests/test_fitting -q`
    - `35 passed`
- warmed realistic DE timing, Dean CPU incumbent versus fused-layer Dean GPU:
  - `notebook_4param`, `one_generation`, population `128`:
    - CPU: about `0.655 s`
    - GPU: about `0.0192 s`
    - speedup: about `34.1x`
    - versus Stage 6 GPU:
      - previous GPU: about `0.0647 s`
      - new GPU: about `0.0192 s`
      - improvement: about `3.37x`
    - `GF` and `BIC` matched exactly
  - `notebook_multilayer_12param`, `one_generation`, population `64`:
    - CPU: about `3.85 s`
    - GPU: about `0.0690 s`
    - speedup: about `55.7x`
    - versus Stage 6 GPU:
      - previous GPU: about `0.0806 s`
      - new GPU: about `0.0690 s`
      - improvement: about `1.17x`
    - `GF` and `BIC` matched to floating-point noise
  - `notebook_4param`, `multi_generation`, population `256`:
    - CPU: about `1.45 s`
    - GPU: about `0.0387 s`
    - speedup: about `37.5x`
    - versus Stage 6 GPU:
      - previous GPU: about `0.0419 s`
      - new GPU: about `0.0387 s`
      - improvement: about `1.08x`
    - `GF` and `BIC` matched exactly
- warmed realistic vectorized MCMC evaluation-budget timing:
  - `notebook_4param`, walker batch `16`:
    - CPU: about `0.0946 s`
    - GPU: about `0.0121 s`
    - speedup: about `7.8x`
    - versus Stage 6 GPU:
      - previous GPU: about `0.0195 s`
      - new GPU: about `0.0121 s`
      - improvement: about `1.61x`
    - best objective, `GF`, and `BIC` matched exactly
  - `notebook_4param`, walker batch `32`:
    - CPU: about `0.0952 s`
    - GPU: about `0.00653 s`
    - speedup: about `14.6x`
    - versus Stage 6 GPU:
      - previous GPU: about `0.0105 s`
      - new GPU: about `0.00653 s`
      - improvement: about `1.61x`
    - best objective, `GF`, and `BIC` matched exactly
  - `notebook_multilayer_12param`, walker batch `24`:
    - CPU: about `1.06 s`
    - GPU: about `0.0390 s`
    - speedup: about `27.3x`
    - versus Stage 6 GPU:
      - previous GPU: about `0.0856 s`
      - new GPU: about `0.0390 s`
      - improvement: about `2.19x`
    - best objective, `GF`, and `BIC` matched exactly
  - `notebook_multilayer_12param`, walker batch `48`:
    - CPU: about `1.08 s`
    - GPU: about `0.0208 s`
    - speedup: about `52.0x`
    - versus Stage 6 GPU:
      - previous GPU: about `0.0432 s`
      - new GPU: about `0.0208 s`
      - improvement: about `2.08x`
    - best objective, `GF`, and `BIC` matched exactly
- native notebook MCMC smoke on `notebook_4param`:
  - CPU: about `0.182 s`
  - GPU-backed model: about `0.243 s`
  - speedup: about `0.75x`
  - effective samples, acceptance, `GF`, and `BIC` matched exactly
  - interpretation:
    - the kernel win helps the vectorized GPU-relevant MCMC throughput path, not native scalar notebook MCMC timing

Outcome:

- keep

Notes:

- this candidate produced the first major post-compatibility reduction in resident GPU time
- the warmed benchmark protocol matters here because the new fused layer kernel has first-use JIT cost in a fresh process
- future GPU timing comparisons for this path should pre-warm representative shapes before making keep/reject decisions

## 2026-03-25 / Stage 8 / Single-Layer GPU Fast Path

- area: trapezoid GPU one-layer specialization
- candidate: dedicated `layers == 1` fast path inside `_free_form_trapezoid_gpu(...)`
- status: rejected

Motivation:

- after Stage 7, the base `notebook_4param` workflow was much faster, but it is still the common one-layer notebook case
- try a direct single-layer specialization that skips the generic loop and intermediate form accumulation

Files touched during experiment:

- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`

Benchmark protocol:

- validation while the experiment was applied:
  - `tests/test_fitting/test_dean_optimization_trapezoid.py`
  - `tests/test_fitting/test_dean_optimization_trapezoid_realistic.py`
  - `tests/test_fitting/test_dean_mcmc_vectorized_eval_budget.py`
- warmed performance comparison against the kept Stage 7 baseline:
  - pre-warm `notebook_4param` GPU objective batches
  - compare:
    - realistic DE `notebook_4param`, `one_generation`, population `128`
    - native notebook MCMC smoke on `notebook_4param`
    - vectorized MCMC evaluation-budget timing on `notebook_4param`

Results:

- validation passed during the experiment
- warmed base realistic DE:
  - Stage 7 GPU baseline: about `0.0192 s`
  - single-layer fast path candidate: about `0.0217 s`
  - result: regression, about `0.89x`
- warmed native notebook MCMC smoke:
  - Stage 7 GPU baseline: about `0.243 s`
  - single-layer fast path candidate: about `0.257 s`
  - result: regression, about `0.95x`
- warmed vectorized MCMC evaluation-budget on `notebook_4param`:
  - walker batch `16`:
    - Stage 7 GPU baseline: about `0.0121 s`
    - candidate: about `0.0118 s`
    - result: tiny gain, about `1.03x`
  - walker batch `32`:
    - Stage 7 GPU baseline: about `0.00653 s`
    - candidate: about `0.00698 s`
    - result: regression, about `0.94x`

Outcome:

- reject and revert

Notes:

- the specialization preserved parity but did not deliver a consistent or significant steady-state speedup over the kept Stage 7 fused-layer kernel
- the extra branch and code size were not justified by the mixed results

## 2026-03-25 / Final Appraisal / Realistic CPU vs Current Optimized GPU

- area: final realistic benchmark appraisal for the kept Stage 7 GPU path
- baseline: incumbent CPU `TrapezoidModelArrayDean`
- candidate: current optimized GPU `TrapezoidModelArrayDeanGPUFused`
- environment: `cdsaxs-dev`
- status: recorded

Benchmark protocol:

- warm representative GPU shapes before timing to avoid first-use JIT distortion
- run the realistic harnesses serially:
  - DE optimizer profiles
  - native notebook MCMC smoke profiles
  - vectorized MCMC evaluation-budget profiles
- compare only CPU versus the current kept GPU implementation
- require parity on `GF` / `BIC` and on the vectorized best-objective outputs

Results:

- DE, `notebook_4param`:
  - one generation, population `128`:
    - CPU: about `0.724 s`
    - GPU: about `0.0220 s`
    - speedup: about `32.8x`
    - `GF` and `BIC` matched exactly
  - multi generation, population `256`:
    - CPU: about `1.53 s`
    - GPU: about `0.0389 s`
    - speedup: about `39.4x`
    - `GF` and `BIC` matched exactly
- DE, `notebook_multilayer_12param`:
  - one generation, population `64`:
    - CPU: about `3.79 s`
    - GPU: about `0.0688 s`
    - speedup: about `55.0x`
    - `GF` and `BIC` deltas: about `9.09e-13`
  - multi generation, population `64`:
    - CPU: about `3.77 s`
    - GPU: about `0.0692 s`
    - speedup: about `54.4x`
    - `GF` and `BIC` deltas: about `9.09e-13`
- native notebook MCMC smoke, `notebook_4param`:
  - CPU: about `0.194 s`
  - GPU-backed model: about `0.257 s`
  - speedup: about `0.76x`
  - effective samples, acceptance, `GF`, and `BIC` matched exactly
- native notebook MCMC smoke, `notebook_multilayer_12param`:
  - CPU: about `1.18 s`
  - GPU-backed model: about `1.52 s`
  - speedup: about `0.78x`
  - effective samples, acceptance, `GF`, and `BIC` matched exactly
- vectorized MCMC evaluation-budget, `notebook_4param`:
  - walker batch `16`:
    - CPU: about `0.0946 s`
    - GPU: about `0.0116 s`
    - speedup: about `8.17x`
    - best objective, `GF`, and `BIC` matched exactly
  - walker batch `32`:
    - CPU: about `0.0905 s`
    - GPU: about `0.00650 s`
    - speedup: about `13.9x`
    - best objective, `GF`, and `BIC` matched exactly
- vectorized MCMC evaluation-budget, `notebook_multilayer_12param`:
  - walker batch `24`:
    - CPU: about `1.04 s`
    - GPU: about `0.0381 s`
    - speedup: about `27.3x`
    - best objective, `GF`, and `BIC` matched exactly
  - walker batch `48`:
    - CPU: about `1.02 s`
    - GPU: about `0.0284 s`
    - speedup: about `35.8x`
    - best objective, `GF`, and `BIC` matched exactly

Outcome:

- for realistic DE workflows, the current GPU path is a decisive win over the CPU incumbent
- for realistic vectorized MCMC throughput workflows, the current GPU path is also a strong win, with larger gains on the 12-parameter case
- for native notebook MCMC smoke workflows, the current GPU-backed path preserves parity but is still slower end-to-end than CPU

Recommendation:

- use the optimized GPU path for DE and vectorized MCMC evaluation-heavy workflows
- keep treating native notebook MCMC as a parity-verified path rather than a GPU-time win
