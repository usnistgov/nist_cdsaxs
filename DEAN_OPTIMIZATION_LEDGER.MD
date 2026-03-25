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
- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`

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
