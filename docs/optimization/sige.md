# SiGe Optimization Plan

## Purpose

This page is the active roadmap for SiGe optimization.

It captures:

- the current state of the SiGe model
- how much trapezoid work can actually be reused
- the detailed resumable plan for a CPU-vectorized `SimTrap_GF` path
- the basic follow-on GPU plan

## Current Status As Of 2026-03-26

The next meaningful SiGe milestone is not "make `SiGe_model.py` use CuPy".

The next milestone is:

- establish a realistic package-native SiGe harness
- refactor the scalar SiGe objective into explicit helpers
- land a correct vectorized CPU objective path
- then evaluate how much of that path should move to GPU

Current important fact:

- the present SiGe batched objective path is not working as a true vectorized path

The existing `SimTrap_GF(...)` and `_trapezoid_optimization_wrapper(...)` are still scalar by structure:

- `src/cdsaxs/Fitting/SiGe_model.py`

## Resume Context

Repository and runtime context:

- repo: `/homes/deand/dev/nist_cdsaxs`
- branch: `feature/dean_optimization`
- env: `cdsaxs-dev`
- editable install expected

Realistic reference workflow:

- notebook: `/resdata/DeLongchamp/scratchdisk/nist_cdsaxs_sige/CDSAXS_imec_d24_DFS1_ModelV2.ipynb`
- reduced data: `/resdata/DeLongchamp/scratchdisk/nist_cdsaxs_sige/SMI_imec_nsh_d24b_D260126_T104543_reduced_results.csv`

Important observations from that workflow and the current code:

- the realistic notebook uses `importCDSAXS_GUI(...)`
- the realistic notebook uses scalar DE with `workers`, not `vectorized=True`
- the realistic design stack includes typed layers, especially `Layer_Type: 'Ellipse'`
- on realistic notebook-style load, SiGe expands from a design stack to a larger simulation stack
- the current 2D candidate path does not yet return a proper GF vector for SiGe

Key code anchors for resumption:

- `src/cdsaxs/Fitting/CDSAXS_base_model.py`
  - `CDSAXS_Optimize(...)`
  - `_resolve_active_parameter_names(...)`
- `src/cdsaxs/Fitting/SiGe_model.py`
  - `_apply_constraints(...)`
  - `_expand_typed_layers(...)`
  - `_ensure_expanded_model_params(...)`
  - `SimTrap_GF(...)`
  - `_trapezoid_optimization_wrapper(...)`
- trapezoid optimization references:
  - `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
  - `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`

## Reuse Appraisal

### Reusable Now

These trapezoid results can be reused immediately:

- optimizer-facing 1D and 2D objective semantics
- active parameter-name resolution
- realistic harness structure
- dataset invariant caching
- parameter-plan caching
- candidate normalization for `(S, D)` and `(D, S)`
- CPU-versus-batched parity strategy
- GPU dataset residency and fused residual reduction patterns

### Reusable With Adaptation

These ideas are reusable, but need SiGe-specific implementation:

- parameter-plan compilation
- batched parameter application
- batched coordinate construction
- realistic DE and candidate-budget timing helpers
- parity tests between incumbent scalar and new vectorized paths

### Not Reusable As-Is

These trapezoid pieces should not be treated as drop-in SiGe solutions:

- direct `PAR`-only mapping logic
- trapezoid `SymCoordAssign(...)` assumptions
- trapezoid geometry kernels
- trapezoid GPU form-factor kernels
- tests that assume design parameters map directly to simulation parameters without expansion

## Naming For New SiGe Work

Do not introduce new `Dean` names.

Recommended future files and classes:

- `src/cdsaxs/Fitting/SiGe_model_vectorized.py`
  - `SiGeModelArray_vectorized`
  - `SiGeModel_vectorized`
- `src/cdsaxs/Fitting/SiGe_model_vectorized_GPU.py`
  - `SiGeModelArray_vectorized_GPU`
  - `SiGeModel_vectorized_GPU`
- `src/cdsaxs/Fitting/optimization_sige_realistic.py`
- `tests/test_fitting/test_sige_model_vectorized.py`
- `tests/test_fitting/test_optimization_sige_realistic.py`
- later, if needed:
  - `tests/test_fitting/test_sige_model_vectorized_gpu.py`

The existing `SiGe_model.py` should remain the scalar reference and fallback during the first implementation phase.

## Detailed Plan: CPU Vectorized `SimTrap_GF`

Goal:

- make SiGe support a real batched objective path compatible with `vectorized=True`
- preserve current scalar workflow semantics
- keep scalar SiGe as the reference implementation while the vectorized path is brought up

### Invariants To Hold Fixed

The first vectorized implementation should assume these are fixed within one optimization run:

- design-layer count
- typed-layer topology
- `Layer_Type`
- `num_layers` for discretized layers such as ellipses
- constraint graph
- imported data shape

These assumptions match the realistic notebook-style optimization surface and make batched preprocessing tractable.

### Stage 0: Build A Realistic SiGe Harness First

Create a package-native harness before changing internals.

Recommended file:

- `src/cdsaxs/Fitting/optimization_sige_realistic.py`

Required scope:

- recreate the realistic imec SiGe workflow in package code
- load the reduced CSV through the package path, not through notebook-only `sys.path` manipulation
- reconstruct the realistic design stack, optimization bounds, and constraint setup
- provide short reproducible helpers for:
  - model construction only
  - objective evaluation only
  - one-generation DE smoke

Why this comes first:

- it gives a stable target for correctness and timing
- it avoids doing vectorization work against synthetic or incomplete SiGe setups
- it provides the same role for SiGe that `dean_optimization_trapezoid_realistic.py` provides for trapezoid

### Stage 1: Refactor Scalar SiGe Objective Into Explicit Helpers

Before any real batching, split the scalar path into reusable helpers inside `SiGe_model_vectorized.py`.

Recommended helper boundaries:

1. normalize optimization input
2. resolve active parameter names
3. compile a SiGe-specific parameter plan
4. copy design-level state for one candidate
5. apply candidate values to design-level parameters
6. apply constraints
7. expand typed layers into a pure simulation stack
8. materialize dense expanded arrays for geometry, SLD, `DW`, `I0`, and `Bk`
9. evaluate form factor and GF from the expanded arrays

The first pass should prefer clarity and parity over speed.
The scalar oracle should remain easy to compare against.

### Stage 2: Make 2D Candidate Input Correct Before Making It Fast

The first vectorized milestone is interface correctness.

Required behavior:

- accept 1D input and return a scalar
- accept 2D input in either `(S, D)` or `(D, S)` form
- return an `S`-length GF vector for 2D input
- route through `_resolve_active_parameter_names(...)` correctly

Implementation guidance:

- in the first pass, the 2D path may loop over candidates internally after normalization
- that is acceptable if it produces the correct vector output and unlocks `vectorized=True`
- this stage is about fixing optimizer compatibility, not about immediate speedup

Exit criteria:

- `vectorized=True` no longer fails due to shape handling
- batched output shape is correct and finite for a realistic smoke workload

### Stage 3: Compile A SiGe Parameter Plan

After interface correctness, reduce Python overhead.

The SiGe parameter plan should capture:

- which parameters target design trapezoids
- which target design SLDs
- which target `DW`, `I0`, scalar `Bk`, or column backgrounds
- how each parameter maps into design-level arrays

Important difference from trapezoid:

- the plan must target design-layer state, not only expanded simulation arrays
- constraints and typed-layer expansion occur after design-level updates

Recommended outcome:

- one cached compiled plan per active `param_names` sequence
- no repeated string parsing inside the hot path

### Stage 4: Batch The Design-Level Preprocessing

Once the plan exists, batch the candidate-to-design-state mapping.

Recommended dense batched arrays:

- design widths
- design heights
- design top widths
- design depths where applicable
- design SLDs
- `DW`
- `I0`
- `Bk`

At this stage, some operations may still remain Python loops if needed:

- constraint application
- typed-layer expansion

That is acceptable for the first performance pass, because it still removes repeated string parsing and state mutation overhead.

### Stage 5: Batch Constraints And Typed-Layer Expansion

This is the critical SiGe-specific middle layer.

Work items:

- compile equality and inequality constraints into index-based operations
- apply those constraints over batched design arrays
- precompute a design-to-expanded mapping for fixed topology
- expand batched typed layers into dense batched simulation arrays

Important note:

- this stage is where SiGe diverges most strongly from trapezoid
- it is also the stage most likely to dominate runtime before any GPU work

Design target:

- for a fixed realistic workflow, the expanded topology should be stable across candidates
- expansion should fill preallocated dense arrays rather than rebuild Python dict lists every time

### Stage 6: Vectorize The Expanded-Stack Numerical Tail

After the preprocessing shape is stable, move the dominant numerical tail into real batched NumPy operations.

Reuse from trapezoid:

- cached `log(Intensity)`
- cached `Qx**2 + Qz**2`
- batched Debye-Waller application
- batched residual reduction

Do not force reuse:

- trapezoid form-factor kernels

Implementation guidance:

- if the existing SiGe numerical helpers are hard to batch safely, add batched helpers in the vectorized module
- keep the scalar implementation available as a fallback until parity is established

### Stage 7: Integrate With Realistic DE And Public Exports

Once batched SiGe evaluation is numerically stable:

- add the realistic DE smoke runner in `optimization_sige_realistic.py`
- run short seeded vectorized DE
- expose the new class through `src/cdsaxs/Fitting/__init__.py` only when the implementation is ready for use

Do not switch the public `SiGeModel` alias to the new class until:

- scalar parity is established
- short vectorized DE smoke passes
- the realistic harness is present

## Can Existing CPU-Array Logic Test This?

Yes, but only partially.

Yes:

- the existing optimizer contract already supports vectorized objective calls
- trapezoid CPU-array logic provides the correct interface pattern
- scalar-versus-batch parity tests are directly applicable as a testing strategy
- realistic objective-only and short DE smoke harnesses can follow the trapezoid pattern closely

No:

- the trapezoid CPU-array implementation is not a drop-in SiGe evaluator
- its direct `PAR` mapping logic does not handle design-level constraints or typed-layer expansion
- existing SiGe 2D handling cannot be trusted as the validation oracle because that is part of what is broken

Correct testing stance:

- use incumbent scalar SiGe as the oracle
- use trapezoid vectorized logic as the pattern
- validate SiGe batching against realistic SiGe workflows, not only synthetic arrays

## Recommended Test Matrix For The CPU Vectorized Phase

Add tests in at least these groups:

1. shape and interface tests
   - 1D candidate returns scalar
   - `(S, D)` and `(D, S)` inputs both work
   - 2D input returns an `S`-length vector

2. scalar-versus-batch parity tests
   - same candidate values through scalar and 2D-singleton batch agree
   - shared candidate matrices agree candidate-by-candidate

3. design-processing parity tests
   - constraints match scalar behavior
   - typed-layer expansion matches scalar behavior
   - realistic ellipse-layer cases match scalar behavior

4. realistic harness smoke tests
   - realistic model build succeeds
   - objective-only smoke returns finite values
   - one-generation vectorized DE smoke returns finite `GF`

5. regression guards
   - objective evaluation does not mutate persistent model state
   - realistic imported-data workflow stays package-native

## Suggested First Implementation Slice

If resuming from a new context, the first code slice should be:

1. add `optimization_sige_realistic.py`
2. add `SiGe_model_vectorized.py` as a new class that initially wraps scalar logic cleanly
3. make 2D input return a real GF vector, even if implemented by looping internally
4. add parity and shape tests

This is the smallest change set that unlocks real vectorized optimizer integration without committing too early to a complex kernel design.

## Basic Follow-On Plan: GPU SiGe

The GPU phase should begin only after the CPU vectorized phase is correct and measurable.

### Preconditions

- realistic SiGe harness exists
- CPU vectorized path is correct
- short vectorized DE smoke works
- profiling identifies where time actually goes

### GPU V1 Scope

Recommended file:

- `src/cdsaxs/Fitting/SiGe_model_vectorized_GPU.py`

Recommended first scope:

- keep optimizer on CPU
- keep design-level mapping, constraints, and typed-layer expansion on CPU
- move batched expanded-stack form-factor evaluation and GF reduction to GPU
- reuse GPU dataset residency and fused residual reduction patterns from the trapezoid GPU path
- add a batch-size threshold so small workloads fall back to CPU vectorized execution

Why this split first:

- it is the lowest-risk way to reuse the existing GPU optimization architecture
- it avoids blocking on a full GPU rewrite of design-layer preprocessing
- it will quickly reveal whether the numerical tail is dominant enough to justify more GPU work

### GPU V2 Only If Profiling Justifies It

Consider more GPU migration only if CPU preprocessing still dominates after V1.

Possible later targets:

- batched constraint application on device
- typed-layer expansion on device
- GPU-resident coordinate construction

Do not start here.

## Recommended Test Matrix For The GPU Phase

1. CPU-vectorized versus GPU-vectorized parity on shared candidate batches
2. finite objective results on realistic batch smoke
3. explicit fallback behavior on small or unsupported workloads
4. realistic short vectorized DE smoke with GPU enabled

## Decision Rule

The working decision rule for SiGe remains:

- CPU vectorization first
- GPU second
- keep scalar SiGe as the oracle until vectorized SiGe is stable
- only broaden abstraction after a second geometry actually benefits from it
