# SiGe Optimization Plan

## Purpose

This page is the active roadmap for SiGe optimization.

It captures:

- the current state of the SiGe model
- how much trapezoid work can actually be reused
- the detailed resumable plan for a CPU-vectorized `SimTrap_GF` path
- the basic follow-on GPU plan

## Current Status As Of 2026-04-02

The realistic imec ellipse-stack workflow now has a package-native CPU-vectorized path.

Completed in the current pass:

- `src/cdsaxs/Fitting/optimization_sige_realistic.py` provides the realistic harness, objective helpers, and timing helpers
- `src/cdsaxs/Fitting/SiGe_model_vectorized.py` provides the new realistic batched SiGe implementation
- `src/cdsaxs/Fitting/__init__.py` exports the new SiGe vectorized entry points
- targeted tests landed for realistic shape handling, parity, and seeded DE behavior

Current important facts:

- realistic `vectorized=True` SiGe is now functionally correct on the ellipse-stack workflow
- the incumbent scalar implementation in `src/cdsaxs/Fitting/SiGe_model.py` remains the physics oracle and fallback
- `curved_sides` is still intentionally out of scope for this first realistic vectorized pass
- CPU speedup is not yet in place on the CPU-vectorized path, but the active kept GPU path now exists and outperforms both scalar CPU and the older GPU loop path
- the next SiGe priority is now convergence-equivalence testing on realistic optimization runs rather than more raw candidate-evaluation microbenchmarking

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
- the realistic notebook does not require `curved_sides` for the first vectorization milestone
- on realistic notebook-style load, SiGe expands from a design stack to a larger simulation stack
- the realistic production target for the current pass is the ellipse-stack workflow, not the optional `curved_sides` path

Validated runtime findings from the realistic imec workload:

- realistic imported data shape is `(121, 49)`
- the realistic design stack has `11` design entries and expands to `26` simulation trapezoids
- the realistic optimization surface currently resolves to `26` active parameters
- scalar default objective evaluation is finite on the realistic harness
- the realistic vectorized objective path now returns correct finite values for:
  - 1D input
  - singleton 2D batches
  - true multi-candidate `(S, D)` and `(D, S)` batches
- short seeded `differential_evolution(vectorized=True)` now completes with exact final `GF` parity versus the scalar realistic run for the fixed timing configuration

Implementation status update as of 2026-04-02:

- `src/cdsaxs/Fitting/SiGe_model_vectorized.py` now exists
- `src/cdsaxs/Fitting/optimization_sige_realistic.py` now exists
- realistic `vectorized=True` SiGe is now functionally correct on the imec ellipse-stack workflow
- the realistic batched path is end-to-end array-based across candidates for:
  - active-parameter application
  - constraints
  - typed ellipse expansion
  - form-factor accumulation
  - GF reduction
- parity against incumbent scalar CPU mode is established for:
  - 1D objective evaluation
  - `(S, D)` and `(D, S)` batched objective evaluation
  - short seeded realistic DE runs
- measured parity quality on the realistic objective path is at floating-point noise level:
  - max absolute `GF` difference in batched objective checks: about `9.09e-13` to `1.82e-12`
  - realistic DE final `GF` matched exactly in the fixed-generation timing comparison
- current speed result:
  - objective throughput is about parity to slightly slower than scalar CPU
  - fixed-generation realistic DE timing is currently slower than scalar CPU by about `0.937x`
- current dominant hotspot:
  - `_batched_form_factor(...)` in `src/cdsaxs/Fitting/SiGe_model_vectorized.py`
  - profiling shows the remaining cost is concentrated in the Python layer-accumulation loop, not in batched parameter application, constraints, ellipse expansion, or GF reduction
- full layer-axis NumPy broadcast reduction has now been tried once and is not a kept result:
  - the all-at-once `(candidate, layer, qz, qx)` reduction increased temporary pressure enough to slow the realistic CPU path further
  - reduction-order changes also loosened objective parity from about `1e-12` to about `5.46e-12`
  - conclusion: a naive full-4D NumPy rewrite is not the right kept CPU solution for this kernel
- current CPU interpretation after fresh profiling:
  - the realistic batched path is already end-to-end vectorized at the workflow level
  - the remaining performance issue is that `_batched_form_factor(...)` still spends almost all wall time in elementwise complex math over large CPU arrays
  - candidate batching on CPU increases array size and temporary traffic more than it reduces useful overhead
- `src/cdsaxs/Fitting/SiGe_model_vectorized_GPU.py` now contains multiple realistic GPU layer-reduction variants:
  - `loop`
  - `4d_naive`
  - `4d_tiled_layers`
  - `4d_tiled_layers_candidates`
  - `4d` / `4d_rawkernel`
- the kept realistic GPU default is now the `4d` raw-kernel path
- the GPU loop path remains available as an opt-in comparison path by setting `_vectorized_gpu_layer_algorithm = "loop"`
- the naive full-broadcast 4D GPU path is now retained only as a rejected experimental reference
- blocked broadcast 4D GPU variants were tested in this session and improved over the naive 4D broadcast path
- the custom raw-kernel 4D GPU reducer was then tested and became the kept result
- realistic DE-shaped candidate-evaluation throughput now strongly favors GPU:
  - practical DE batch `104` (`popsize=4`, `26` parameters):
    - scalar CPU: about `28.37` candidates/s
    - opt-in GPU loop path: about `3328.71` candidates/s
    - kept GPU 4D path: about `3976.56` candidates/s
  - the kept GPU 4D path is about `1.19x` faster than the GPU loop path at the practical DE batch
  - on the scanned DE-shaped ladder up to batch `1040`, the kept GPU 4D path continued improving and reached about `4279.72` candidates/s at the largest tested batch
- fixed-generation realistic DE timing now strongly favors the kept GPU 4D path:
  - scalar CPU: about `9.41 s`
  - opt-in GPU loop path: about `0.179 s`
  - kept GPU 4D path: about `0.149 s`
- parity remains within floating-point noise on the kept GPU path:
  - GPU loop versus scalar CPU objective checks: about `6e-12` to `1e-11`
  - kept GPU 4D versus scalar CPU objective checks: about `6e-12` to `1e-11`
  - kept GPU 4D versus GPU loop objective checks: about `0.0` to `1.82e-12`

Validation completed in `cdsaxs-dev`:

- `pytest -q tests/test_fitting/test_sige_model_vectorized.py tests/test_fitting/test_optimization_sige_realistic.py`
- result: `9 passed`
- `pytest -q tests/test_fitting/test_sige_model_vectorized_gpu.py`
- result: `6 passed`

Current measured realistic timing snapshot:

- objective throughput speedup, scalar versus current CPU-vectorized path:
  - batch `1`: about `1.018x`
  - batch `4`: about `0.981x`
  - batch `8`: about `0.972x`
  - batch `16`: about `0.971x`
  - batch `32`: about `0.962x`
- fixed-generation realistic DE timing with `seed=1234`, `maxiter=2`, `popsize=4`, `tol=0.0`, `polish=False`, `updating='deferred'`:
  - scalar CPU: about `9.12 s`
  - current CPU-vectorized path: about `9.74 s`
  - speedup: about `0.937x`

Most important interpretation:

- correctness is now in place
- meaningful CPU speedup is not yet in place
- the kept realistic GPU path is now in place and should be treated as the default GPU evaluator for realistic SiGe work
- the opt-in GPU loop path remains useful as:
  - a fallback implementation
  - a parity comparison path
  - a lower-risk debugging path when the raw-kernel implementation needs isolation
- additional raw candidate-throughput work is no longer the most valuable next SiGe milestone
- the next priority is now a convergence-equivalence test:
  - verify that scalar CPU, CPU-vectorized, GPU loop, and kept GPU 4D realistic optimization runs converge to equivalent results under matched seeds and optimizer settings
  - emphasize optimizer trajectory and final-fit equivalence, not only isolated candidate-evaluation parity
  - keep the existing candidate-evaluation timing harness as support tooling rather than the main scientific gate

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

Additional first-milestone scope restriction:

- target the realistic ellipse-layer workflow first
- keep `curved_sides` on the incumbent scalar path until the ellipse-driven CPU vectorized path is correct and benchmarked
- only generalize the first vectorized implementation to `curved_sides` if a real workload requires it

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
- short seeded realistic DE with `vectorized=True` must change the objective away from the initial default value rather than immediately degenerating to all-`inf` behavior

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

Current status of Stage 6:

- a first realistic batched numerical-tail implementation is now present in `SiGe_model_vectorized.py`
- it delivers parity, but not a CPU speedup on the realistic imec workflow
- therefore Stage 6 should not be considered complete yet

Next Stage 6 priority:

- re-audit the realistic CPU batched path for any remaining structure that is vectorized only at the API level but not yet efficient in the numerical kernel
- explicitly inspect:
  - per-layer Python control flow in the form-factor accumulation
  - temporary array creation volume
  - reuse of cached dataset invariants
  - whether the current batched kernel is memory-bandwidth bound rather than Python-overhead bound
- only after that audit should the team decide whether the next CPU step is:
  - further NumPy kernel restructuring
  - a compiled CPU kernel
  - or shifting priority to the GPU numerical tail

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
- treat the realistic ellipse-layer workflow as the primary CPU-vectorization benchmark surface
- treat `curved_sides` as an explicit fallback case until there is a dedicated vectorized implementation

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

6. performance benchmarks
   - keep these in the realistic harness, not in unit-test-only synthetic fixtures
   - measure both objective throughput and optimizer wall time
   - use seeded DE so scalar and vectorized runs consume comparable candidate streams

Recommended CPU-vectorized benchmark pair:

1. objective throughput benchmark
   - file: `src/cdsaxs/Fitting/optimization_sige_realistic.py`
   - build a deterministic candidate matrix near the realistic default vector
   - run scalar incumbent candidate-by-candidate
   - run vectorized candidate batches with the same candidate matrix
   - report:
     - batch size
     - total candidates
     - wall time
     - candidates per second
     - max absolute GF difference versus scalar oracle
   - suggested batch ladder for CPU bring-up:
     - `1, 4, 8, 16, 32`

2. fixed-generation differential evolution timing
   - file: `src/cdsaxs/Fitting/optimization_sige_realistic.py`
   - compare:
     - incumbent scalar path: `vectorized=False`
     - candidate CPU-vectorized path: `vectorized=True`
   - keep the workload fixed with:
     - realistic imec SiGe harness
     - `seed=1234`
     - `maxiter=2`
     - `popsize=4`
     - `polish=False`
     - `tol=0.0`
     - `updating='deferred'`
   - report:
     - elapsed seconds
     - final `GF`
     - number of parameters
     - implied population size
     - estimated objective evaluations
   - the fixed-generation DE benchmark should be marked slow and used as a benchmark gate, not a default smoke test

Current measured realistic results from the implemented CPU-vectorized path:

- objective throughput, scalar versus CPU-vectorized candidate model:
  - batch `1`: about `1.01x`
  - batch `4`: about `0.97x`
  - batch `8`: about `0.97x`
  - batch `16`: about `0.98x`
  - batch `32`: about `0.97x`
- fixed-generation realistic DE timing with:
  - `seed=1234`
  - `maxiter=2`
  - `popsize=4`
  - `polish=False`
  - `tol=0.0`
  - `updating='deferred'`
  - result:
    - scalar CPU: about `9.21 s`
    - current CPU-vectorized path: about `9.88 s`
    - speedup: about `0.93x`
    - final `GF` matched exactly

Decision rule for keeping the CPU-vectorized SiGe path:

- first gate: correctness
  - vectorized DE must produce finite `GF`
  - vectorized objective batches must match scalar oracle within agreed tolerance
- second gate: speed
  - the vectorized path should beat the scalar incumbent on the realistic harness at batch sizes that actually occur in DE
  - if the DE timing does not improve materially after interface correctness is fixed, move next to profiling before writing GPU code
- current status against the speed gate:
  - not yet passed
  - current realistic DE timing is still slower than scalar CPU
- practical initial success criterion:
  - at least parity on fixed-generation DE wall time after PR1
  - clear speedup in objective throughput after batched preprocessing lands

## Suggested First Implementation Slice

If resuming from a new context, the first code slice should be:

1. add `optimization_sige_realistic.py`
2. add `SiGe_model_vectorized.py` as a new class that initially wraps scalar logic cleanly
3. make 2D input return a real GF vector, even if implemented by looping internally
4. add parity, shape, and realistic performance benchmarks
5. keep the first implementation explicitly scoped to the realistic ellipse-layer workflow and leave `curved_sides` on scalar fallback

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

### GPU V1.5: 4D Layer Reduction Results

The SiGe GPU file now contains several 4D experiments in
`src/cdsaxs/Fitting/SiGe_model_vectorized_GPU.py`:

- `4d_naive`
  - historical full-broadcast CuPy reference
- `4d_tiled_layers`
  - blocked layer-axis CuPy broadcast reduction
- `4d_tiled_layers_candidates`
  - blocked layer-axis plus candidate-axis CuPy broadcast reduction
- `4d` / `4d_rawkernel`
  - custom raw-kernel layer reducer

Kept result from this session:

- the kept realistic GPU default is now `4d`
- the kept `4d` implementation is the raw-kernel reducer
- the GPU loop path remains available as an explicit opt-in by setting `_vectorized_gpu_layer_algorithm = "loop"`
- the naive full-broadcast 4D CuPy path is now a rejected experimental reference only

Measured stepwise acceptance sequence on the realistic harness:

1. blocked layer tiling beat the naive 4D broadcast baseline by about `1.15x` at the best tested tile while preserving parity
2. blocked candidate-plus-layer tiling then beat the accepted blocked-layer baseline by about `1.18x` while preserving parity
3. the custom raw-kernel reducer then beat the accepted blocked candidate-plus-layer baseline by about `1.82x` on objective throughput and about `1.29x` on fixed-generation DE timing while preserving parity

Current realistic interpretation:

- the original naive 4D GPU loss was a temporary-allocation and memory-traffic problem, not a proof that 4D reduction was the wrong direction
- shrinking broadcast tiles helped, which supports the temporary-pressure interpretation
- eliminating the broadcasted `a1` and `a2` temporaries entirely with a custom reducer helped much more, which is why the raw-kernel path is now the kept result

Measured realistic DE-shaped candidate-evaluation scaling, opt-in GPU loop versus kept GPU 4D:

- practical DE batch `104` (`popsize=4`, `26` parameters):
  - GPU loop: about `3328.71` candidates/s
  - kept GPU 4D: about `3976.56` candidates/s
  - 4D relative speed: about `1.19x`
- larger DE-shaped batches continued to favor kept GPU 4D:
  - batch `260` (`popsize=10`): about `1.19x`
  - batch `416` (`popsize=16`): about `1.22x`
  - batch `624` (`popsize=24`): about `1.25x`
  - batch `1040` (`popsize=40`): about `1.25x`

Current best observed throughput in the tested realistic DE-shaped ladder:

- opt-in GPU loop peak: about `3505.53` candidates/s at batch `260` (`popsize=10`)
- kept GPU 4D peak so far: about `4279.72` candidates/s at batch `1040` (`popsize=40`)
- the kept GPU 4D path was still slowly improving at the top of the scanned ladder, so a hard throughput optimum has not yet been established

Fixed-generation realistic DE timing with `seed=1234`, `maxiter=2`, `popsize=4`, `tol=0.0`, `polish=False`, `updating='deferred'`:

- scalar CPU: about `9.41 s`
- opt-in GPU loop path: about `0.179 s`
- kept GPU 4D path: about `0.149 s`

Parity status for the kept GPU 4D line:

- kept GPU 4D versus scalar CPU objective checks stayed at about `6e-12` to `1e-11`
- kept GPU 4D versus opt-in GPU loop objective checks stayed at about `0.0` to `1.82e-12`
- realistic seeded DE final `GF` matched exactly across the compared paths

Next step for SiGe after the current GPU keep decision:

- stop prioritizing more raw candidate-throughput work as the main milestone
- develop a realistic convergence-equivalence test for SiGe:
  - scalar CPU versus CPU-vectorized
  - scalar CPU versus opt-in GPU loop
  - scalar CPU versus kept GPU 4D
  - matched seeds and optimizer settings
  - compare convergence behavior and final-fit equivalence, not only per-candidate objective parity

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
