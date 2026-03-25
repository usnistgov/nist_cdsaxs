# DEAN Optimization Action Plan

## Purpose

Improve fitting throughput in `nist_cdsaxs` while preserving:

- nearly identical user-facing workflows
- repo structure parallel to the existing fitting package
- scientific parity on fit quality and structure interpretation

The current priority order is:

1. trapezoid GPU optimization for realistic vectorized workflows
2. benchmark coverage for short MCMC workflows that reflect real usage
3. SiGe fitting support and optimization, now a high-priority geometry for the team
4. cylinder follow-on work after trapezoid and SiGe are on firmer ground

## Current Branch Context

- repo: `/homes/deand/dev/nist_cdsaxs`
- working branch: `feature/dean_optimization`
- environment: `cdsax-dev`
- install mode: editable
- GPU host used so far:
  - `3 x Quadro RTX 8000`
  - `49152 MiB` each
- GPU package support:
  - `cupy-cuda12x`
- dev dependencies needed for the rebased package:
  - `scikit-image`
  - `scikit-learn`

## Current Artifacts

Added optimization and harness files:

- `DEAN_OPTIMIZATION.md`
- `DEAN_OPTIMIZATION_LEDGER.MD`
- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py`
- `tests/test_fitting/test_dean_optimization_trapezoid.py`
- `tests/test_fitting/test_dean_optimization_trapezoid_realistic.py`
- `tests/test_fitting/test_dean_optimization_structure_similarity.py`
- `tests/test_fitting/test_dean_workflow_semantics.py`

## Current Harness And Test Status

The prerequisite harness and test refactor for resuming trapezoid GPU work is now in place.

Completed:

- layer-growth now preserves the concrete runtime class, including Dean CPU and GPU variants
- layer-growth normalizes `slds` to the new layer count instead of requiring a harness-only repair
- copied-data layer-growth models re-run their normal data initialization path
- `parameter_sweep_2d` is exposed again as a compatibility entry point for the notebooks
- trapezoid wrapper parameter resolution now selects the name set that matches the active workflow state
- unconditional optimizer and Cython import debug prints have been removed from routine runs
- realistic trapezoid profiles now build through `add_layer_at_percentage(...)` and `add_multiple_layers(...)` instead of cloned parameter dictionaries
- notebook-parallel realistic optimization ladders are now fixed at:
  - `notebook_4param`: geometry plus `DW`
  - layered `8`- and `12`-parameter profiles: geometry plus `DW`, `I0`, and scalar `Bk`, without auto-added `sld_*`
- workflow-native runners now cover DE, dual annealing, batch initialization, MCMC, and sweep replay smoke paths
- fitting coverage is now split between:
  - semantic workflow regressions in `tests/test_fitting/test_dean_workflow_semantics.py`
  - benchmark and harness smoke coverage in the existing Dean optimization test modules
- current local validation baseline:
  - `mamba run -n cdsax-dev python -m pytest tests/test_fitting -q`
  - `27 passed`

Remaining follow-on work after GPU optimization resumes:

- strengthen regression coverage only for workflow-native paths that prove benchmark value
- extend opt-in MCMC benchmarking beyond smoke/trial level where it remains informative

## Current Harness Tuning Surface

This is what the current trapezoid harness can and cannot tune directly.

### Objective Batch Tuning

- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py` can sweep raw batched objective size independently of any optimizer budget.
- named batch-size profiles already exist for:
  - `smoke`
  - `broadcast`
  - `gpu_candidate`
- use this path when the question is:
  - GPU launch overhead versus useful work
  - vectorization crossover point
  - sensitivity to candidate-matrix chunk size alone

### Realistic DE Tuning

- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py` exposes realistic DE tuning through:
  - profile
  - model class
  - `freeform_use_cupy`
  - `population_sizes`
  - `maxiter`
  - `tol`
  - `polish`
  - `seed`
  - `repeats`
- in the realistic DE harness, `population_size` plays two roles at once:
  - SciPy DE `popsize`
  - generation-sized vectorized evaluation chunk
- the current realistic candidate-budget harness formalizes this as:
  - `candidate_count = parameter_count * population_size`
- the older lightweight DE harness still exposes:
  - `vectorized`
  - CPU `workers`
  - `popsize`
  for scalar-versus-vectorized and CPU-parallel comparisons

Current DE limitation:

- the realistic harness cannot yet decouple optimizer `popsize` from vectorized evaluation chunk size
- DE `mutation` and `recombination` are still fixed inside the realistic helper instead of being first-class tuning knobs

### Realistic MCMC Tuning

- the current realistic MCMC harness exposes:
  - profile
  - family: `mcmc_smoke` or `mcmc_trial`
  - model class
  - `n_walkers`
  - `n_steps`
  - `burn_in`
  - `thin`
- the core `CDSAXS_MCMC(...)` API still accepts additional `emcee` kwargs, but the realistic harness wrapper does not yet surface them directly

Current MCMC limitation:

- the current MCMC path is not walker-vectorized in the same sense that DE generations are vectorized
- changing `n_walkers` changes total sampler work, but does not currently create a clean DE-like GPU batch-chunk control
- this means current MCMC runs are useful for workflow validation and coarse throughput trends, but not yet as a strong GPU chunk-shape tuning harness

### Current Validation State

What is already valid:

- DE speed harnesses exist for:
  - raw objective batch timing
  - realistic candidate-budget timing
  - realistic short DE timing
- DE validation exists through:
  - exact realistic profile-shape assertions
  - objective-value parity checks
  - fixed-seed structure-similarity regression
- MCMC validation exists through:
  - smoke and trial runners
  - positive acceptance and effective-sample checks
  - finite `GF` and `BIC`
  - stateful sweep-to-MCMC workflow-consistency coverage

What is still missing:

- a robust wall-clock-to-convergence harness
- a time-to-threshold comparison such as:
  - first finite fit below a target `GF`
  - best `GF` reached after equal wall-clock budget
- a stronger CPU-versus-GPU MCMC parity regression comparable to the current DE structure-similarity check

### Current Interpretation Rule

- the current harness can answer:
  - which path evaluates more candidates per second
  - which path is faster under the same short DE or MCMC budget
  - whether core workflow semantics still hold
- the current harness cannot yet answer:
  - which path gets to fit convergence first in a robust optimizer-level sense

Working inference, not yet a gated conclusion:

- current results already show that GPU advantage grows as vectorized candidate sets or DE generation sizes grow
- because of that, if a GPU path already beats the scalar incumbent under the current short fixed-budget comparisons, it is reasonable to expect the advantage to widen at larger `popsize` and longer runs
- treat that as a planning assumption, not as a proven convergence claim, until the harness includes explicit time-to-threshold measurements

Rebase and compatibility updates already applied:

- merged `README.md`
- merged `src/cdsaxs/__init__.py`
- merged `src/cdsaxs/Fitting/__init__.py`
- fixed `src/cdsaxs/Fitting/SiGe_model.py` to use relative import of `CDSAXS_base_model`
- exposed `create_model` at top-level `cdsaxs` package to support workflow helpers that import it from `cdsaxs`

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
   Reserve `*Array` and `*Dean*` names for implementation and explicit benchmark selection.

4. Separate algorithm families.
   Objective microbenchmarks, DE timing, batch initialization, and MCMC should not be mixed into one number.

5. Separate incumbents by workload shape.
   There is not one universal incumbent anymore.

6. Keep scientific gating proportional to value.
   Validate speedup first.
   Tighten structure-similarity and stochastic regression only for candidates worth keeping.

## Rebased Repo Read

The rebased repo changed the fitting landscape in ways that matter for the harness.

### Public And Workflow Surface

The key public and workflow entry points now include:

- `src/cdsaxs/Fitting/CDSAXS_base_model.py`
  - `CDSAXS_Optimize`
  - `CDSAXS_MCMC`
  - `add_layer_at_percentage`
  - `add_multiple_layers`
  - `batch_initialize_and_fit`
  - `show_best_fit_results`
- `src/cdsaxs/Fitting/Trapezoid_model.py`
  - current base trapezoid implementation
  - current vectorized batched objective path
- `src/cdsaxs/Fitting/SiGe_model.py`
  - new geometry with typed design layers and constraints
- `src/cdsaxs/Fitting/optimization_logger.py`
  - multi-layer optimization workflow helper

### Real-World Workflow Representations Now Present

The repo now represents realistic workflows in two places:

1. notebooks
   - `examples/fitting_examples/CDSAXS_DeRocher_V2.ipynb`
   - `examples/fitting_examples/CDSAXS_DeRocher_V3.ipynb`

2. API methods in `CDSAXS_base_model.py`
   - layer growth
   - automatic optimization-bound generation
   - repeated initialization workflows
   - MCMC
   - uncertainty-envelope plotting
   - best-fit replay from sweeps

This means the harness should now benchmark workflow semantics directly, not just raw parameter dictionaries.

## Current Incumbent Policy

### Tiny Trapezoid Workloads

For scalar or tiny-workload trapezoid fitting, the current incumbent remains:

- `AcceleratedTrapezoidModel`

Measured status on the rebased branch:

- scalar objective: about `1.5x` faster than base `TrapezoidModelArray`
- tiny scalar DE: still fastest among valid CPU paths tested
- tiny base `workers=8` DE: much slower than local scalar or vectorized tiny runs

### Realistic CPU Broadcast Workloads

For realistic vectorized CPU workflows, the meaningful incumbent is:

- `TrapezoidModelArray`

Reason:

- the current vectorized batched objective is implemented in the base trapezoid model
- the Cython wrapper does not provide a distinct winning batched path for realistic vectorized fitting
- the current accelerated model still routes batched objective work effectively through the base broadcast implementation

### Current GPU Baseline

The leading GPU path remains:

- `TrapezoidModelArrayDeanGPUFused`

### Invalid Incumbent To Avoid

Do not treat the current baseline `freeform_use_cupy=True` path in `Trapezoid_model.py` as a valid incumbent.

Observed behavior:

- candidate-budget timing can look unrealistically fast
- objective values can become `inf`
- fit parity is not reliable

## Validated Findings So Far

### Early Dean CPU Cleanup

Kept:

- cached `log(Intensity)`
- cached `Qx**2 + Qz**2`
- cached parameter parsing
- reduced Python-side temporary churn

Measured effect:

- modest but real speedup
- worth keeping as a cleaner baseline
- not sufficient by itself as the main breakthrough

### Bottleneck Attribution

For realistic vectorized trapezoid runs:

- `FreeFormTrapezoid` dominates objective time
- `GF_calc` is secondary
- `SymCoordAssign` is minor in the measured 4 to 12 parameter profiles

### Cython Status

The repo-local `cdsaxs_cython` path is useful as a scalar CPU reference.

Measured effect:

- scalar objective win over base CPU
- tiny scalar DE win over base CPU
- no clear winning batched/vectorized path
- batched Cython calls still fall back because kernels expect lower-dimensional inputs

Conclusion:

- keep as scalar CPU incumbent
- do not center future GPU work on Cython

### Rebased Realistic GPU Result

After rebase, the fused Dean GPU path still clearly wins for realistic workloads.

Representative measured results:

- `notebook_4param`
  - candidate-budget speedup vs realistic base CPU: about `5.7x` to `6.2x`
  - one-generation DE speedup vs realistic base CPU: about `5.2x`
- `notebook_multilayer_12param`
  - candidate-budget speedup vs realistic base CPU: about `15.9x` to `21.2x`
  - one-generation DE speedup vs realistic base CPU: about `13.9x`

It also continues to beat the scalar Cython incumbent on those realistic broadcast-shaped runs.

Conclusion:

- no restart is needed after rebase
- the GPU direction is still justified
- the comparison set changed more than the conclusion changed

## Real-World Workflows To Mirror In The Harness

These are the workflow shapes the harness should explicitly reflect.

### Trapezoid DE Base Workflow

Source:

- `CDSAXS_DeRocher_V3.ipynb`

Shape:

- one-layer trapezoid
- `CDSAXS_Optimize`
- `differential_evolution`
- compare:
  - scalar CPU
  - `workers`
  - `vectorized=True`
  - GPU-vectorized path

### Trapezoid Layer-Growth Workflow

Source:

- V3 notebook cells using:
  - `add_layer_at_percentage(10)`
  - `add_layer_at_percentage(90)`
  - sequential growth from an already grown model

API support:

- `CDSAXS_base_model.add_layer_at_percentage`
- `CDSAXS_base_model.add_multiple_layers`

### Trapezoid Alternative Optimizer Workflow

Source:

- V3 notebook cells using `dual_annealing`

Why it matters:

- not all real use follows DE
- alternative optimizer support should not silently break or bypass acceleration choices

### Batch Initialization Workflow

Source:

- notebook usage of `batch_initialize_and_fit`

API support:

- `CDSAXS_base_model.batch_initialize_and_fit`

Why it matters:

- this is a real workflow-level multiplier on objective cost
- GPU wins on the objective can compound here

### MCMC Workflow

Source:

- notebook usage of `CDSAXS_MCMC`
- uncertainty-envelope plotting after MCMC

API support:

- `CDSAXS_base_model.CDSAXS_MCMC`
- `plot_mcmc_uncertainty_envelope`
- related percentile envelope functions

Why it matters:

- real users do not stop at point estimates
- we need a short trial MCMC benchmark to understand whether objective acceleration survives this usage mode

### Sweep And Best-Fit Replay Workflow

API support:

- `show_best_fit_results`

Why it matters:

- future optimization propagation should reuse winning trapezoid objective machinery inside sweep-centered workflows

## Benchmark Matrix

The benchmark matrix should now be organized by both geometry and workflow family.

### Geometry Priority

1. trapezoid
2. SiGe
3. cylinder

### Workflow Families

1. scalar objective
2. batched objective
3. tiny DE
4. realistic candidate-budget
5. realistic one-generation DE
6. realistic short multi-generation DE
7. batch initialization smoke
8. MCMC smoke
9. MCMC trial benchmark

### Trapezoid Profiles

Keep the current notebook-parallel profiles:

- `notebook_4param`
- `notebook_layer10_8param`
- `notebook_layer90_8param`
- `notebook_multilayer_12param`

Add the next profile family:

- `notebook_v3_largeparam_de`

Intent:

- mirror the much larger V3 optimization cells
- keep it offline or opt-in
- use reduced optimizer budgets to avoid multi-minute runs

### MCMC Profiles

Add two trapezoid MCMC families:

1. `mcmc_smoke`
   - base 4-parameter model
   - short chain
   - enough to validate plumbing and rough throughput

2. `mcmc_trial`
   - one or more notebook-derived layered models
   - reduced but still realistic walker and step counts
   - enough to reveal scaling trends without turning into a full scientific production run

Recommended initial MCMC trial settings:

- `n_walkers`: just above the minimum valid threshold for the chosen dimension
- `n_steps`: low hundreds, not thousands
- `burn_in`: short but explicit
- `thin`: explicit and logged
- plots disabled during timing runs

### SiGe Profiles

Add an initial high-priority SiGe profile set:

1. `sige_flat_twidth`
   - typed trapezoids with `twidth`
   - no elliptical indentation yet

2. `sige_ellipse_single`
   - one `Layer_Type: 'Ellipse'`
   - includes `depth`

3. `sige_constraints`
   - typed layers with explicit `constraints`

These should start as objective and tiny-DE profiles first.

## Runtime Envelope

Keep the runtime classes explicit:

- smoke tests:
  - a few seconds
- realistic routine benchmarks:
  - tens of seconds
- offline scale studies:
  - under roughly one minute per selected profile

MCMC should be split the same way:

- smoke:
  - very short, test-oriented
- trial:
  - enough to reveal scaling
- scientific production:
  - explicitly out of scope for routine benchmarking

## Trapezoid Roadmap

### Stage T0: Preserve And Stabilize Existing Wins

Keep:

- `Trapezoid_model_dean.py`
- `Trapezoid_model_dean_gpu.py`
- current realistic trapezoid harness

Completed stabilization items:

- removed unconditional optimizer debug prints from `CDSAXS_base_model._run_scipy_optimizer`
- removed import-time Cython status prints from routine package loads
- restored `parameter_sweep_2d` notebook compatibility
- fixed stale wrapper parameter-name leakage across stateful workflows such as MCMC
- preserved optimized runtime classes through layer-growth APIs

Remaining cleanup items:

- keep benchmark output deterministic and machine-readable
- keep scalar and batched incumbents explicit in results

### Stage T1: Refactor The Harness Around Current Repo Semantics

Refactor the current trapezoid harness so it interacts more elegantly with the updated repo.

Completed:

- realistic profiles now build through the target model class and call:
  - `add_layer_at_percentage`
  - `add_multiple_layers`
- layered realistic profiles now keep the notebook-parallel `4`/`8`/`12` optimization surfaces instead of inheriting `sld_*` or per-column `Bk_i` parameters
- workflow-native runners now exist for:
  - candidate-budget objective timing
  - DE
  - dual annealing
  - batch initialization smoke
  - MCMC smoke and trial
  - sweep replay smoke
- workflow metadata now records:
  - `optimizer`
  - `vectorized`
  - `workers`
  - `freeform_use_cupy`
  - `height_percentage`
  - `height_percentages`
  - `sequential`
  - `n_walkers`
  - `n_steps`
  - `burn_in`
  - `thin`
- tests are now split into:
  - semantic workflow regressions in `test_dean_workflow_semantics.py`
  - benchmark smoke coverage in the Dean trapezoid benchmark suites

Remaining follow-on:

- strengthen high-value semantic regressions around the new workflow runners
- extend the workflow-native families beyond smoke level where they are worth timing routinely

### Stage T2: Keep The Current GPU Lead Valid

Continue treating `TrapezoidModelArrayDeanGPUFused` as the lead GPU candidate.

Next GPU implementation targets:

1. persistent GPU workspaces
2. even tighter device residency for reused buffers
3. further fusion of:
   - Debye-Waller application
   - magnitude-square
   - background addition
   - residual accumulation

Decision rule:

- keep complexity only if it improves realistic notebook-derived workloads, not just microbenchmarks

### Stage T3: Small-Workload Policy

Do not force GPU on tiny workloads.

Policy:

- scalar or very small batches:
  - prefer scalar CPU incumbent
- realistic broadcast-sized batches:
  - prefer vectorized GPU candidate if parity is maintained

This policy should eventually become an explicit threshold or fallback rule.

### Stage T4: Workflow Propagation

Once trapezoid objective wins are stable, propagate them to:

- `batch_initialize_and_fit`
- sweep-based workflows
- short MCMC workflows

### Stage T5: Trapezoid MCMC Benchmarking

This is now required work, not optional follow-on.

Goals:

- measure whether faster objective evaluation produces real end-to-end benefit in MCMC
- determine whether the accelerated path should be used directly inside MCMC likelihood calls

Required benchmark families:

1. `trapezoid_mcmc_smoke_4param`
2. `trapezoid_mcmc_trial_layer10`
3. `trapezoid_mcmc_trial_multilayer`

Required outputs:

- wall time
- effective samples generated
- mean acceptance fraction
- best-fit GF and BIC after applying best sampled parameters
- whether objective parity issues appear under repeated stochastic calls

## SiGe Roadmap

SiGe is now a high-priority geometry.

### Why It Matters

The rebased repo added a genuinely richer fitting model:

- geometry name: `sige`
- typed layers
- `twidth`
- `depth`
- elliptical indentation via `Layer_Type: 'Ellipse'`
- explicit constraints

This is not just a copy of trapezoid with renamed fields.

### Immediate SiGe Plan

1. make the harness aware of `SiGeModel`
2. add one helper module for reproducible SiGe benchmark models
3. add objective and tiny-DE tests before any GPU attempt
4. profile where SiGe time goes:
   - typed-layer expansion
   - constraint application
   - coordinate construction
   - form-factor evaluation
   - GF

### Initial SiGe Benchmark Families

1. objective scalar
2. objective batched if the implementation supports it cleanly
3. tiny DE
4. short one-generation DE
5. MCMC smoke after DE is stable

### Initial SiGe Incumbent Policy

Until measured otherwise:

- scalar CPU incumbent:
  - `SiGeModel`
- no GPU incumbent yet
- no assumption that trapezoid GPU machinery will transfer cleanly

### SiGe Optimization Direction

Priority order:

1. establish stable realistic SiGe profiles
2. remove obvious Python overhead in typed-layer expansion and constraints
3. determine whether SiGe can reuse trapezoid batched mapping machinery
4. only then evaluate a SiGe-specific GPU path

## Cylinder Roadmap

Cylinder remains the third geometry priority.

Current status:

- still relevant
- now lower priority than trapezoid and SiGe

Plan:

- keep cylinder on the roadmap
- do not start new cylinder optimization until:
  - trapezoid workflow harness refactor is done
  - SiGe baseline harness exists

## Repo Layout And Naming Alignment

### Keep

- optimized geometry variants beside existing geometry modules
- tests under `tests/test_fitting`
- workflow helpers under `src/cdsaxs/Fitting`

### Update In Spirit

The harness should align more closely with current repo naming and structure.

Recommended direction:

- geometry-first helper names
- workflow-family naming that mirrors current APIs
- user-facing examples that use:
  - `cdsaxs.Fitting`
  - `TrapezoidModel`
  - `SiGeModel`
  - `CylinderModel`

### Current Added Files That Still Fit Well

- `src/cdsaxs/Fitting/Trapezoid_model_dean.py`
- `src/cdsaxs/Fitting/Trapezoid_model_dean_gpu.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid.py`
- `src/cdsaxs/Fitting/dean_optimization_trapezoid_realistic.py`

### Next Files To Add

SiGe:

- `src/cdsaxs/Fitting/dean_optimization_sige.py`
- `tests/test_fitting/test_dean_optimization_sige.py`

Optional later refactor if naming pressure grows:

- move reusable benchmark-profile helpers into a shared `dean_optimization_workflows.py`
- keep geometry-specific entry modules thin and parallel

## Necessary Harness Changes For Better Repo Compatibility

These are required, not optional.

1. Stop treating raw parameter cloning as the main realistic builder mechanism.
   Prefer the current model APIs for workflow-derived model construction.

   Status:
   complete for the realistic trapezoid harness.

2. Split incumbent reporting by workload class.
   Every result should state whether the comparator is:
   - scalar CPU incumbent
   - realistic CPU vectorized incumbent
   - lead GPU candidate

3. Add workflow-native benchmark families.
   Specifically:
   - `batch_initialize_and_fit_smoke`
   - `mcmc_smoke`
   - `mcmc_trial`

   Status:
   complete for the trapezoid harness at smoke/trial level.

4. Make benchmark metadata match repo semantics.
   Include:
   - geometry
   - optimizer family
   - layer-growth path
   - constraint presence
   - typed-layer presence

   Status:
   complete for the current trapezoid workflow metadata.
   SiGe and future typed-layer workflows still need to fill in the typed-layer and constraint cases.

5. Keep the existing component-level Cython benchmark separate.
   It is useful support tooling, but it is not a workflow benchmark.

6. Record invalid paths explicitly.
   Example:
   - baseline CuPy path that returns `inf` objective values should be marked invalid, not merely slow or fast

## Decision Rules

### Keep A Candidate If

- it provides significant speedup on realistic benchmark families
- it preserves fit parity on deterministic checks
- it does not make workflow integration materially uglier

### Reject A Candidate If

- it wins only on microbenchmarks
- it silently falls back in real workflows
- it breaks MCMC, layer growth, or batch initialization semantics
- it requires a user-facing API fork that is too different from the current repo

## Known Caveats And Follow-Up Items

- the current repo still has some internal rough edges in workflow propagation, especially where older helpers still call `CDSAXS_DiffEvolution`
- routine timed coverage still centers DE more than batch initialization and MCMC
- the harness still does not measure time-to-convergence directly; current comparisons are fixed-budget rather than time-to-threshold
- `optimization_logger.py` should be treated as a workflow to integrate with, not yet as the main benchmark scaffold
- current mainline CuPy hooks in the base trapezoid model should not be trusted as the GPU baseline

## Continuation Checklist

If continuing from a fresh context, do this first:

1. confirm branch and environment
   - repo at `/homes/deand/dev/nist_cdsaxs`
   - branch `feature/dean_optimization`
   - env `cdsax-dev`

2. verify imports
   - `import cdsaxs`
   - `from cdsaxs.Fitting import TrapezoidModel, SiGeModel, AcceleratedTrapezoidModel`

3. verify current lead GPU class
   - `TrapezoidModelArrayDeanGPUFused`

4. re-run a short realistic trapezoid comparison
   - scalar incumbent
   - realistic vectorized CPU incumbent
   - Dean fused GPU path

5. implement the next harness expansion in this order
   - strengthen regression coverage for the workflow-native trapezoid runners that are worth keeping
   - add trapezoid MCMC benchmark coverage beyond smoke/trial if it still reveals useful scaling
   - SiGe helper module
   - SiGe objective and tiny-DE tests

6. record every attempt in `DEAN_OPTIMIZATION_LEDGER.MD`

## Summary Of What Comes Next

Immediate next work should be:

1. add stronger regression coverage for the new trapezoid workflow-native harness families
2. add trapezoid MCMC benchmark coverage beyond smoke level where it remains informative
3. start a dedicated SiGe benchmark and optimization track
4. continue GPU optimization only on realistic vectorized workflows and only against valid incumbents
