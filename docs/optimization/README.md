# Optimization Planning Docs

## Purpose

This directory holds the active optimization roadmap in geometry-specific pages.

Use these docs for current plans.
Use `DEAN_OPTIMIZATION_LEDGER.MD` only for dated experiment records and planning checkpoints.

## Document Map

- `trapezoid.md`
  - current kept trapezoid outcomes
  - reusable vectorized and GPU patterns
  - reusable versus non-reusable trapezoid work for other geometries
- `sige.md`
  - current SiGe status
  - detailed resumable plan for CPU vectorization
  - basic follow-on GPU plan
  - realistic harness and testing plan

## Naming Rules

For new optimization work, stop minting new `Dean` names.

Use these suffixes instead:

- `_vectorized`
- `_vectorized_GPU`
- `_GPU`

Interpretation:

- keep existing `Dean` files and classes as historical implementation names
- do not expand that naming pattern into new SiGe work
- prefer neutral module names for new harnesses and tests

## Current Status Snapshot

- trapezoid has a validated realistic CPU-vectorized baseline and a kept GPU-resident optimized path
- SiGe does not yet have a working vectorized objective path
- the next major implementation milestone is `SiGeModel.SimTrap_GF` CPU vectorization
- GPU SiGe work should follow that milestone, not precede it

## Resume Checklist

If you are resuming in a new context, start here:

1. Confirm repo and env:
   - repo: `/homes/deand/dev/nist_cdsaxs`
   - branch: `feature/dean_optimization`
   - env: `cdsaxs-dev`
2. Read `sige.md` first if the task is active SiGe work.
3. Read `trapezoid.md` next to identify reusable patterns and existing harnesses.
4. Check `DEAN_OPTIMIZATION_LEDGER.MD` for the last kept or rejected timing result before changing a validated path.
5. Keep new plan and status updates in these docs.
   Put measured outcomes in the ledger.

## Where New Material Belongs

- active plan updates: geometry page in this directory
- cross-geometry status and navigation: `DEAN_OPTIMIZATION.md`
- dated timings, measurements, and keep/reject decisions: `DEAN_OPTIMIZATION_LEDGER.MD`
