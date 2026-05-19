---
name: cdsaxs-analysis
description: "Use when working on Critical Dimension Small-Angle X-ray Scattering (CDSAXS) problems, including line-space gratings, contact holes, hole profiles, cylindrical geometry, trapezoid geometry, profile fitting, parameter identifiability, uncertainty analysis, and Python notebook or script support for CDSAXS interpretation and modeling."
---

# CDSAXS Analysis

Use this skill for CDSAXS interpretation, fitting strategy, and analysis-code assistance when the task involves profile metrology from scattering data. It applies to both line-space gratings and contact-hole style structures.

## Supported Geometry Families

- Line-space gratings and related periodic line profiles.
- Contact holes and hole-profile problems represented with cylindrical or radius-vs-depth parameterizations.
- Geometry-aware workflows where the model choice should match the physical structure rather than forcing one parameterization onto all data.

## Core Operating Rules

- State the assumed geometry early: trapezoid or line-grating style versus cylinder or contact-hole style.
- Check units before reasoning about dimensions, pitch, q-space ranges, or fitted parameters.
- Separate what is directly supported by the data from what is only model-inferred.
- Treat inverse problems as potentially non-unique unless the data and constraints clearly support identifiability.
- Prefer the simplest defensible model first, then add complexity only when residual structure or prior knowledge justifies it.

## Interpretation Workflow

When interpreting CDSAXS data, work in this order:

1. Confirm the sample class and geometry.
2. Confirm the measurement geometry and available reciprocal-space coordinates.
3. Identify which observables are robustly present in the data.
4. Distinguish sample effects from instrument, normalization, or background effects.
5. Explain what structural conclusions are likely, tentative, or unsupported.

For line-space gratings, focus on:

- Pitch and periodic ordering.
- Linewidth or critical dimension.
- Height and sidewall angle or taper.
- Line-edge or line-width roughness and Debye-Waller style damping.
- Form-factor versus structure-factor contributions.

For contact holes or cylindrical profiles, focus on:

- Diameter, radius, or top and bottom CD.
- Depth and residual layer thickness.
- Taper, necking, bowing, footing, or corner rounding when the model supports them.
- Whether the profile is reasonably approximated as axisymmetric.
- Side-wall roughness and Debye-Waller damping in the radial direction.
- Film roughness and Debye-Waller damping in the vertical direction.
- Whether the data can actually separate diameter, taper, depth, and roughness.

## Fitting Workflow

Use this sequence for fitting guidance:

1. Verify geometry, coordinate conventions, and units.
2. Inspect raw and processed data quality before fitting.
3. Choose a parameterization that matches the physical structure.
4. Set physically plausible bounds and initial values.
5. Fit the simplest model that can explain the dominant features.
6. Inspect residuals and parameter correlations.
7. Add complexity only if it improves interpretation, not just numerical fit quality.
8. Report uncertainty, sensitivity, and identifiability limits.

For line gratings:

- Start with pitch, width, height, and sidewall angle or equivalent trapezoid parameters.
- Add roughness, multilayer segmentation, or interfacial effects only after the base model is stable.

For contact holes:

- Start with a simple cylindrical or segmented-radius model.
- Add taper or depth segmentation only if residuals indicate missing profile variation.
- Be explicit that contact-hole fits often have strong degeneracy between diameter, taper, depth, and roughness.
- Avoid claiming unique hole-profile recovery from weak higher-order features alone.

## Code And Notebook Assistance

When helping with Python notebooks or scripts:

- Ask which geometry is intended before proposing parameter names or model structure.
- Prefer transparent, reproducible code over opaque helper layers.
- Preserve the project's geometry-aware modeling style.
- Use existing project concepts when available, such as trapezoid versus cylinder model branches.
- Make unit handling explicit in code and plots.

Useful coding tasks include:

- Loading CDSAXS data and metadata.
- Plotting intensity versus q or angle with clear labels.
- Comparing measured and simulated cuts.
- Building geometry-specific parameter dictionaries.
- Summarizing fitted parameters, residuals, and uncertainty checks.

## Reliability Boundaries

Do:

- Explain CDSAXS concepts and tradeoffs clearly.
- Critique fitting setups and parameter choices.
- Suggest validation checks and alternative parameterizations.
- Call out underconstrained interpretations.

Do not:

- Fabricate beamline constants, calibration values, or missing metadata.
- Claim unique structural solutions without evidence.
- Treat model-dependent contact-hole details as directly observed facts.
- Ignore geometry mismatch between the physical sample and the chosen model.

## Response Style

- State assumptions explicitly.
- Name each parameter in physical terms.
- Include unit checks when discussing dimensions or reciprocal-space values.
- For contact holes, say whether the result is diameter-based, radius-based, or segmented by depth.
- For fitting advice, mention uncertainty and parameter correlation by default.

## Project Alignment

This repository already distinguishes between trapezoid and cylinder geometries in its CDSAXS modeling flow. Match that pattern when discussing implementation or analysis design, and do not collapse contact-hole and line-grating workflows into a single generic recipe.