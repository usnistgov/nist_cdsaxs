# Agent reduction workflow

Use this guide with the JSON-configured harness in this repository. Read
[README.md](README.md) for the settings and command interface. Keep the generated
script and scientific decisions available to the user.

1. Discover the user's input data and any supplied constants/configuration. Generate
   a script/settings file if needed. Use the installed package and explicit paths.
   Inventory filenames and metadata, including the current loader's angle convention.
2. Preserve supplied constants. If the reference is unclear, ask the user to identify
   it or supply a calibration file. Names such as srm and AgBh identify candidates,
   not certified spacing. Inspect reference peak assignments, beam center, and SDD.
3. Follow the ordered stages: raw near-zero image; metadata/intensity corrections;
   apparent-SDD φ diagnostic; **fresh reload and reapplication of corrections with
   accepted φ**; χ at corrected normal incidence; ω at several high-|φ| positions;
   then angular correction. Do not replace calibrated SDD with the apparent-SDD fit.
4. In review mode, show concrete plots and proposed JSON changes before accepting
   estimated choices. Inspect detector orientation, peak identification, fit residuals,
   minimum location, and angle consistency. Use `--accept` only after the user has
   accepted that existing result, or update their supplied settings with accepted values.
5. Inspect boxes on representative angles. Peaks should be centered and contained.
   Backgrounds should be peak-free: usually adjacent for contact holes and farther
   away for gratings. Preview and production must use identical accepted settings.
6. Inspect bowtie/slice overlays. Narrow slices should capture peaks with little
   background. Read coverage.json to identify missing angle points and detector gaps.
   Notify the user of skipped slices with centers, coverage, and reasons.
7. Additional qy rows require an explicit user request and circular contact holes.
   An automatic-mode setting or a circular-looking pattern alone does not authorize
   enabling them. Translate explicit instruction-based permission into settings and
   save the reason in decision_notes. Detect offsets unless supplied, inspect the row
   overlay, and select boxes/backgrounds/qsx peaks separately for every row. Retain
   row provenance when constructing the larger radial-coordinate dataset.
8. Automatic batches may proceed without per-sample review only for decisions the
   user authorized in settings or instructions. Record scoped permissions in the
   effective JSON. Use configured sample-family bounds and inspect diagnostic plots;
   do not relax thresholds merely to force a sample through. Explain any justified
   threshold changes in decision_notes. Held samples go to the review queue while
   independent samples continue. Report fit consistency, not unsupported absolute
   accuracy or fabricated intensity error bars.
9. Offer per-row/combined CSVs, optional positive/negative φ splitting, and the
   consolidated output-root option. Confirm nominal/corrected φ and zero handling
   from settings. Supply links to the generated Python script, settings, report,
   review figures, and exports. Describe assumptions and material limitations briefly.

User instructions take precedence over default settings. Persist instruction-based
changes in the effective settings so an independent rerun reproduces the decisions.
