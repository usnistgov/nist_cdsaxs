# Agent reduction workflow

Use this guide with the JSON-configured harness in this repository. Read
[README.md](README.md) for the settings and command interface. Keep the generated
script and scientific decisions available to the user. Use `output.profile: "compact"`
for routine reductions. Deliver one full settings file, the generated script,
native figures, CSVs, and `summary.md`. Do not add separate audit JSONs, environment
dumps, manifests, duplicate settings snapshots, inspection scripts, or raw profile
exports to routine run folders. The harness consolidates bookkeeping in its hidden
state file. Request/use `full` output only when separate diagnostics are needed
for a specific investigation; place temporary analysis in a development scratch
directory and put its relevant conclusions in concise `decision_notes`.

The harness automatically records invocation/step runtimes and optional published
Codex token deltas in that state and the summary. Let it detect `CODEX_THREAD_ID`,
or pass `--codex-session-log PATH` when the runner does not inherit the thread.
Use loading/metadata timings to diagnose computational delays. Distinguish
unavailable counters from zero usage, and do not add cached input to input totals
or nested steps to parent totals. Agent work after a command returns is captured
only at a later boundary; same-session resume records the intervening activity,
which may include other work. Do not add separate usage audit files to run folders.

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
   away for gratings. Adjacent searches must keep the edge-to-edge gap below one
   integration-box height; set background offsets to null to enable that search.
   Use equal-magnitude top/bottom offsets by default. Enable
   `integration.allow_independent_backgrounds` only when the user requests
   independently placed boxes; do not work around symmetric selection failures
   by supplying unequal offsets. Near a detector/mask gap, the default
   `integration.allow_one_sided_backgrounds` permits one opposite-side background
   when the signal box at the slice columns has the configured unmasked clearance
   and the retained background passes coverage and contrast checks. Use the
   harness's per-frame choice in preview and production. Notify the user of every
   exception with row, corrected angle, retained/discarded offsets, and clearance;
   explain the assumption that the opposite-side background represents the signal
   background. Do not use this exception for bright backgrounds without gap evidence.
   Default background scoring uses requested qsx windows or measured peak windows.
   Inspect the integration checkpoint's `background_diagnostics` for candidate
   offsets, per-frame/per-side scores, mask evidence, and full-box comparison scores.
   Distinguish a failed row-background check from a rejected individual slice;
   unrelated features outside requested windows should not veto the entire row.
   Preview and production must use identical accepted settings.
6. Inspect bowtie/slice overlays. Narrow slices should capture peaks with little
   background. Read `summary.md` for exclusions and missing-angle decisions;
   detailed per-frame coverage is in the hidden checkpoint state (`coverage.json`
   is additionally available in the `full` profile).
   Reject a whole slice when its bin is unavailable in the frame nearest corrected
   phi=0, including supplied centers, even if overall angular coverage passes.
   A gap affecting only a background must first be evaluated with the one-sided
   rule above; retain the slice if its signal and final native coverage pass.
   Notify the user of skipped slices with centers, coverage, and reasons.
7. Additional qy rows require an explicit user request and circular contact holes.
   An automatic-mode setting or a circular-looking pattern alone does not authorize
   enabling them. Translate explicit instruction-based permission into settings and
   save the reason in decision_notes. Detect offsets unless supplied, inspect the row
   overlay, and select boxes/backgrounds/qsx peaks separately for every row. Retain
   row provenance when constructing the larger radial-coordinate dataset. Without
   a request for offset rows, reduce qy=0 and export only the combined result;
   omit qy0/qyall labels from these single-row CSV filenames.
8. Automatic batches may proceed without per-sample review only for decisions the
   user authorized in settings or instructions. Record scoped permissions in the
   effective JSON. Use configured sample-family bounds and inspect diagnostic plots;
   do not relax thresholds merely to force a sample through. Explain any justified
   threshold changes in decision_notes. Held samples go to the review queue while
   independent samples continue. Report fit consistency, not unsupported absolute
   accuracy or fabricated intensity error bars.
9. Export one CSV per φ group for a central-only run. Offer per-row/combined CSVs
   when offset qy rows were requested, optional positive/negative φ splitting, and
   the consolidated output-root option. Confirm nominal/corrected φ and zero handling
   from settings. Supply links to the generated Python script, full settings, summary,
   review figures, and exports. Describe assumptions and material limitations briefly.

User instructions take precedence over default settings. Persist instruction-based
changes in the effective settings so an independent rerun reproduces the decisions.
