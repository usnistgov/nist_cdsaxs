# CD-SAXS reduction harness

The harness generates a runnable Python script and keeps scientific settings in
JSON. An agent follows [AGENT_WORKFLOW.md](AGENT_WORKFLOW.md); the same reduction
can run independently of an agent. Numerical reduction uses the existing package.

## Start a run

Install the repository in your Python environment:

```bash
pip install -e .[dev]
python -m cdsaxs.harness --init my_run
```

Edit `my_run/settings.json`, then run:

```bash
python my_run/reduce_sample.py --stage discover
python my_run/reduce_sample.py --stage phi
```

`cdsaxs-reduce` is an equivalent installed command. A supplied configuration can
prepopulate a new run: `python -m cdsaxs.harness --config experiment.json --init my_run`.
The complete template is [settings.example.json](settings.example.json). Settings
are merged with defaults; unknown keys and nonfinite JSON numbers are rejected.

A small reduction using the repository's TIFF fixtures and known constants is
provided in [validation.example.json](validation.example.json):

```bash
python -m cdsaxs.harness --config harness/validation.example.json --output-root /tmp/cdsaxs-validation
```

Relative data, calibration, profile, and output paths are resolved from the JSON
file's directory. Generated run settings contain absolute paths. The installed
package is required; scripts do not modify `sys.path` or assume a working directory.

## Optional start-stop mode

Tell the agent **“Use start-stop mode for this reduction.”** to request that it
launch the whole reduction in the background, acknowledge launch once, and end
its turn until Python completes. The default remains `--agent-wait foreground`.
Lifecycle options are command options, separate from scientific settings and
checkpoint fingerprints; generated scripts keep the foreground default.

```bash
python -m cdsaxs.harness --config experiment.json --agent-wait suspend
# If CODEX_THREAD_ID is not inherited, add --agent-thread-id UUID.
python -m cdsaxs.harness --agent-job-status /absolute/path/to/job.json
python -m cdsaxs.harness --agent-job-cancel /absolute/path/to/job.json
python -m cdsaxs.harness --agent-job-notify /absolute/path/to/job.json
```

Suspend mode requires a POSIX host and a Codex CLI supporting `queue --resume`
(verified with 0.167.0). Initialization and discovery use foreground mode. Launch
validates settings, expands samples, and reserves output directories before
starting a detached supervisor in the same Python environment. It returns the
job ID and record path promptly. Records, effective settings/argument snapshots,
and worker logs live under the launch directory's `.harness/jobs/<job-id>/`.
The supervisor holds run-directory locks while loading, metadata updates,
computation, review checkpoints, and exports run in the existing worker. CLI
foreground runs honor the same locks. Separate new runs can run concurrently;
resuming a directory already in use is rejected.

Keep the host and originating Codex server running, and establish required
output/server permissions before detaching. The notifier inherits the launching
process's Codex configuration and environment; it must already be authorized to
reach the originating server. CLI capability checks do not verify server access.
The harness does not elevate permissions, change scientific authority, or accept
review proposals automatically. See the [agent workflow](AGENT_WORKFLOW.md).

After success, a review stop, or an exception, the supervisor records the exit
code, sample outcomes, run directories, and summary paths, then invokes
`codex queue --thread UUID --message TEXT --resume` once. Invocation recordings
include the job ID and execution mode. Sample summaries include background
duration and notification outcome. Published token counters retain their existing
unavailable/reset semantics; this mode makes no guaranteed billing-savings claim.

Execution and notification status are separate. A zero queue exit code with a
warning is recorded as `pending`, because Codex can queue successfully while
failing to resume. Results survive notifier failure. Explicit `--agent-job-notify`
retries notification without rerunning Python, and skips an already delivered
notification. A pending delivery may already be queued, so there is no automatic
retry. Cancellation terminates the worker and suppresses continuation; it cannot
recall an already delivered message. A host/supervisor crash may require manual
inspection of the record and log; there is no automatic restart of computation.

The authorized live fixture test completed and its queued message reached the
originating conversation, but Codex reported an `already has an active writer`
resumption warning. The job correctly remained `pending`. This can occur when
completion precedes the launching turn becoming idle. Later message delivery
does not establish that the CLI's automatic resume succeeded; do not requeue
that message automatically. Clean automatic resumption of an idle conversation
still needs deployment verification.

The paired benchmark on Codex 0.167.0 also found that its server rejects
`queue --resume` for unloaded spawned subagents with `direct app-server input
is not allowed for unloaded spawned sub-agents`. These jobs retain their results
and record notification as `pending`; a parent-mediated continuation measures
assisted operation rather than automatic start-stop. A managed tool sandbox can
also terminate detached children before the supervisor starts. Establish a launch
environment that permits surviving detached processes before relying on this mode.

Automated tests use a fake notifier. Before relying on automatic continuation in
a deployment, perform an explicitly authorized live test that resumes the same
conversation, including completion before the launching agent becomes idle.
Codex's documented [app-server lifecycle](https://developers.openai.com/codex/app-server/)
describes thread resumption and turn completion; the installed CLI's
`codex queue --help` establishes the queue warning behavior.

## Settings and known constants

Use `null` for an unknown value requiring estimation. `0.0` is a known offset.
Explicit constants stay fixed unless `refit_known_constants` is true. Explicit
input metadata may also supply calibration values. A calibration `source_file`
can contain `sdd_cm` and `center_px` directly, or inside a `calibration` object.
Nominal SDD read from detector files is an initial estimate, not an accepted
absolute calibration.

`input` accepts explicit `filenames`, native filename `filter_substrings`, filename
`metadata_pattern`/`metadata_scales`, a native `metadata_csv`, common `metadata`, and
`user_params`. Explicit filenames override discovery. Frame IDs combine filename
and frame index, preserving repeated angles in HDF5 scans. Metadata CSV uses the
repository's CSV loader and should describe the desired sample's files.

| Setting | Meaning |
| --- | --- |
| `calibration.center_px` | Zero-based `[row, column]` beam center in pixels, in the repository's beam-coordinate convention |
| `calibration.sdd_cm` | Reference-calibrated sample-to-detector distance in cm |
| `calibration.reference_sample` | Filename substring identifying the reference; `reference_filenames` permits an explicit list |
| `calibration.reference_pitch_nm` | Known reference spacing in nm; supply its certificate value |
| `calibration.beam_center_guess_px` | Initial `[row, column]` guess for reference peak finding; default is detector midpoint |
| `calibration.sdd_guess_cm` | Approximate SDD used only to index reference peaks when detector metadata lacks a positive nominal value |
| `offsets.sample_*_offset_deg` | Additive φ, χ, and ω corrections in degrees; recorded angles are preserved |
| `corrections.normalize_by` | Metadata/user-parameter divisors, normally `bpm` and `exposure_time_s`; values must be positive scalars |
| `corrections.substrate_*` | Thickness in µm and attenuation coefficient in µm⁻¹; required when absorption is enabled |
| `phi.pitch_nm`, `phi.peak_orders` | Sample pitch (an approximate value may be used for this diagnostic) and fixed positive order magnitudes sorted by detector column; null orders request inference checked for consistency across the sweep |
| `*.search` | Native peak-search arguments, including pixel widths, exclusions, thresholds, and refinement; null `threshold_abs` derives a threshold above the measured profile baseline |
| `integration.height_px`, `width_px` | Box dimensions; `null` requests measured-peak-based proposals |
| `integration.background_offsets_px` | Background displacement along detector rows; these use the native subtraction convention |
| `integration.allow_independent_backgrounds` | Default `false`: require a symmetric `[-d, +d]` pair. Set `true` to allow independently placed backgrounds |
| `integration.allow_one_sided_backgrounds` | Default `true`: if a detector/mask gap affects one background, use the opposite side for that frame when the signal is clear and the remaining background passes the contrast limit |
| `integration.background_score_scope` | Default `slice_windows`: score supplied qsx windows, or measured peak windows when slices are being detected. `full_box` restores the earlier whole-profile score |
| `quality.min_signal_gap_clearance_px` | Default `1` px: required unmasked margin around the signal box at the slice columns for a one-sided gap exception |
| `slices.centers_qsx_Ainv`, `widths_Ainv` | Slice centers and full widths in Å⁻¹; null widths choose one common automatic width across peaks/rows. A scalar or matching list supplies explicit widths |
| `integration.mode`, `slices.mode` | Independently choose `sum` or `mean` for detector-box integration and qsx slicing |

Reference candidates named `srm` or `AgBh` are listed by discovery. An unambiguous
candidate can be used, but reference spacing is never guessed from its name. The
beam-center algorithm requires symmetric peaks; the SDD algorithm accepts explicit
order assignments. Ring patterns may require carefully chosen one-axis searches.
If reference files lack angles, normal incidence is assumed and recorded. A
separate `reference_metadata_pattern` can read reference angles from filenames.
The default search uses Gaussian peak refinement. Absolute-threshold inference
avoids selecting flat log-intensity backgrounds; supplied thresholds stay fixed.

The SMI loader handles its historical φ sign change; do not apply another sign
flip in the harness. Historical angle keys and constants need checking against
the current loader's coordinate conventions.

## Stages and review

Stages are `discover`, `calibration`, `phi`, `chi`, `omega`, `preview`, `reduce`,
`export`, and `all`. A stage executes its prerequisites, reusing unchanged
accepted numerical estimates. `reduce`, `export`, and `all` finish with CSV export.

1. Calibrate geometry from a reference or supplied constants.
2. Load and plot the raw image nearest nominal φ = 0; apply intensity corrections.
3. Estimate φ from a local apparent-SDD quadratic fit and save residuals and the
   proposed offset. Absolute reference SDD stays fixed.
4. **Reload raw data, apply the accepted φ offset, and reapply BPM/exposure,
   absorption, and footprint corrections.**
5. Estimate χ nearest corrected normal incidence; set its offset metadata.
6. Estimate ω at several angles away from normal incidence; set its offset.
7. Apply rotation corrections once, preview boxes and backgrounds across
   representative frames, integrate the bowtie, and inspect slice overlays.
8. Export accepted reductions.

Review mode holds at estimated choices. Inspect `figures/` and the proposed values
and checks in `summary.md`, then accept that existing result. The `full` output
profile also exposes the checkpoint details in `state.json`:

```bash
python my_run/reduce_sample.py --accept phi --stage chi
python my_run/reduce_sample.py --accept chi --stage omega
python my_run/reduce_sample.py --accept omega --stage preview
python my_run/reduce_sample.py --accept integration --stage preview
python my_run/reduce_sample.py --accept slices --stage reduce
```

Calibration and detected qy rows have their own `--accept calibration` and
`--accept qy_rows` checkpoints. For multiple rows, repeat the command as each
new row's proposal becomes available. Acceptance applies only to an already
computed, unchanged proposal. Supplying accepted dimensions, centers, and widths
in JSON also settles those selection decisions. Failed quality checks cannot be
overridden by `--accept`; tune settings or tolerances with a recorded reason.

Each reconstruction starts from raw images. Stage fingerprints include relevant
settings, file identity (path, size, modification time), and dependencies. Changed
inputs or settings invalidate dependent estimates and approvals. File identities
are not full-content hashes. Package versions and a source-code digest also enter
the stage fingerprints. The runner does not serialize mutable detector objects.
When a changed or failed resumed run requires review, previous CSVs are preserved
under `archives/` rather than left as current results. New exports are assembled
in a pending directory and installed together after successful reduction.

## Runtime and token recordings

Every reduction invocation records UTC start/end times, elapsed seconds, and a
step breakdown in the checkpoint state (`.harness/state.json` for compact output,
`state.json` for full output). `summary.md`/`report.md` includes invocation totals
and the latest step table. Resumes append to the history, including review stops,
failed steps, and cached checkpoints. Console messages identify the active step
and print its timing and token delta on completion. Loading separates detector
reads, frame metadata, calibration metadata, and angle metadata; detector reads
also include any metadata/q calculations performed inside the native loader.
Corrections, angular fits, rotation, row integration, and export are also timed.
Parent timings include their nested steps; do not sum both. Recording does not
change scientific settings or checkpoint acceptance.

Token snapshots are optional and require no API calls. When `CODEX_THREAD_ID` is
available, the harness finds that thread's JSONL log under `CODEX_HOME` (default
`~/.codex`). It never chooses an unrelated latest session. An external runner can
pass a specific log:

```bash
python my_run/reduce_sample.py --stage phi --codex-session-log /path/to/rollout.jsonl
```

Alternatively set `CDSAXS_CODEX_SESSION_LOG` to its path. The explicit CLI path
takes precedence. Each step and invocation saves the latest published cumulative
input, cached-input, output, and total counters at both boundaries, their event
timestamps, and their differences. Only token-count events are retained, not
prompts or tool output. After the initial read, only appended log lines are read.
Missing, unreadable, or reset counters produce unavailable deltas (`null` in
state), rather than an assumed zero. Partially appended events are retried at
the next boundary; until then the latest complete valid counters remain in use.
Unchanged valid counters produce zero. Cached-input tokens are already included
in input tokens.

These deltas describe published activity in the shared Codex session, not billing
or exact attribution to numerical computation. Python computation itself uses no
model tokens, and counts may remain unchanged throughout a long command. Usage
published after the command returns cannot appear in that invocation's end
snapshot. On resume in the same session, the record also captures activity since
the previous invocation ended; that interval includes any other work in the
session. It is unavailable when the log changes. Discovery prints recordings
with its inventory; the full profile additionally saves them in `inventory.json`.

## Automatic batches

Set `mode` to `automatic` and list the decisions the runner may make:

```json
{
  "mode": "automatic",
  "sample_kind": "circular_contact_holes",
  "allowed_decisions": [
    "accept_calibration",
    "accept_angle_corrections",
    "choose_integration_box",
    "choose_backgrounds",
    "choose_slice_locations_and_widths",
    "skip_gap_slices",
    "choose_qy_offsets"
  ],
  "samples": [
    {"name": "sample_A", "input": {"filter_substrings": ["sample_A"]}},
    {"name": "sample_B", "input": {"filter_substrings": ["sample_B"]}}
  ]
}
```

This is a partial configuration to merge with your scientific settings. A
`sample_profile` JSON file can provide shared geometry, peak-search settings, and
quality limits; it must not contain `samples` or another `sample_profile`. Individual
sample objects can override any setting. User-supplied settings override a profile.
Explicit `null` requests re-estimation rather than inheriting a profile constant.

The initial quality limits are configurable heuristics, not certified accuracy
requirements. Set them for your detector, sample family, and accepted representative
data. Checks cover φ residuals and minimum location, χ proximity to normal incidence,
ω consistency, corrected alignment, peak containment, background contamination,
and valid slice coverage. Calibration and corrections are checked separately for
each sample. A held sample enters the batch manifest with `needs_review`; other
samples continue. Exit status is 2 when any sample needs review, otherwise 0.

Automatic processing only accepts decisions listed in `allowed_decisions`.
Known constants are not refitted merely because automatic mode is enabled.

## Integration, gaps, and extra qy rows

`sample_kind` is `contact_holes`, `circular_contact_holes`, or `line_grating`.
Automatic backgrounds use `adjacent` for contact holes and `distant` for line
gratings, then evaluate measured background peak contrast. `adjacent` searches
each side with an edge-to-edge gap from zero to one pixel less than the integration
box height. For height 14 px, gaps are 0–13 px and displacements are 14–27 px.
`distant` searches displacements of 5–9 box heights. `background_strategy` can
explicitly select either strategy. Set `background_offsets_px: null` to use the
automatic search; an explicit list fixes the offsets and overrides the strategy.
The default search selects a common displacement for both sides, minimizing the
worse background contrast across the retained boxes and all checked frames. Ties
prefer the closest pair. Set `allow_independent_backgrounds: true`
to select the best usable offset on each side separately. Explicit offsets must
also form a symmetric `[-d, +d]` pair unless that option is enabled. When resuming
an older run with unequal saved offsets, set `background_offsets_px: null` to
request a new symmetric pair, or explicitly enable independent backgrounds.
`subtract_background: false` explicitly disables subtraction.

Background contrast is `(maximum - median) / signal contrast`, using mean intensity
per detector pixel. By default the maxima are evaluated in the requested qsx slice
windows, mapped to signal columns separately at each corrected angle; the medians
use the full profiles. With unknown slices, measured peak widths and padding define
the columns; if neither windows nor peaks are available, scoring uses the full box.
An unrelated bright feature elsewhere in a wide box therefore cannot reject a whole
row. The configurable contrast limit remains unchanged. The hidden integration
checkpoint's `background_diagnostics` records every candidate offset, frame, side,
score, mask evidence, and the earlier full-box score for comparison. `summary.md`
identifies the scoring scope and worst checked frame. A failed background check is
a row-geometry rejection before slice selection; inspect that checkpoint separately
from individual slice coverage decisions.

Near a detector gap, retain unaffected signal and use one background on the
opposite side when `allow_one_sided_backgrounds` is true. The gap must overlap a
background at the usable slice columns (or that box must leave the detector),
while the signal box and those columns, expanded by `min_signal_gap_clearance_px`,
remain unmasked and finite. The retained background must pass coverage and contrast
checks. A bright background without gap evidence cannot trigger this exception.
The selected offsets are checked on every production frame; per-frame choices are
used identically in preview and integration and saved in the checkpoint. Console
notifications and the summary list every one-sided frame, its corrected angle,
retained/discarded side, and clearance. Set the option to false to require both
backgrounds throughout. One-sided subtraction assumes the opposite-side background
represents the background under the signal; spatial gradients can limit that assumption.

Preview and final reduction use identical accepted geometry and background
choices. Regions that overlap or leave the detector are rejected unless a gap-side
background is omitted under the rule above. Slice coverage
uses the library's strict mask rule: an angle point touching any masked bin is
omitted. An entire slice is rejected if its bin is unavailable in the frame nearest
corrected φ = 0°, even when its overall angular coverage passes the configured
minimum. This applies to supplied centers and detected peaks on every requested
qy row. Missing reference-frame data include masked/nonfinite selections and bins
outside its integrated q range; finite nonpositive intensity alone is not a veto.
The checkpoint state records the reference frame and corrected angle, its per-slice
validity/rejection reason, overall coverage, and `skip_reasons`. Rejected bins are
excluded from slice overlays and exports. Automatic mode can skip rejected slices
with `skip_gap_slices`; otherwise the reduction enters `needs_review`. Review users
remove that center or revise its width. Reduction uses measured, unmasked data.

By default, `slices.widths_Ainv: null` selects one equal qsx width for all peaks
and requested qy rows. Each row's frame nearest corrected φ=0 supplies measured
peak widths; multiply by 1.5, convert to Å⁻¹, apply `width_bounds_Ainv`
(default 0.0002–0.003 Å⁻¹), and use the largest estimate everywhere. This preserves
the widest automatic proposal. The selected q width stays fixed across φ frames;
the number of included detector pixels may vary. If the common width reaches a
neighboring peak center or violates a row's width bounds, the harness requests
review. Supply a scalar for a fixed equal width, or a list/row override to request
explicit widths. Equal accepted widths are exported as one scalar; unequal
explicit widths remain a list. The standalone script performs no width estimation.

Additional rows are **disabled by default** and require the user's explicit
request for circular contact holes:

```json
{
  "sample_kind": "circular_contact_holes",
  "additional_qy_rows": {
    "enabled": true,
    "offsets_qsy_Ainv": null,
    "offsets_qdy_px": null
  }
}
```

Both lists null means detect additional peak rows; review them or authorize
`choose_qy_offsets`. Supply one list to request known rows. Physical sample-qsy
values are in Å⁻¹ and tracked per frame. Legacy `offsets_qdy_px` are integer pixel
displacements with positive offsets upward, matching the example. Background
offsets instead follow native array-row subtraction. Do not mix these conventions.
The central qy0 row is included once. All rows use the same signal/background box
height and width and the same background-offset list across every frame. Unspecified
dimensions are chosen to contain measured peaks across all rows and production
frames. Background candidates must pass every row/frame using that row's slice
windows. A failure requests review; multirow reductions never drop a background
on individual frames. The single-row detector-gap fallback remains available.
Offset rows inherit the accepted qy0 qsx centers and widths by default. Supply
`additional_qy_rows.row_settings.<row_id>.slices.centers_qsx_Ainv` or
`widths_Ainv` to override either constant. Set `slices.inherit_qy0: false` for
an offset row to request independent automatic peak centers; automatic widths
remain shared across rows unless explicitly supplied. Normal harness coverage
checks and authorized slice omissions still apply to each row.
The checkpoint state stores their settings;
the `full` output profile additionally writes `row_reductions.json`.
Accepted per-row differences appear under `additional_qy_rows.row_settings` in
the resolved harness JSON, keyed by row ID. Redundant inherited settings are omitted.
Row integration overrides must match the shared integration settings; redundant
overrides are removed from resolved multirow settings.

Combined data include the library's radial in-plane coordinate
`qsr = sqrt(qsx**2 + qsy**2)`. Measurements and row identities are retained;
overlapping measurements are not automatically averaged. Circular symmetry is a
user-declared assumption, not a conclusion from image appearance.

## Exports and output locations

Run directly from supplied settings or select a consolidated root:

```bash
python -m cdsaxs.harness --config experiment.json --output-root /path/to/reductions
python -m cdsaxs.harness --config experiment.json --resume /path/to/batch_TIMESTAMP.json
```

The layout is `<root>/<experiment>/<sample>/<UTC-run-id>/`. Without an output
root, new runs use `Results` beside the input data. Staged harness scripts resume in
their own directory; an explicit `--output-root` starts a run at that location
unless `--resume` is also supplied. `--sample NAME` selects one batch sample. A sample run can
also be resumed explicitly with `--resume RUN_DIRECTORY`.

The default `output.profile: "compact"` produces this visible layout:

```text
settings.json       accepted reduction constants after completion
reduce_sample.py    standalone native reduction script after completion
summary.md          parameters, exclusions, and decisions requiring review
figures/            native diagnostic PNGs
csv/                native reduced-slice exports
```

Checkpoint state, input identities, software versions, initial/invocation settings,
and per-frame coverage are consolidated in `.harness/state.json`. Pending and
superseded CSVs are retained under `.harness/`. They do not clutter the visible
deliverables. Single-sample compact runs do not create batch manifests. Multi-sample
runs keep their resume manifest under `<root>/<experiment>/.harness/`.

While a run is paused or only a stage is complete, `settings.json` retains the
requests used to produce its proposals so `--accept` can resume unchanged results.
Once the reduction completes, the default standalone output contains accepted
calibration, angular offsets, integration/background geometry, fixed rows and
slice centers/widths, and effective export options. Initial requests and intermediate
results remain in the hidden state. The completed settings can reproduce CSVs in
a new directory. `summary.md` explains rejected slice centers and reasons, retained
slices with missing angles, excluded diagnostic frames, nonpositive CSV omissions,
and unresolved quality checks.

Choose `output.profile: "full"` or `--output-profile full` when separate diagnostic
JSON files are useful. This adds reusable `calibration.json`,
`state.json`, `phi_measurements.json` when fitted, `coverage.json`,
`row_reductions.json`, `report.md`, and batch manifests. `--output-profile compact`
overrides a saved full setting. Profile changes can resume existing checkpoint
state; existing files are preserved, so use a new run directory for a clean compact
package. Full invocation/resolved settings snapshots are included only when
`output.export_harness_settings` is enabled or staged `harness` script mode is
explicitly selected. Compact discovery prints the inventory to the console rather than
creating `inventory.json`.

Figures are saved as PNG only, at `output.dpi`. Older `figure_formats` settings
remain loadable; the effective value is recorded as `["png"]` in the saved settings.

### Standalone rerun script

The default is `output.script_mode: "standalone"`:

```bash
python -m cdsaxs.harness --config experiment.json
```

Successful completion replaces root `reduce_sample.py` and `settings.json` with a
native reduction script and resolved constants only, for either output profile.
Before completion, those files remain the harness entry point and its settings.
The constants include explicit input filenames, calibration, angular offsets,
corrections, common geometry, fixed rows and slices, and output choices. Searches,
quality thresholds, permissions, fitting configuration, and checkpoint state are
excluded. Full harness settings are not exported by default, including in the
`full` output profile. Internal checkpoint state remains available. Select
`--script-mode harness` or `output.script_mode: "harness"` explicitly to retain
the staged script and its full root settings.

Copy the two root files, edit paths and constants, then run:

```bash
python reduce_sample.py
```

The installed `nist_cdsaxs` package is required. Relative paths use the JSON
directory; `output.root` defaults to `.` and writes `csv/` and `figures/` there.
The script loads fresh data, applies accepted offsets and intensity corrections
once, corrects rotation, integrates fixed rows, extracts fixed slices, and exports.
It imports no harness code, fits nothing, and performs no harness quality checks
or automatic reselection. High backgrounds and peak-free slices do not stop the
script for review. Native masks, ordinary API/file errors, and CSV omission of
nonpositive intensities still apply. Linear figures allow inspection of negative
background-subtracted intensities. Review copied reductions manually.
For a single-row run using an accepted detector-gap exception, any frame-specific
background choice is saved as explicit constants, without rerunning gap decisions.
Update those frame IDs and offsets when changing input filenames.

Standalone settings store shared constants once. `slices` contains the qy0 centers,
widths in Å⁻¹, and mode. Offset rows inherit these settings; an optional row `slices`
object overrides only the specified keys. Editing the shared centers or width
therefore updates all rows that inherit them. For example:

```json
{
  "integration": {
    "height_px": 6,
    "width_px": 1400,
    "background_choices": [-9, 9],
    "subtract_background": true,
    "mode": "sum"
  },
  "slices": {
    "centers_qsx_Ainv": [-0.026, 0.026],
    "widths_Ainv": 0.003,
    "mode": "sum"
  },
  "rows": [
    {"row_id": "qy0", "target_qsy_Ainv": 0.0, "offset_qdy_px": null},
    {"row_id": "qy_001", "target_qsy_Ainv": null, "offset_qdy_px": 15}
  ]
}
```

`background_choices` is one selection shared across slices and frames. A scalar
such as `-9` requests one signed pixel offset; a list such as `[-9, 9]` averages
those background boxes. Thus a symmetric pair remains a two-element list.
Equal widths are exported as a scalar; different widths may be a list with one
value per center. Explicit background exceptions use
`{"default": [-9, 9], "by_frame_px": {"frame_id": -9}}` or
`by_slice_frame_px`, a list of sparse frame overrides, one dictionary per slice.
A null slice/frame override preserves a saved masked bin. A row may supply its
own `background_choices`. Existing row-local slice lists,
`background_offsets_px`, `background_offsets_by_frame_px`, and detailed
`background_choices_by_slice_frame_px` settings remain supported by newly
generated scripts.

To also export `.harness/settings.json` for resume, explicitly request
`--export-harness-settings` or set `output.export_harness_settings: true`:

```bash
python -m cdsaxs.harness --config experiment.json --export-harness-settings
```

Resume using that exported configuration:

```bash
python -m cdsaxs.harness --config RUN/.harness/settings.json --resume RUN
```

Portable settings are not harness configuration. Ordinary harness saves preserve
edits to the portable files; publishing changed accepted constants preserves edited
files under `archives/` before replacement. `--no-export-harness-settings` overrides
an inherited export request. Previously exported files are preserved but are not
updated when export is disabled. Without a settings export, resume with the
original harness configuration and `--resume RUN`.

Run development tests in the `cdsaxs_dev` conda environment:

```bash
conda run -n cdsaxs_dev python -m pytest
```

With only the central qy=0 row, the harness exports one combined CSV per φ group,
without a qy label: `<sample>_all.csv`, and optionally `<sample>_posPhi.csv` and
`<sample>_negPhi.csv`. It does not duplicate the central row in separate qy0 and
qyall files. This applies whenever either native CSV export setting is enabled.

Offset qy rows require an explicit user request. When they are included,
`export.per_row` controls the row-labeled CSVs and `combine_qy_rows` adds the
combined `<sample>_qyall_<group>.csv`. The resolved settings record the effective
export choices.

`split_phi` adds positive/negative outputs using native
metadata selection before slicing. `phi_basis` selects `nominal` or `corrected`;
`zero_phi` selects `positive`, `negative`, `separate`, or `exclude`, with a
configurable zero tolerance. Groups are disjoint. The all-angle export is retained.

CSV exports use the existing reduction code's analysis-compatible wide format
and exclude nonpositive intensities. Each export reports discarded points.
Intensities are arbitrary normalized units.

Waterfalls use `cdsaxs.plotting.plotting.plot_reduced_slices(..., offset_order=2)`
for every angle group, matching the example reduction's native waterfall. Slice
centers determine the order and successive curves are scaled by `10**(-2*i)`;
there is no cyclic offset or intensity rescaling in the CSVs. Slice overlays and
detector images also use the native plotting functions.

The harness delegates peak detection and center-of-mass refinement to `cdsaxs.tools`,
box bounds to `Data2D.get_box_dims_size`, and profile integration to the native
box methods. Native fractional-center conventions apply; clipped integration boxes
are rejected. SDD peak orders come from the native calibration diagnostic, avoiding
a second peak search. Coverage reports use provenance from `slice_reduced_dataset`,
so the exported and previewed selections share the same mask and boundary rules.
Automatic proposal limits, background scoring, fit-quality checks, and review
checkpoints remain harness responsibilities.

Conditional φ-fit uncertainty and inter-order/inter-angle scatter are reported
separately from metrological accuracy. Full intensity uncertainty propagation is
not implemented by the current numerical reduction. The supplied Q11-R1 figures
are review references; exact reproduction requires its original raw scans.


### Requesting one background box

Set `integration.background_count` to `1` to request one background box for a row. The default is `2`, retaining symmetric two-background selection and the existing detector-gap fallback. An explicit list such as `background_offsets_px: [-24]` fixes that offset relative to the signal box for every frame. Independent backgrounds need not be enabled for a single box.

With count `1`, `background_offsets_px: null` selects one constant offset from the configured search range, using every production frame. It first preserves the greatest minimum unmasked support for the requested signal columns, then minimizes the worst background contrast across frames. Ties use smallest absolute offset, then signed offset. If no common offset passes every frame, the harness requests review rather than switching offsets between phi frames. Resolved settings record the selected fixed offset.

Every production frame must pass background contrast and coverage checks. An offset with no measured background under the requested slice columns is rejected, even when unrelated detector columns are valid. Signal masks and slice-coverage limits still apply; no masked signal is filled.

`allow_one_sided_backgrounds` controls the existing gap fallback from a two-background request. A deliberate `background_count: 1` request does not require a gap or claim that the signal is clear of gaps. Reports identify it as a requested single background and record its fixed offset, frame and contrast. This assumes the selected side represents the background under the signal.

### Excluding nominal phi frames

`input.exclude_nominal_phi_deg` defaults to `[]`. Specify nominal sample angles such as `[-2, 0, 2]` to exclude matching measurement frames, using the same 0.05 deg tolerance as explicit phi-angle selection. Raw input files are unchanged, calibration reference frames are not excluded, and retained frame identifiers preserve their original scan indexes. Exclusions are applied before all analysis stages and recorded in input identities so changing this list invalidates affected cached fits.
