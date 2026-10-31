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
| `slices.centers_qsx_Ainv`, `widths_Ainv` | Slice centers in Å⁻¹ and full widths; widths may be a scalar or a matching list |
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

Review mode holds at estimated choices. Inspect `figures/` and the proposed result
in `state.json`, then accept that existing result:

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
`subtract_background: false` explicitly disables subtraction.

Preview and final reduction use identical accepted geometry and background
offsets. Regions that overlap or leave the detector are rejected. Slice coverage
uses the library's strict mask rule: an angle point touching any masked bin is
omitted. `coverage.json` records lost frames and reasons. Automatic mode can skip
an entire inadequately covered slice only with `skip_gap_slices`; review users
remove that center or revise its width. Reduction uses measured, unmasked data.

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
The central qy0 row is included once. Boxes, backgrounds, and qsx slices are
selected independently for each row. `row_reductions.json` stores their settings.
Accepted per-row settings also appear under `additional_qy_rows.row_settings` in
the resolved JSON, keyed by row ID. These integration/slice overrides can be
supplied directly when rows require different known parameters.

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
root, new runs use `Results` beside the input data. Generated scripts resume in
their own directory; an explicit `--output-root` starts a run at that location
unless `--resume` is also supplied. `--sample NAME` selects one batch sample. A sample run can
also be resumed explicitly with `--resume RUN_DIRECTORY`.

Run files include original `settings.json`, latest `invocation_settings.json`,
`resolved_settings.json`, reusable `calibration.json`, stage state, figures,
coverage, row settings, CSVs, and `report.md`. Figures are saved as PNG only, at
`output.dpi`. Older `figure_formats` settings remain loadable; their effective
value is recorded as `["png"]` in the resolved settings.

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
