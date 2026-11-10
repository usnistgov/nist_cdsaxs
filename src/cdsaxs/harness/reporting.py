"""Human-readable reduction decisions for compact output packages."""
from collections import Counter


def number(value):
    return "not determined" if value is None else f"{value:.6g}"


def status(record):
    if not record:
        return "not run"
    value = record.get("status", "")
    if value == "failed_quality":
        return "failed quality"
    return "accepted" if value.startswith("accepted") else "proposed"


def lost_frames(item):
    names = {"outside_q_range": "outside q range",
             "masked_or_nonfinite_selection": "masked/nonfinite"}
    counts = Counter(point.get("reason") for point in item.get("coverage", [])
                     if not point["valid"])
    return "; ".join(f"{names.get(reason, reason or 'unavailable')}: {count}"
                     for reason, count in sorted(counts.items(), key=lambda pair: str(pair[0])))


def recording_summary(state):
    """Report invocation totals and the latest step breakdown without double counting."""
    records = state.get("recordings", [])
    if not records:
        return []

    def tokens(delta):
        fields = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
        return " | ".join(str(delta[key]) if delta is not None else "unavailable" for key in fields)

    lines = ["", "## Runtime and token usage", "",
             "Elapsed times are in seconds. Token deltas use published Codex session counters.",
             "Cached input is included in input tokens. Nested steps are included in their parents;",
             "do not add nested times or token deltas to their parent totals.", "",
             "| Invocation | Requested stage | Status | Elapsed (s) | Input Δ | Cached input Δ | Output Δ | Total Δ |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for index, record in enumerate(records, 1):
        elapsed = record.get("elapsed_seconds")
        lines.append(f"| {index} | {record.get('requested_stage') or 'unspecified'} | {record['status']} | "
                     f"{number(elapsed)} | {tokens(record.get('token_delta'))} |")
    for record in records:
        if record.get("execution_mode") == "suspend":
            lines += ["", f"Background job `{record.get('agent_job_id')}`: execution mode suspend; "
                      f"duration {number(record.get('background_elapsed_seconds'))} s; "
                      f"notification {record.get('notification_status', 'not_attempted')}."]
    latest = records[-1]
    if latest.get("token_note"):
        lines += ["", f"Token source: {latest['token_note']}."]
    between = latest.get("between_invocations_token_delta")
    if between is not None:
        lines += ["", "Session activity between the previous invocation and this invocation:", "",
                  "| Input Δ | Cached input Δ | Output Δ | Total Δ |", "|---:|---:|---:|---:|",
                  f"| {tokens(between)} |"]
    lines += ["", "Counters may stay unchanged during Python computation. They update when Codex publishes usage;",
              "agent work after the command returns is excluded. Between-invocation deltas include any other",
              "work in the same Codex session and are unavailable when the session changes."]
    if latest["steps"]:
        lines += ["", "Latest invocation steps:", "",
                  "| Step | Stage | Parent step | Status | Elapsed (s) | Input Δ | Cached input Δ | Output Δ | Total Δ |",
                  "|---:|---|---:|---|---:|---:|---:|---:|---:|"]
        for index, step in enumerate(latest["steps"], 1):
            parent = step["parent_index"]
            outcome = step["status"] + (" (cached)" if step["cached"] else "")
            lines.append(f"| {index} | {step['stage']} | {parent + 1 if parent is not None else '—'} | "
                         f"{outcome} | {number(step['elapsed_seconds'])} | {tokens(step['token_delta'])} |")
    return lines


def compact_summary(directory, settings, resolved, state):
    """Summarize accepted parameters, proposed choices, and actual exclusions."""
    stages = state.get("stages", {})
    lines = [f"# Reduction: {settings['sample_name']}", "",
             f"Status: **{state['status']}**", "",
             "Coordinates are in Å⁻¹, angles in degrees, box dimensions in pixels.",
             "Intensity is normalized in arbitrary units; full intensity uncertainties are unavailable.", "",
             "## Geometry and angular corrections", "",
             "| Parameter | Value | Decision |", "|---|---|---|"]
    calibration = stages.get("calibration", {})
    values = calibration.get("result", resolved["calibration"])
    center = values.get("center_px")
    lines += [f"| SDD (cm) | {number(values.get('sdd_cm'))} | {status(calibration)} |",
              f"| Beam center [row, column] (px) | {center} | {status(calibration)} |"]
    for stage, symbol in (("phi", "φ"), ("chi", "χ"), ("omega", "ω")):
        record = stages.get(stage, {})
        result = record.get("result", {})
        offset = result.get(f"sample_{stage}_offset_deg",
                            resolved["offsets"][f"sample_{stage}_offset_deg"])
        uncertainty = result.get("fit_uncertainty_deg")
        detail = f" ± {number(uncertainty)} (conditional fit uncertainty)" if uncertainty is not None else ""
        if result.get("scatter_deg") is not None:
            detail += f"; frame scatter {number(result['scatter_deg'])}"
        lines.append(f"| {symbol} offset (°) | {number(offset)}{detail} | {status(record)} |")
    if settings["phi"]["pitch_nm"] is not None:
        lines += ["", f"Configured pitch for the φ diagnostic: {number(settings['phi']['pitch_nm'])} nm."]
    lines += ["", "Fit uncertainties and scatter exclude calibration systematics."]

    problems = []
    for name, record in stages.items():
        problems.extend(f"{name}: {reason}" for reason in record.get("result", {}).get("failures", []))
        if not record.get("status", "").startswith("accepted") and not record.get("result", {}).get("failures"):
            problems.append(f"{name}: inspect the proposed values and figures before accepting.")
    if state["status"] == "needs_review" and state.get("events"):
        problems.append(state["events"][-1])
    if problems:
        lines += ["", "## Review required", ""]
        lines += [f"- {reason}" for reason in dict.fromkeys(problems)]

    geometries = [(name.split(":", 1)[1], record) for name, record in stages.items()
                  if name.startswith("integration:")]
    if geometries:
        lines += ["", "## Integration and backgrounds", "",
                  "| Row | Height × width (px) | Background offsets (px) | Edge gaps (px) | Decision |",
                  "|---|---|---|---|---|"]
        for row, record in geometries:
            geometry = record["result"]
            offsets = geometry.get("background_offsets_px")
            gaps = [abs(offset) - geometry["height_px"] for offset in offsets] if offsets else []
            lines.append(f"| {row} | {geometry.get('height_px')} × {geometry.get('width_px')} | "
                         f"{offsets if offsets is not None else 'disabled'} | {gaps} | {status(record)} |")
        lines.append("")
        for row, record in geometries:
            metrics = record["result"].get("metrics", {})
            if metrics:
                lines.append(f"- {row}: 95th-percentile alignment residual "
                             f"{number(metrics.get('alignment_residual_px'))} px; "
                             f"contained peak fraction {number(metrics.get('peak_inside_fraction'))}; "
                             f"background/signal peak contrast {number(metrics.get('background_peak_fraction'))}.")
            diagnostics = record["result"].get("background_diagnostics", {})
            selected = diagnostics.get("selected_frames", [])
            if selected:
                worst = max(selected, key=lambda item: item["contrast_fraction"])
                lines.append(f"- {row}: background scoring scope `{diagnostics['requested_scope']}`; "
                             f"worst checked frame `{worst['frame']}` at corrected φ "
                             f"{number(worst['phi_deg'])}°, contrast {number(worst['contrast_fraction'])} "
                             f"(limit {number(settings['quality']['max_background_peak_fraction'])}). "
                             "Candidate offsets, sides, mask evidence, and full-box comparison scores "
                             "are stored in the integration checkpoint's `background_diagnostics`.")
            exceptions = [item for item in diagnostics.get("production_frames", selected)
                          if item.get("one_sided_reason")]
            for item in exceptions:
                if item["one_sided_reason"] == "single_background_requested":
                    lines.append(f"- **Single background requested — {row}**, frame `{item['frame']}`, "
                                 f"corrected φ {number(item['phi_deg'])}°: use offsets "
                                 f"{item['offsets_px']} px; contrast {number(item['contrast_fraction'])}. "
                                 "The selected side is assumed to represent the signal background; "
                                 "signal masks and slice-coverage checks still apply.")
                    continue
                lines.append(f"- **One-sided background — {row}**, frame `{item['frame']}`, "
                             f"corrected φ {number(item['phi_deg'])}°: use offsets {item['offsets_px']} px; "
                             f"discard {item['discarded_offsets_px']} px on the detector/mask-gap side. "
                             f"Signal box and slice columns are clear by at least {item['signal_clearance_px']} px; "
                             f"retained background contrast {number(item['contrast_fraction'])}. "
                             "This assumes the opposite-side background represents the signal background.")

    rejected, partial, kept = [], [], 0
    for name, record in stages.items():
        if not name.startswith("slices:"):
            continue
        row = name.split(":", 1)[1]
        proposal = record["result"]
        for item in proposal.get("slices", []):
            coverage = item.get("coverage", [])
            valid = sum(point["valid"] for point in coverage)
            description = f"{valid}/{len(coverage)} ({item['angle_coverage']:.1%})"
            location = f"| {row} | {number(item['center_qsx_Ainv'])} | {number(item['width_Ainv'])} | {description} |"
            if item["skip"]:
                reasons = []
                if "invalid_near_zero_frame" in item.get("skip_reasons", []):
                    reasons.append("masked/unavailable in the frame nearest corrected φ = 0° "
                                   f"(φ = {number(proposal.get('reference_phi_deg'))}°)")
                if "insufficient_angle_coverage" in item.get("skip_reasons", []) or not reasons:
                    reasons.append("coverage below the configured minimum")
                rejected.append(location + " " + "; ".join(reasons) + " |")
            else:
                kept += 1
                if valid < len(coverage):
                    partial.append(location + " " + lost_frames(item) + " |")
    lines += ["", "## Slice decisions", "", f"Retained proposals: {kept}. Rejected proposals: {len(rejected)}."]
    header = ["| Row | qsx center (Å⁻¹) | Full width (Å⁻¹) | Valid frames | Reason |", "|---|---:|---:|---|---|"]
    if rejected:
        lines += ["", "Rejected proposals are excluded from completed CSV exports.", "", *header, *rejected]
    if partial:
        lines += ["", "Retained slices with omitted angle points:", "", *header, *partial]

    for stage, label in (("phi", "φ"), ("omega", "ω")):
        result = stages.get(stage, {}).get("result", {})
        if stage == "phi" and not result:
            result = state.get("phi_measurements", {})
        failed = result.get("failed_frames", [])
        if failed:
            lines += ["", f"Omitted {label} diagnostic frames:", ""]
            lines += [f"- `{item['frame']}`: {item['reason']}" for item in failed]
    omitted_rows = stages.get("qy_rows", {}).get("result", {}).get("omitted_detector_rows_px", [])
    if omitted_rows:
        lines += ["", f"Additional qy rows omitted by the configured limit (detector row pixels): {omitted_rows}."]
    if state["status"] == "complete" and any(state.get("export_omissions", {}).values()):
        lines += ["", "Native CSV export also omits nonpositive/nonfinite measurements after background subtraction:", ""]
        lines += [f"- `{name}`: {count} measurements." for name, count in state["export_omissions"].items() if count]
    if settings["decision_notes"]:
        lines += ["", "## Supplied decision notes", ""]
        lines += [f"- {note}" for note in settings["decision_notes"]]

    lines += recording_summary(state)
    settings_label = "Reduction constants" if state.get("standalone_published") else "Full settings"
    lines += ["", "## Outputs", "", f"- [{settings_label}](settings.json)",
              "- [Rerun script](reduce_sample.py)"]
    if state.get("standalone_published"):
        if settings["output"]["export_harness_settings"]:
            lines.append("- [Harness settings for resume](.harness/settings.json)")
        lines += ["", "The rerun script uses fixed native reduction steps without harness quality checks. "
                  "Native masks and CSV omission of nonpositive intensities still apply."]
    if (directory / "figures").is_dir():
        lines.append("- [Diagnostic figures](figures/)")
    csv_files = sorted((directory / "csv").glob("*.csv"))
    lines += [f"- [{path.name}](csv/{path.name})" for path in csv_files]
    return "\n".join(lines) + "\n"
