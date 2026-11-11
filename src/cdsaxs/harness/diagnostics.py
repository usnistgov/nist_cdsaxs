"""Fit checks, measured peak selection, and publication-ready diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import peak_widths

from .. import tools
from ..data.dataset import ReducedDataset
from ..reduction import slice_reduced_dataset
from copy import deepcopy
from .config import is_symmetric_background_pair


class NeedsReview(ValueError):
    """A missing scientific choice or failed check holds this sample."""


def check(condition, message):
    if not condition:
        raise NeedsReview(message)


def fit_phi(angles, distances, quality):
    """Local quadratic diagnostic; uncertainty is conditional on this model."""
    angles, distances = np.asarray(angles), np.asarray(distances)
    valid = np.isfinite(angles) & np.isfinite(distances) & (distances > 0)
    x, y = angles[valid], distances[valid]
    check(len(np.unique(x)) >= max(4, quality["min_valid_phi_points"]),
          "Insufficient distinct valid angles for a phi fit and uncertainty")
    coefficients, covariance = np.polyfit(x, y, 2, cov=True)
    a, b, _ = coefficients
    check(a > 0, "The apparent-SDD fit has no positive-curvature minimum")
    vertex = -b / (2 * a)
    residuals = y - np.polyval(coefficients, x)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    relative_rmse = rmse / float(np.mean(y))
    gradient = np.array([b / (2 * a ** 2), -1 / (2 * a), 0])
    uncertainty = float(np.sqrt(max(0, gradient @ covariance @ gradient)))
    result = {
        "sample_phi_offset_deg": float(-vertex),
        "fit_uncertainty_deg": uncertainty,
        "uncertainty_model": "quadratic residual covariance; calibration systematics excluded",
        "coefficients": coefficients.tolist(), "angles_deg": x.tolist(),
        "apparent_sdd_cm": y.tolist(), "residuals_cm": residuals.tolist(),
        "rmse_cm": rmse, "relative_rmse": relative_rmse,
        "discarded_nonfinite_points": int(len(angles) - len(x)),
    }
    failures = []
    if not x.min() < vertex < x.max():
        failures.append("Phi minimum is outside the sampled angular window")
    if abs(vertex) > quality["max_phi_offset_abs_deg"]:
        failures.append("Phi offset exceeds its configured bound")
    if relative_rmse > quality["max_phi_fit_relative_rmse"]:
        failures.append("Phi fit residuals exceed the configured limit")
    if uncertainty > quality["max_phi_fit_uncertainty_deg"]:
        failures.append("Phi fit uncertainty exceeds the configured limit")
    result["failures"] = failures
    return result


def measured_peaks(values, prominence_relative=0.1, min_distance=5):
    """Find separated peaks after mild smoothing, retaining measured coverage."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return np.array([], dtype=int), np.array([])
    baseline = float(np.nanmedian(values[valid]))
    profile = gaussian_filter1d(np.where(valid, values, baseline), 0.7)
    contrast = float(np.max(profile) - np.median(profile))
    if contrast <= 0:
        return np.array([], dtype=int), np.array([])
    peaks = tools.find_peaks_1D(
        profile, log_scale=False, refinement=False, mask=~valid,
        algorithm="scipy", prominence=contrast * prominence_relative,
        distance=max(1, int(min_distance))).astype(int)
    widths = peak_widths(profile, peaks, rel_height=0.5)[0] if len(peaks) else np.array([])
    return peaks, widths


def representative_frames(dataset, count=5):
    frames = sorted(dataset.datas.values(), key=lambda d: d.sample_phi_deg)
    indexes = set(np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int))
    indexes.add(min(range(len(frames)), key=lambda i: abs(frames[i].sample_phi_deg)))
    return [frames[i] for i in sorted(indexes)]


def row_center(data, row):
    """Translate a sample-qsy row or legacy upward pixel offset per frame."""
    beam_y, beam_x = data.metadata["center_px_detector"]
    if row.get("offset_qdy_px") is not None:
        return float(beam_y - row["offset_qdy_px"])
    column = int(np.clip(round(beam_x), 0, data.image.shape[1] - 1))
    axis = data.qsy[:, column]
    valid = np.flatnonzero(np.isfinite(axis))
    check(len(valid) > 0, "No usable sample-qsy coordinates for this row")
    target = row["target_qsy_Ainv"]
    check(np.nanmin(axis) <= target <= np.nanmax(axis),
          f"qy row {target:g} A^-1 is outside the detector")
    order = valid[np.argsort(axis[valid])]
    return float(np.interp(target, axis[order], order))


def box_ranges(data, row, geometry):
    height, width = geometry["height_px"], geometry["width_px"]
    yr, xr = data.get_box_dims_size(
        height, width,
        center=(row_center(data, row), data.metadata["center_px_detector"][1]))
    check(yr[1] - yr[0] == height and xr[1] - xr[0] == width,
          f"Integration box exceeds the detector for {data.name}")
    return yr, xr


def row_peaks(data, row, settings):
    """Estimate peak centers and transverse extents in a search strip."""
    y = row_center(data, row)
    half = settings["integration"]["search_height_px"] // 2
    lo, hi = max(0, round(y) - half), min(data.image.shape[0], round(y) + half + 1)
    image = np.where(data.mask[lo:hi], np.nan, data.image[lo:hi])
    valid_columns = np.isfinite(image).any(axis=0)
    profile = np.max(np.where(np.isfinite(image), image, -np.inf), axis=0)
    profile[~valid_columns] = np.nan
    s = settings["slices"]
    peaks, widths = measured_peaks(profile, s["prominence_relative"], s["min_distance_px"])
    if row.get("row_id") == "qy0":
        center = data.metadata["center_px_detector"][1]
        keep = abs(peaks - center) > s["exclude_center_px"]
        peaks, widths = peaks[keep], widths[keep]
    points = []
    for x, width in zip(peaks, widths):
        column = image[:, x]
        finite = np.isfinite(column)
        if not finite.any():
            continue
        candidates, _ = measured_peaks(column, s["prominence_relative"], 1)
        if not len(candidates):
            continue
        candidate = min(candidates, key=lambda index: abs(index + lo - y))
        # Isolate this row even if the search strip contains neighboring rows.
        near_peak = abs(np.arange(len(column)) - candidate) <= 4
        weights = np.where(finite & near_peak, np.maximum(column - np.nanmin(column), 0), 0)
        if weights.sum() <= 0:
            continue
        axis = np.arange(lo, hi)
        peak_y, _ = tools.com_refinement(weights[:, None])
        py = float(lo + peak_y)
        sigma = float(np.sqrt(np.sum((axis - py) ** 2 * weights) / weights.sum()))
        points.append({"x_px": int(x), "y_px": py,
                       "sigma_y_px": sigma, "fwhm_x_px": float(width)})
    return points


def geometry_proposal(dataset, row, settings, *, all_frames=False,
                      select_backgrounds=True):
    frames = list(dataset.datas.values()) if all_frames else representative_frames(dataset)
    measurements = [(data, row_peaks(data, row, settings)) for data in frames]
    points = [point for _, peaks in measurements for point in peaks]
    check(points, "No peaks found for integration-box selection; tune the search")
    cfg = settings["integration"]
    residuals = [abs(p["y_px"] - row_center(data, row))
                 for data, peaks in measurements for p in peaks]
    extent = max(abs(p["y_px"] - row_center(data, row)) + 2.5 * p["sigma_y_px"]
                 for data, peaks in measurements for p in peaks)
    required_height = max(cfg["height_bounds_px"][0], int(np.ceil(2 * extent + 1)))
    height = cfg["height_px"] if cfg["height_px"] is not None else required_height
    if cfg["height_px"] is None:
        check(height <= cfg["height_bounds_px"][1],
              "Required integration height exceeds its configured bound")
    # Box is centered at the beam column; include peaks on both sides.
    extent_x = max(abs(p["x_px"] - data.metadata["center_px_detector"][1])
                   + p["fwhm_x_px"] / 2 + cfg["peak_padding_px"]
                   for data, peaks in measurements for p in peaks)
    width = cfg["width_px"] if cfg["width_px"] is not None else int(np.ceil(2 * extent_x + 1))
    geometry = {"height_px": int(height), "width_px": int(width),
                "background_offsets_px": cfg["background_offsets_px"], "mode": cfg["mode"]}
    fractions = []
    valid_fractions = []
    for data, peaks in measurements:
        yrange, xrange = box_ranges(data, row, geometry)
        fractions.extend(yrange[0] <= p["y_px"] < yrange[1]
                         and xrange[0] <= p["x_px"] < xrange[1] for p in peaks)
        roi = data.mask[yrange[0]:yrange[1], xrange[0]:xrange[1]]
        valid_fractions.append(float(np.mean(~roi)))
    geometry["metrics"] = {
        "alignment_residual_px": float(np.percentile(residuals, 95)),
        "peak_inside_fraction": float(np.mean(fractions)),
        "min_roi_valid_fraction": min(valid_fractions),
        "frames_checked": [d.name for d in frames],
    }
    failures = []
    q = settings["quality"]
    if row["row_id"] == "qy0" and geometry["metrics"]["alignment_residual_px"] > q["max_alignment_residual_px"]:
        failures.append("Corrected peak alignment exceeds its residual limit")
    if geometry["metrics"]["peak_inside_fraction"] < q["min_peak_inside_fraction"]:
        failures.append("Integration box does not contain enough detected peaks")
    if min(valid_fractions) < q["min_roi_valid_fraction"]:
        failures.append("Integration region has insufficient unmasked coverage")
    if cfg["subtract_background"] and select_backgrounds:
        try:
            # Evaluate one fixed selection against every production frame.
            measured = (cfg["background_score_scope"] == "slice_windows"
                        and settings["slices"]["centers_qsx_Ainv"] is None)
            background_measurements = [(data, row_peaks(data, row, settings) if measured else [])
                                       for data in dataset.datas.values()]
            geometry["background_offsets_px"], background_metric = choose_backgrounds(
                background_measurements, row, geometry, settings, require_complete=True)
            geometry["metrics"]["background_peak_fraction"] = background_metric
            geometry["background_diagnostics"]["production_frames"] = deepcopy(
                geometry["background_diagnostics"]["selected_frames"])
        except NeedsReview as error:
            geometry["failures"] = failures + [str(error)]
            error.result = geometry
            raise
        if background_metric > q["max_background_peak_fraction"]:
            failures.append("Background regions contain excessive peak signal")
    elif not cfg["subtract_background"]:
        geometry["background_offsets_px"] = None
    geometry["failures"] = failures
    return geometry


def shared_geometry_proposal(dataset, row_configs):
    """Choose one geometry against every requested row and production frame."""
    proposals = [geometry_proposal(dataset, row, settings, all_frames=True,
                                   select_backgrounds=False)
                 for row, settings in row_configs]
    settings = deepcopy(row_configs[0][1])
    cfg = settings["integration"]
    cfg["height_px"] = max(item["height_px"] for item in proposals)
    cfg["width_px"] = max(item["width_px"] for item in proposals)
    geometry = {key: cfg[key] for key in (
        "height_px", "width_px", "mode", "background_offsets_px")}
    geometry["failures"] = []
    geometry["rows"] = {}
    contexts = []
    for row, original in row_configs:
        fixed = deepcopy(original)
        fixed["integration"] = deepcopy(cfg)
        proposal = geometry_proposal(dataset, row, fixed, all_frames=True,
                                     select_backgrounds=False)
        geometry["rows"][row["row_id"]] = proposal["metrics"]
        geometry["failures"].extend(
            f"{row['row_id']}: {failure}" for failure in proposal["failures"])
        if cfg["subtract_background"]:
            for data in dataset.datas.values():
                peaks = row_peaks(data, row, fixed)
                context = background_context(data, row, geometry, fixed, peaks)
                context["row_id"] = row["row_id"]
                contexts.append(context)
    if cfg["subtract_background"]:
        try:
            offsets, score = choose_backgrounds(
                [], row_configs[0][0], geometry, settings, contexts=contexts,
                require_complete=True)
            geometry["background_offsets_px"] = offsets
            geometry["background_diagnostics"]["production_frames"] = deepcopy(
                geometry["background_diagnostics"]["selected_frames"])
            for row_id, metrics in geometry["rows"].items():
                metrics["background_peak_fraction"] = max(
                    item["contrast_fraction"] for item in geometry["background_diagnostics"]["selected_frames"]
                    if item["row_id"] == row_id)
            if score > settings["quality"]["max_background_peak_fraction"]:
                geometry["failures"].append("Shared backgrounds contain excessive peak signal")
        except NeedsReview as error:
            geometry["failures"].append(str(error))
            error.result = geometry
            raise
    else:
        geometry["background_offsets_px"] = None
    return geometry


def background_context(data, row, geometry, settings, peaks):
    """Use native signal columns; background artifacts outside slices cannot veto a row."""
    yr, xr = box_ranges(data, row, geometry)
    signal, _, _ = data.mean_box(yr, xr, axis=0)
    columns = np.ones(len(signal), dtype=bool)
    scope = "full_box"
    cfg = settings["slices"]
    if settings["integration"]["background_score_scope"] == "slice_windows":
        centers, widths = cfg["centers_qsx_Ainv"], cfg["widths_Ainv"]
        if centers is not None and widths is not None:
            qsx = np.nanmean(data.qsx[yr[0]:yr[1], xr[0]:xr[1]], axis=0)
            widths = widths if isinstance(widths, list) else [widths] * len(centers)
            columns[:] = False
            for center, width in zip(centers, widths):
                columns |= abs(qsx - center) <= width / 2
            scope = "supplied_slice_windows"
        elif peaks:
            columns[:] = False
            axis = np.arange(xr[0], xr[1])
            for peak in peaks:
                radius = peak["fwhm_x_px"] / 2 + settings["integration"]["peak_padding_px"]
                columns |= abs(axis - peak["x_px"]) <= radius
            scope = "measured_peak_windows"
    # Signal bins already lost to a mask are handled by native slice coverage.
    columns &= np.isfinite(signal)
    baseline = float(np.nanmedian(signal)) if np.isfinite(signal).any() else 0.0
    contrast = (max(0.0, float(np.max(signal[columns]) - baseline))
                if columns.any() else 0.0)
    global_contrast = (max(0.0, float(np.nanmax(signal) - baseline))
                       if np.isfinite(signal).any() else 0.0)
    return {"data": data, "yr": yr, "xr": xr, "columns": columns,
            "scope": scope, "signal_contrast": contrast,
            "global_signal_contrast": global_contrast}


def background_candidate(data, offset, context, settings):
    """Save the frame, side, mask evidence, and measured contrast for each candidate."""
    yr, xr, columns = context["yr"], context["xr"], context["columns"]
    lo, hi = yr[0] + offset, yr[1] + offset
    outside = lo < 0 or hi > data.image.shape[0]
    result = {"frame": data.name, "phi_deg": float(data.sample_phi_deg),
              "offset_px": int(offset), "rows_px": [int(lo), int(hi)],
              "score_scope": context["scope"], "usable": False,
              "signal_rows_px": [int(value) for value in yr],
              "scored_column_count": int(columns.sum()), "scored_valid_column_count": 0,
              "gap_overlap": outside, "reason": "outside_detector" if outside else None,
              "contrast_fraction": None, "full_box_contrast_fraction": None}
    if "row_id" in context:
        result["row_id"] = context["row_id"]
    if outside:
        return result
    profile, image, mask = data.mean_box((lo, hi), xr, axis=0)
    invalid = mask | ~np.isfinite(image)
    result["valid_fraction"] = float(np.mean(~invalid))
    result["gap_overlap"] = bool(columns.any() and invalid[:, columns].any())
    finite = np.isfinite(profile)
    if (result["valid_fraction"] < settings["quality"]["min_roi_valid_fraction"]
            or not finite.any()):
        result["reason"] = "insufficient_unmasked_background"
        return result
    baseline = float(np.nanmedian(profile))
    relevant = columns & finite
    contrast = (max(0.0, float(np.max(profile[relevant]) - baseline))
                if relevant.any() else 0.0)
    full_contrast = max(0.0, float(np.nanmax(profile) - baseline))
    scored_max = (int(xr[0] + np.flatnonzero(relevant)[np.argmax(profile[relevant])])
                  if relevant.any() else None)
    result.update(usable=True, scored_valid_column_count=int(relevant.sum()),
                  contrast_fraction=contrast / max(context["signal_contrast"], 1e-30),
                  full_box_contrast_fraction=full_contrast / max(context["global_signal_contrast"], 1e-30),
                  max_column_px=int(xr[0] + np.nanargmax(profile)),
                  scored_max_column_px=scored_max,
                  signal_contrast=context["signal_contrast"], background_contrast=contrast)
    return result


def background_frame_diagnostics(candidates, context, settings):
    """Score a fixed background selection without changing offsets by frame."""
    single_requested = settings["integration"].get("background_count", 2) == 1
    if single_requested and len(candidates) != 1:
        return None
    if not candidates or not all(item["usable"] and not item["gap_overlap"] for item in candidates):
        return None
    support = min(item.get("scored_valid_column_count", item["scored_column_count"])
                  / max(item["scored_column_count"], 1) for item in candidates)
    if single_requested and (support <= 0 or support < settings["quality"]["min_roi_valid_fraction"]
            or any(item["contrast_fraction"] > settings["quality"]["max_background_peak_fraction"]
                   for item in candidates)):
        return None
    return {"frame": candidates[0]["frame"], "phi_deg": candidates[0]["phi_deg"],
            "offsets_px": [item["offset_px"] for item in candidates],
            "contrast_fraction": max(item["contrast_fraction"] for item in candidates),
            "single_background_requested": single_requested,
            "signal_column_support_fraction": support}


def choose_backgrounds(measurements, row, geometry, settings, *, contexts=None,
                       require_complete=False):
    """Select one common offset list that passes the checked frames."""
    cfg = settings["integration"]
    h = geometry["height_px"]
    offsets = cfg["background_offsets_px"]
    independent = cfg["allow_independent_backgrounds"]
    count = cfg.get("background_count", 2)
    if offsets is not None and count == 1:
        check(len(offsets) == 1, "background_count=1 requires exactly one background offset")
    if offsets is not None and count != 1 and not independent:
        check(is_symmetric_background_pair(offsets),
              "Background offsets must be a symmetric [-d, +d] pair; "
              "set integration.allow_independent_backgrounds=true "
              "to use independent offsets")
    if offsets is None:
        strategy = cfg["background_strategy"]
        if strategy == "auto":
            check(settings["sample_kind"] is not None,
                  "Specify sample_kind or background_strategy for background selection")
            strategy = "distant" if settings["sample_kind"] == "line_grating" else "adjacent"
        # A displacement h touches the signal box; 2*h would leave a gap h.
        distances = range(5 * h, 10 * h, h) if strategy == "distant" else range(h, 2 * h)
        offsets = [sign * distance for distance in distances
                   for sign in (-1, 1)]
    if contexts is None:
        contexts = [background_context(data, row, geometry, settings, peaks)
                    for data, peaks in measurements]
    records = {}
    for offset in offsets:
        check(abs(offset) >= h, "Background regions overlap the integration box")
        records[offset] = [background_candidate(context["data"], offset, context, settings)
                           for context in contexts]
    geometry["background_diagnostics"] = {"requested_scope": cfg["background_score_scope"],
        "background_count_requested": count,
        "candidates": [item for frame_records in records.values() for item in frame_records]}
    def complete_candidate(item):
        return (item["usable"] and not item["gap_overlap"]
                and item["scored_valid_column_count"] > 0
                and item["scored_valid_column_count"] / max(item["scored_column_count"], 1)
                >= settings["quality"]["min_roi_valid_fraction"]
                and item["contrast_fraction"] <= settings["quality"]["max_background_peak_fraction"])

    scored = [(offset, max(item["contrast_fraction"] for item in items))
              for offset, items in records.items()
              if all(complete_candidate(item) if require_complete else item["usable"] for item in items)]
    check(any(item["usable"] for items in records.values() for item in items),
          "No usable background regions; specify different offsets")

    def choices(selected, *, strict=False):
        if (require_complete or strict) and any(not complete_candidate(records[offset][i])
                for offset in selected for i in range(len(contexts))):
            return None
        result = [background_frame_diagnostics([records[offset][i] for offset in selected],
                                          context, settings)
                  for i, context in enumerate(contexts)]
        for context, item in zip(contexts, result):
            if item is not None and "row_id" in context:
                item["row_id"] = context["row_id"]
        return result if all(item is not None for item in result) else None

    def finish(selected, frame_choices, *, omitted=None):
        diagnostics = geometry["background_diagnostics"]
        diagnostics["selected_frames"] = frame_choices
        if omitted is not None:
            affected = [{key: item[key] for key in (
                "frame", "phi_deg", "offset_px", "rows_px", "row_id") if key in item}
                for item in records[omitted] if item["gap_overlap"]]
            diagnostics["gap_fallback"] = {
                "retained_offsets_px": [int(offset) for offset in selected],
                "omitted_offsets_px": [int(omitted)],
                "affected_frames": affected,
                "reason": "detector_mask_gap",
            }
        return [int(offset) for offset in selected], float(max(
            item["contrast_fraction"] for item in frame_choices))

    def gap_fallback(pairs):
        # Drop a gap-affected side for the whole dataset, never by frame/row.
        # The retained side must independently pass coverage and contrast checks.
        candidates = []
        for negative, positive in pairs:
            for retained, omitted in ((negative, positive), (positive, negative)):
                if not any(item["gap_overlap"] and item["reason"] != "outside_detector"
                           for item in records[omitted]):
                    continue
                frames = choices([retained], strict=True)
                if frames is not None:
                    candidates.append((retained, omitted, frames))
        if not candidates:
            return None
        retained, omitted, frames = min(candidates, key=lambda item: (
            -min(frame["signal_column_support_fraction"] for frame in item[2]),
            max(frame["contrast_fraction"] for frame in item[2]),
            abs(item[0]), item[0], abs(item[1])))
        return finish([retained], frames, omitted=omitted)

    if cfg["background_offsets_px"] is None:
        if count == 1:
            fixed = [(offset, choices([offset])) for offset in offsets]
            fixed = [(offset, frames) for offset, frames in fixed if frames is not None]
            check(fixed, "No constant single background passes every frame; "
                  "adjust its fixed offset or search range")
            offset, frame_choices = min(fixed, key=lambda item: (
                -min(frame["signal_column_support_fraction"] for frame in item[1]),
                max(frame["contrast_fraction"] for frame in item[1]), abs(item[0]), item[0]))
            return finish([offset], frame_choices)
        elif independent:
            selected = []
            for sign in (-1, 1):
                side = [item for item in scored if np.sign(item[0]) == sign
                        and choices([item[0]]) is not None]
                if side:
                    selected.append(min(side, key=lambda item: (item[1], abs(item[0])))[0])
            if len(selected) != 2:
                fallback = gap_fallback([(negative, positive) for negative in offsets if negative < 0
                                         for positive in offsets if positive > 0])
                if fallback is not None:
                    return fallback
                check(False, "No usable background region on one side; specify different offsets")
        else:
            pairs = []
            for distance in offsets:
                if distance > 0 and -distance in records:
                    frame_choices = choices([-distance, distance])
                    if frame_choices is not None:
                        score = max(item["contrast_fraction"] for item in frame_choices)
                        pairs.append((score, distance))
            if not pairs:
                fallback = gap_fallback([(-distance, distance) for distance in offsets
                                         if distance > 0 and -distance in records])
                if fallback is not None:
                    return fallback
                check(False, "No usable symmetric background pair; adjust the search "
                      "or explicitly allow independent backgrounds")
            _, distance = min(pairs)
            selected = [-distance, distance]
    else:
        selected = offsets
    frame_choices = choices(selected)
    if frame_choices is None and count == 2 and len(selected) == 2 and selected[0] * selected[1] < 0:
        fallback = gap_fallback([tuple(sorted(selected))])
        if fallback is not None:
            return fallback
    message = ("Single background fails contrast or unmasked-coverage requirements" if count == 1
               else "Background offsets fail shared detector coverage or contrast requirements")
    check(frame_choices is not None, message)
    return finish(selected, frame_choices)


def detect_qy_rows(data, settings):
    cfg = settings["additional_qy_rows"]
    image = np.where(data.mask, np.nan, data.image).copy()
    x = round(data.metadata["center_px_detector"][1])
    radius = settings["slices"]["exclude_center_px"]
    image[:, max(0, x - radius):x + radius + 1] = np.nan
    profile = np.nanmax(image, axis=1)
    peaks, _ = measured_peaks(profile, cfg["prominence_relative"], cfg["min_separation_px"])
    center = data.metadata["center_px_detector"][0]
    peaks = peaks[abs(peaks - center) >= cfg["min_separation_px"]]
    ranked = sorted(peaks, key=lambda p: profile[p], reverse=True)
    kept, dropped = ranked[:cfg["max_rows"]], ranked[cfg["max_rows"]:]
    offsets = []
    for peak in kept:
        lo, hi = max(0, peak - 3), min(len(profile), peak + 4)
        weights = np.maximum(np.nan_to_num(profile[lo:hi], nan=0) - np.nanmedian(profile), 0)
        position, _ = tools.com_refinement(weights[:, None])
        position += lo
        valid = np.flatnonzero(np.isfinite(data.qsy[:, x]))
        if len(valid):
            offsets.append(float(np.interp(position, valid, data.qsy[valid, x])))
    offsets.sort()
    check(offsets, "No additional qy rows detected; supply offsets or tune row detection")
    return {"offsets_qsy_Ainv": offsets,
            "omitted_detector_rows_px": [int(p) for p in dropped],
            "reference_frame": data.name}


def slice_coverage(dataset, center, width, *, mode="sum"):
    """Report native per-frame slice acceptance without repeating its rules.

    The native slicer exposes selected pixels only for accepted frames and
    does not distinguish rejection causes. Rejected pixel counts are unknown.
    """
    coverage = []
    for data in dataset.datas:
        slices, _ = slice_reduced_dataset(
            ReducedDataset([data]), [center], q_widths=width,
            show_plot=False, mode=mode)
        native_slice = slices.data[0]
        valid = bool(len(native_slice.Iq))
        selected_pixels = len(native_slice.plotting_data["qsx"]) if valid else None
        coverage.append({"frame": data.data2d.name,
                         "phi_deg": float(data.sample_phi_deg),
                         "valid": valid, "selected_pixels": selected_pixels,
                         "reason": None if valid else "native_slice_rejected"})
    return coverage


def slice_parameters(integrated, settings):
    """Find centers and per-peak width estimates before choosing a common width."""
    cfg = settings["slices"]
    reference_index = min(range(len(integrated.datas)),
                          key=lambda index: abs(integrated.datas[index].sample_phi_deg))
    reference = integrated.datas[reference_index]
    centers = cfg["centers_qsx_Ainv"]
    detected_widths = None
    if centers is None:
        profile = np.where(reference.mask, np.nan, reference.Iq)
        peaks, widths = measured_peaks(profile, cfg["prominence_relative"], cfg["min_distance_px"])
        if getattr(reference, "row_id", "qy0") == "qy0":
            keep = abs(peaks + reference.limits_axis1[0]
                       - reference.data2d.metadata["center_px_detector"][1]) > cfg["exclude_center_px"]
            peaks, widths = peaks[keep], widths[keep]
        check(len(peaks) > 0, "No peak slice centers detected; supply centers or tune the search")
        centers = reference.qsx[peaks].tolist()
        spacing = abs(float(np.nanmedian(np.diff(reference.qsx))))
        detected_widths = np.clip(1.5 * widths * spacing, *cfg["width_bounds_Ainv"]).tolist()
    elif cfg["widths_Ainv"] is None:
        profile = np.where(reference.mask, np.nan, reference.Iq)
        peaks, peak_width = measured_peaks(profile, cfg["prominence_relative"], cfg["min_distance_px"])
        spacing = abs(float(np.nanmedian(np.diff(reference.qsx))))
        detected_widths = []
        for center in centers:
            width = spacing * 1.5
            if len(peaks):
                nearest = int(np.argmin(abs(reference.qsx[peaks] - center)))
                width = max(width, 1.5 * peak_width[nearest] * spacing)
            detected_widths.append(float(np.clip(width, *cfg["width_bounds_Ainv"])))
    widths = cfg["widths_Ainv"]
    if widths is None:
        widths = detected_widths or [cfg["width_bounds_Ainv"][0]] * len(centers)
    elif not isinstance(widths, list):
        widths = [widths] * len(centers)
    check(len(widths) == len(centers), "Slice widths do not match detected centers")
    return centers, widths, reference_index


def common_slice_width(row_integrations):
    """Choose the largest bounded automatic estimate across all qy rows.

    Explicit scalar/list widths are preserved and do not participate in
    automatic selection. Conflicting row-specific bounds require review.
    """
    automatic = [(integrated, settings) for integrated, settings in row_integrations
                 if settings["slices"]["widths_Ainv"] is None]
    if not automatic:
        return None
    width = max(value for integrated, settings in automatic
                for value in slice_parameters(integrated, settings)[1])
    for _, settings in automatic:
        low, high = settings["slices"]["width_bounds_Ainv"]
        check(low <= width <= high,
              "Shared automatic qsx width exceeds a row's configured bounds; "
              "align width bounds or supply explicit slice widths")
    return float(width)


def slice_proposal(integrated, settings, *, shared_width_Ainv=None):
    cfg = settings["slices"]
    centers, widths, reference_index = slice_parameters(integrated, settings)
    reference = integrated.datas[reference_index]
    if cfg["widths_Ainv"] is None:
        width = max(widths) if shared_width_Ainv is None else shared_width_Ainv
        widths = [width] * len(centers)
    proposals = []
    for center, width in zip(centers, widths):
        neighbors = [abs(center - other) for other in centers if other != center]
        if neighbors:
            check(width < min(neighbors), "A slice width reaches an adjacent peak center")
        coverage = slice_coverage(integrated, center, width, mode=cfg["mode"])
        fraction = sum(item["valid"] for item in coverage) / len(coverage)
        reference_coverage = coverage[reference_index]
        skip_reasons = []
        if not reference_coverage["valid"]:
            skip_reasons.append("invalid_near_zero_frame")
        if fraction < settings["quality"]["min_slice_angle_coverage"]:
            skip_reasons.append("insufficient_angle_coverage")
        proposals.append({"center_qsx_Ainv": float(center), "width_Ainv": float(width),
                          "angle_coverage": fraction, "coverage": coverage,
                          "reference_frame_valid": reference_coverage["valid"],
                          "reference_frame_reason": reference_coverage["reason"],
                          "skip_reasons": skip_reasons, "skip": bool(skip_reasons)})
    return {"slices": proposals, "reference_frame": reference.data2d.name,
            "reference_phi_deg": float(reference.sample_phi_deg), "mode": cfg["mode"]}


def phi_figure(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    x = np.array(result["angles_deg"])
    grid = np.linspace(x.min(), x.max(), 200)
    axes[0].scatter(x, result["apparent_sdd_cm"], color="#0072B2")
    axes[0].plot(grid, np.polyval(result["coefficients"], grid), color="#D55E00")
    axes[0].axvline(-result["sample_phi_offset_deg"], color="0.3", ls="--")
    axes[0].set_ylabel("Apparent SDD (cm)")
    axes[0].set_title(f"Phi offset = {result['sample_phi_offset_deg']:.4g} ± "
                      f"{result['fit_uncertainty_deg']:.2g}° (fit only)")
    axes[1].scatter(x, result["residuals_cm"], color="#0072B2")
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set(xlabel="Nominal phi (degrees)", ylabel="Residual (cm)")
    fig.tight_layout()
    return fig


def row_figure(data, rows):
    fig = data.plot_data(show_q=False)
    ax = fig.axes[0]
    for row in rows:
        y = row_center(data, row)
        ax.axhline(y, color="#E69F00", lw=1.3)
        ax.text(2, y, row["row_id"], color="white", va="bottom")
    ax.set(xlabel="Detector column (px)", ylabel="Detector row (px)",
           title=f"Selected qy rows: {data.name}")
    fig.tight_layout()
    return fig
