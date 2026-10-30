"""Fit checks, measured peak selection, and publication-ready diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


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
    peaks, _ = find_peaks(profile, prominence=contrast * prominence_relative,
                          distance=max(1, int(min_distance)))
    peaks = peaks[valid[peaks]]
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
    # Interpolation locates a center only; integration uses detector pixels.
    order = valid[np.argsort(axis[valid])]
    return float(np.interp(target, axis[order], order))


def box_ranges(data, row, geometry):
    height, width = geometry["height_px"], geometry["width_px"]
    y = round(row_center(data, row))
    x = round(data.metadata["center_px_detector"][1])
    ymin, xmin = y - height // 2, x - width // 2
    ymax, xmax = ymin + height, xmin + width
    check(0 <= ymin < ymax <= data.image.shape[0]
          and 0 <= xmin < xmax <= data.image.shape[1],
          f"Integration box exceeds the detector for {data.name}")
    return (int(ymin), int(ymax)), (int(xmin), int(xmax))


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
        py = float(np.sum(axis * weights) / weights.sum())
        sigma = float(np.sqrt(np.sum((axis - py) ** 2 * weights) / weights.sum()))
        points.append({"x_px": int(x), "y_px": py,
                       "sigma_y_px": sigma, "fwhm_x_px": float(width)})
    return points


def geometry_proposal(dataset, row, settings):
    frames = representative_frames(dataset)
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
    if cfg["subtract_background"]:
        geometry["background_offsets_px"], background_metric = choose_backgrounds(
            measurements, row, geometry, settings)
        geometry["metrics"]["background_peak_fraction"] = background_metric
        if background_metric > q["max_background_peak_fraction"]:
            failures.append("Background regions contain excessive peak signal")
    else:
        geometry["background_offsets_px"] = None
    geometry["failures"] = failures
    return geometry


def choose_backgrounds(measurements, row, geometry, settings):
    cfg = settings["integration"]
    h = geometry["height_px"]
    offsets = cfg["background_offsets_px"]
    if offsets is None:
        strategy = cfg["background_strategy"]
        if strategy == "auto":
            check(settings["sample_kind"] is not None,
                  "Specify sample_kind or background_strategy for background selection")
            strategy = "distant" if settings["sample_kind"] == "line_grating" else "adjacent"
        start = 5 if strategy == "distant" else 2
        offsets = [sign * multiple * h for multiple in range(start, start + 5)
                   for sign in (-1, 1)]
    scored = []
    for offset in offsets:
        check(abs(offset) >= h, "Background regions overlap the integration box")
        scores = []
        usable = True
        for data, _ in measurements:
            yr, xr = box_ranges(data, row, geometry)
            lo, hi = yr[0] + offset, yr[1] + offset
            if lo < 0 or hi > data.image.shape[0]:
                usable = False
                break
            bg = np.where(data.mask[lo:hi, xr[0]:xr[1]], np.nan,
                          data.image[lo:hi, xr[0]:xr[1]])
            sig = np.where(data.mask[yr[0]:yr[1], xr[0]:xr[1]], np.nan,
                           data.image[yr[0]:yr[1], xr[0]:xr[1]])
            if np.mean(np.isfinite(bg)) < settings["quality"]["min_roi_valid_fraction"]:
                usable = False
                break
            # Compare peak contrast per pixel, independently of box height.
            bg_profile = np.nanmean(bg, axis=0)
            sig_profile = np.nanmean(sig, axis=0)
            signal_contrast = float(np.nanmax(sig_profile) - np.nanmedian(sig_profile))
            bg_contrast = float(np.nanmax(bg_profile) - np.nanmedian(bg_profile))
            scores.append(max(0.0, bg_contrast) / max(signal_contrast, 1e-30))
        if usable:
            scored.append((int(offset), max(scores)))
        elif cfg["background_offsets_px"] is not None:
            raise NeedsReview(f"Background offset {offset} is outside usable detector coverage")
    check(scored, "No usable background regions; specify different offsets")
    if cfg["background_offsets_px"] is None:
        selected = []
        for sign in (-1, 1):
            side = [item for item in scored if np.sign(item[0]) == sign]
            if side:
                selected.append(min(side, key=lambda item: item[1]))
    else:
        selected = scored
    return [item[0] for item in selected], float(max(item[1] for item in selected))


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
        position = float(np.sum(np.arange(lo, hi) * weights) / weights.sum())
        valid = np.flatnonzero(np.isfinite(data.qsy[:, x]))
        if len(valid):
            offsets.append(float(np.interp(position, valid, data.qsy[valid, x])))
    offsets.sort()
    check(offsets, "No additional qy rows detected; supply offsets or tune row detection")
    return {"offsets_qsy_Ainv": offsets,
            "omitted_detector_rows_px": [int(p) for p in dropped],
            "reference_frame": data.name}


def slice_coverage(dataset, center, width):
    """Match the library's strict mask rule, also rejecting nonfinite coordinates."""
    coverage = []
    for data in dataset.datas:
        selection = np.flatnonzero(abs(data.qsx - center) <= width / 2)
        valid = (len(selection) > 0 and not data.mask[selection].any()
                 and all(np.isfinite(getattr(data, key)[selection]).all()
                         for key in ("Iq", "qsx", "qsy", "qsz", "qsr", "qs")))
        reason = None if valid else (
            "outside_q_range" if not len(selection) else "masked_or_nonfinite_selection"
        )
        coverage.append({"frame": data.data2d.name,
                         "phi_deg": float(data.sample_phi_deg),
                         "valid": bool(valid), "selected_pixels": int(len(selection)),
                         "reason": reason})
    return coverage


def slice_proposal(integrated, settings):
    cfg = settings["slices"]
    reference = min(integrated.datas, key=lambda d: abs(d.sample_phi_deg))
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
    proposals = []
    for center, width in zip(centers, widths):
        neighbors = [abs(center - other) for other in centers if other != center]
        if neighbors:
            check(width < min(neighbors), "A slice width reaches an adjacent peak center")
        coverage = slice_coverage(integrated, center, width)
        fraction = sum(item["valid"] for item in coverage) / len(coverage)
        proposals.append({"center_qsx_Ainv": float(center), "width_Ainv": float(width),
                          "angle_coverage": fraction, "coverage": coverage,
                          "skip": fraction < settings["quality"]["min_slice_angle_coverage"]})
    return {"slices": proposals, "reference_frame": reference.data2d.name, "mode": cfg["mode"]}


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
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(7, 6))
    values = np.ma.masked_where(data.mask | (data.image <= 0), data.image)
    artist = ax.imshow(values, cmap="viridis", norm=LogNorm(), origin="upper")
    for row in rows:
        y = row_center(data, row)
        ax.axhline(y, color="#E69F00", lw=1.3)
        ax.text(2, y, row["row_id"], color="white", va="bottom")
    ax.set(xlabel="Detector column (px)", ylabel="Detector row (px)",
           title=f"Selected qy rows: {data.name}")
    fig.colorbar(artist, ax=ax, label="Normalized intensity (a.u.)")
    fig.tight_layout()
    return fig
