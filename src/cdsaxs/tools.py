"""General tools for the code."""
import inspect
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from sklearn.linear_model import LinearRegression
from skimage import transform
from tqdm import tqdm

from .calculators import gaussian


def _peak_q_from_geometry(
        peak_positions,
        beam_center_px,
        sdd_cm,
        pixel_size_um,
        wavelength_nm):
    """Calculate q magnitudes for peak positions from detector geometry."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    beam_center_px = np.asarray(beam_center_px, dtype=float)

    radial_distance_cm = np.linalg.norm(
        peak_positions - beam_center_px,
        axis=1,
    ) * pixel_size_um * 1e-4
    theta = np.arctan2(radial_distance_cm, sdd_cm)
    q = 4 * np.pi * np.sin(theta / 2) / wavelength_nm

    return q


def _fit_diffraction_direction(peak_positions):
    """Estimate the diffraction-line direction from the peak cloud."""
    centered_positions = np.asarray(peak_positions, dtype=float)
    centered_positions = centered_positions - np.mean(centered_positions, axis=0)
    _, _, vh = np.linalg.svd(centered_positions, full_matrices=False)
    direction = vh[0]
    if direction[1] < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def _score_sdd_candidate(
        peak_positions,
        beam_center_px,
        sdd_cm,
        pixel_size_um,
        wavelength_nm,
        pitch_nm,
        diffraction_direction):
    """Score how well a geometry candidate matches the signed q lattice."""
    q = _peak_q_from_geometry(
        peak_positions=peak_positions,
        beam_center_px=beam_center_px,
        sdd_cm=sdd_cm,
        pixel_size_um=pixel_size_um,
        wavelength_nm=wavelength_nm,
    )
    signed_distance_px = np.dot(
        np.asarray(peak_positions, dtype=float) - beam_center_px,
        diffraction_direction,
    )
    q_signed = np.sign(signed_distance_px) * q
    q_spacing = 2 * np.pi / pitch_nm
    order_float = q_signed / q_spacing
    order_int = np.rint(order_float)
    order_int = np.where(order_int == 0, 1, order_int)
    q_model = q_spacing * order_int
    residual = q_signed - q_model

    return float(np.mean(residual**2)), order_int.astype(int), q_signed


def _score_sdd_ring_candidate(
        peak_positions,
        beam_center_px,
        sdd_cm,
        pixel_size_um,
        wavelength_nm,
        pitch_nm):
    """Score how well a geometry candidate matches radial diffraction rings."""
    q = _peak_q_from_geometry(
        peak_positions=peak_positions,
        beam_center_px=beam_center_px,
        sdd_cm=sdd_cm,
        pixel_size_um=pixel_size_um,
        wavelength_nm=wavelength_nm,
    )
    q_spacing = 2 * np.pi / pitch_nm
    ring_index_float = q / q_spacing
    ring_indices = np.rint(ring_index_float)
    ring_indices = np.where(ring_indices < 1, 1, ring_indices)
    unique_ring_indices, ring_counts = np.unique(
        ring_indices.astype(int),
        return_counts=True,
    )
    supported_ring_indices = unique_ring_indices[ring_counts >= 2]
    if supported_ring_indices.size > 0:
        supported_mask = np.isin(ring_indices, supported_ring_indices)
    else:
        supported_mask = np.ones(ring_indices.shape, dtype=bool)
    q_model = q_spacing * ring_indices
    residual = q[supported_mask] - q_model[supported_mask]
    unsupported_fraction = 1.0 - np.mean(supported_mask.astype(float))
    score = float(np.mean(residual**2)) + (q_spacing ** 2) * unsupported_fraction

    return (
        score,
        ring_indices.astype(int),
        q,
        supported_mask.astype(bool),
    )


def _estimate_ring_point_radial_uncertainty(
        peak_positions,
        beam_center_px,
        ring_indices,
        minimum_uncertainty_px=0.5):
    """Estimate radial uncertainty from within-ring radial spread."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    beam_center_px = np.asarray(beam_center_px, dtype=float)
    ring_indices = np.asarray(ring_indices, dtype=int)
    radial_distances_px = np.linalg.norm(peak_positions - beam_center_px, axis=1)
    radial_uncertainty_px = np.full(
        radial_distances_px.shape,
        float(minimum_uncertainty_px),
        dtype=float,
    )

    for ring_index in np.unique(ring_indices):
        ring_mask = ring_indices == ring_index
        ring_radii = radial_distances_px[ring_mask]
        if ring_radii.size >= 2:
            ring_sigma = float(np.std(ring_radii, ddof=1))
            radial_uncertainty_px[ring_mask] = max(
                ring_sigma,
                float(minimum_uncertainty_px),
            )

    return radial_distances_px, radial_uncertainty_px


def _gaussian_2d_model(coordinates, row0, col0, sigma_row, sigma_col,
                       amplitude, offset):
    """Evaluate an axis-aligned 2D Gaussian on flattened coordinates."""
    row, col = coordinates
    exponent = (
        ((row - row0) ** 2) / (2 * sigma_row ** 2)
        + ((col - col0) ** 2) / (2 * sigma_col ** 2)
    )
    return amplitude * np.exp(-exponent) + offset


def _estimate_peak_covariances_from_image(
        image,
        peak_positions,
        refinement_size=9):
    """Estimate per-peak center covariance matrices from local 2D fits."""
    image = np.asarray(image, dtype=float)
    peak_positions = np.asarray(peak_positions, dtype=float)
    covariances = []
    fitted_positions = []
    refinement_size = max(int(refinement_size), 5)

    for row_peak, col_peak in peak_positions:
        row_center = int(np.round(row_peak))
        col_center = int(np.round(col_peak))
        row_min = max(0, row_center - refinement_size // 2)
        row_max = min(image.shape[0], row_min + refinement_size)
        row_min = max(0, row_max - refinement_size)
        col_min = max(0, col_center - refinement_size // 2)
        col_max = min(image.shape[1], col_min + refinement_size)
        col_min = max(0, col_max - refinement_size)

        image_window = image[row_min:row_max, col_min:col_max]
        row_grid, col_grid = np.indices(image_window.shape, dtype=float)
        row_grid += row_min
        col_grid += col_min

        offset = float(np.nanmin(image_window))
        amplitude = float(np.nanmax(image_window) - offset)
        if not np.isfinite(amplitude) or amplitude <= 0:
            amplitude = 1.0
        p0 = [
            float(row_peak),
            float(col_peak),
            max(refinement_size / 4, 1.0),
            max(refinement_size / 4, 1.0),
            amplitude,
            offset,
        ]
        lower_bounds = [
            row_min,
            col_min,
            0.25,
            0.25,
            0.0,
            -np.inf,
        ]
        upper_bounds = [
            max(row_max - 1, row_min),
            max(col_max - 1, col_min),
            max(refinement_size, 1.0),
            max(refinement_size, 1.0),
            np.inf,
            np.inf,
        ]

        try:
            popt, pcov = curve_fit(
                _gaussian_2d_model,
                (row_grid.ravel(), col_grid.ravel()),
                image_window.ravel(),
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
            )
            fitted_positions.append(popt[:2])
            covariances.append(np.asarray(pcov[:2, :2], dtype=float))
        except (RuntimeError, ValueError):
            fitted_positions.append([row_peak, col_peak])
            covariances.append(np.eye(2, dtype=float) * 0.25)

    return np.asarray(fitted_positions, dtype=float), covariances


def _project_peak_covariances_to_radial_uncertainty(
        peak_positions,
        beam_center_px,
        peak_covariances):
    """Project 2D peak covariance matrices onto radial directions."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    beam_center_px = np.asarray(beam_center_px, dtype=float)
    sigma_r_px = []

    for peak_position, covariance in zip(peak_positions, peak_covariances):
        radial_vector = peak_position - beam_center_px
        radial_norm = np.linalg.norm(radial_vector)
        if radial_norm <= 0:
            sigma_r_px.append(float(np.sqrt(np.max(np.diag(covariance)))))
            continue
        radial_unit = radial_vector / radial_norm
        variance_r = float(radial_unit @ covariance @ radial_unit)
        sigma_r_px.append(float(np.sqrt(max(variance_r, 1e-12))))

    return np.asarray(sigma_r_px, dtype=float)


def _estimate_peak_radial_uncertainty_from_image(
        image,
        peak_positions,
        beam_center_px,
        refinement_size=9):
    """Estimate fitted peak positions and radial uncertainties."""
    fitted_peak_positions, peak_covariances = (
        _estimate_peak_covariances_from_image(
            image=image,
            peak_positions=peak_positions,
            refinement_size=refinement_size,
        )
    )
    peak_position_uncertainty_px = (
        _project_peak_covariances_to_radial_uncertainty(
            peak_positions=fitted_peak_positions,
            beam_center_px=beam_center_px,
            peak_covariances=peak_covariances,
        )
    )

    return (
        fitted_peak_positions,
        peak_position_uncertainty_px,
        peak_covariances,
    )


def _radial_gaussian_model(radius, radius0, sigma_r, amplitude, offset):
    """Evaluate a 1D Gaussian in detector radius."""
    return amplitude * np.exp(
        -((radius - radius0) ** 2) / (2 * sigma_r ** 2)
    ) + offset


def _estimate_ring_peak_radial_uncertainty_from_image(
        image,
        peak_positions,
        beam_center_px,
        refinement_size=9,
        minimum_uncertainty_px=0.5):
    """Refine arc-like ring samples by fitting a local radial profile."""
    image = np.asarray(image, dtype=float)
    peak_positions = np.asarray(peak_positions, dtype=float)
    beam_center_px = np.asarray(beam_center_px, dtype=float)
    refinement_size = max(int(refinement_size), 5)

    fitted_peak_positions = []
    radial_uncertainties_px = []
    fit_success = []

    for row_peak, col_peak in peak_positions:
        seed_position = np.asarray([row_peak, col_peak], dtype=float)
        radial_vector = seed_position - beam_center_px
        radial_norm = np.linalg.norm(radial_vector)
        if radial_norm <= 0:
            fitted_peak_positions.append(seed_position)
            radial_uncertainties_px.append(float(minimum_uncertainty_px))
            fit_success.append(False)
            continue

        radial_unit = radial_vector / radial_norm

        row_center = int(np.round(row_peak))
        col_center = int(np.round(col_peak))
        row_min = max(0, row_center - refinement_size // 2)
        row_max = min(image.shape[0], row_min + refinement_size)
        row_min = max(0, row_max - refinement_size)
        col_min = max(0, col_center - refinement_size // 2)
        col_max = min(image.shape[1], col_min + refinement_size)
        col_min = max(0, col_max - refinement_size)

        image_window = image[row_min:row_max, col_min:col_max]
        row_grid, col_grid = np.indices(image_window.shape, dtype=float)
        row_grid += row_min
        col_grid += col_min

        window_positions = np.column_stack((row_grid.ravel(), col_grid.ravel()))
        window_values = image_window.ravel()
        finite_mask = np.isfinite(window_values)
        if not np.any(finite_mask):
            fitted_peak_positions.append(seed_position)
            radial_uncertainties_px.append(float(minimum_uncertainty_px))
            fit_success.append(False)
            continue

        window_positions = window_positions[finite_mask]
        window_values = window_values[finite_mask]
        radial_offsets = np.dot(
            window_positions - beam_center_px,
            radial_unit,
        )

        offset = float(np.nanmin(window_values))
        amplitude = float(np.nanmax(window_values) - offset)
        if not np.isfinite(amplitude) or amplitude <= 0:
            fitted_peak_positions.append(seed_position)
            radial_uncertainties_px.append(float(minimum_uncertainty_px))
            fit_success.append(False)
            continue

        sigma_guess = max(refinement_size / 4, 1.0)
        try:
            popt, pcov = curve_fit(
                _radial_gaussian_model,
                radial_offsets,
                window_values,
                p0=[float(radial_norm), sigma_guess, amplitude, offset],
                bounds=(
                    [
                        max(radial_norm - refinement_size, 0.0),
                        0.25,
                        0.0,
                        -np.inf,
                    ],
                    [
                        radial_norm + refinement_size,
                        max(refinement_size, 1.0),
                        np.inf,
                        np.inf,
                    ],
                ),
                maxfev=10000,
            )
            refined_radius = float(popt[0])
            if pcov.size == 0 or not np.isfinite(pcov[0, 0]):
                sigma_r_px = float(minimum_uncertainty_px)
            else:
                sigma_r_px = float(
                    max(np.sqrt(max(pcov[0, 0], 0.0)), minimum_uncertainty_px)
                )
            fitted_peak_positions.append(
                beam_center_px + radial_unit * refined_radius
            )
            radial_uncertainties_px.append(sigma_r_px)
            fit_success.append(True)
        except (RuntimeError, ValueError):
            fitted_peak_positions.append(seed_position)
            radial_uncertainties_px.append(float(minimum_uncertainty_px))
            fit_success.append(False)

    return (
        np.asarray(fitted_peak_positions, dtype=float),
        np.asarray(radial_uncertainties_px, dtype=float),
        np.asarray(fit_success, dtype=bool),
    )


def _aggregate_ring_observations(
        peak_positions,
        beam_center_px,
        ring_indices,
        peak_position_uncertainty_px,
        minimum_uncertainty_px=0.5):
    """Aggregate repeated ring samples into one effective observation."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    beam_center_px = np.asarray(beam_center_px, dtype=float)
    ring_indices = np.asarray(ring_indices, dtype=int)
    peak_position_uncertainty_px = np.asarray(
        peak_position_uncertainty_px,
        dtype=float,
    )

    radial_distances_px = np.linalg.norm(
        peak_positions - beam_center_px,
        axis=1,
    )
    aggregated_peak_positions = []
    aggregated_ring_indices = []
    aggregated_radial_uncertainty_px = []
    aggregated_radial_distances_px = []

    for ring_index in np.unique(ring_indices):
        ring_mask = ring_indices == ring_index
        ring_positions = peak_positions[ring_mask]
        ring_radii = radial_distances_px[ring_mask]
        ring_sigma = np.maximum(
            peak_position_uncertainty_px[ring_mask],
            float(minimum_uncertainty_px),
        )
        ring_weights = 1.0 / np.maximum(ring_sigma**2, 1e-12)

        weighted_radius = float(
            np.sum(ring_weights * ring_radii) / np.sum(ring_weights)
        )
        weighted_position = np.sum(
            ring_positions * ring_weights[:, np.newaxis],
            axis=0,
        ) / np.sum(ring_weights)

        if ring_radii.size >= 2:
            scatter_px = float(np.std(ring_radii, ddof=1))
            standard_error_px = float(np.sqrt(1.0 / np.sum(ring_weights)))
            effective_uncertainty_px = max(
                standard_error_px,
                scatter_px / np.sqrt(ring_radii.size),
                float(minimum_uncertainty_px),
            )
        else:
            effective_uncertainty_px = max(
                float(ring_sigma[0]),
                float(minimum_uncertainty_px),
            )

        radial_vector = weighted_position - beam_center_px
        radial_norm = np.linalg.norm(radial_vector)
        if radial_norm > 0:
            weighted_position = (
                beam_center_px + radial_vector / radial_norm * weighted_radius
            )

        aggregated_peak_positions.append(weighted_position)
        aggregated_ring_indices.append(int(ring_index))
        aggregated_radial_uncertainty_px.append(effective_uncertainty_px)
        aggregated_radial_distances_px.append(weighted_radius)

    return {
        "radial_distances_px": radial_distances_px,
        "aggregated_peak_positions": np.asarray(
            aggregated_peak_positions,
            dtype=float,
        ),
        "aggregated_ring_indices": np.asarray(
            aggregated_ring_indices,
            dtype=int,
        ),
        "aggregated_radial_uncertainty_px": np.asarray(
            aggregated_radial_uncertainty_px,
            dtype=float,
        ),
        "aggregated_radial_distances_px": np.asarray(
            aggregated_radial_distances_px,
            dtype=float,
        ),
    }


def _refine_sdd_from_ring_observations(
        peak_positions,
        beam_center_px,
        ring_indices,
        wavelength_nm,
        pixel_size_um,
        pitch_nm,
        sdd_initial_cm,
        peak_position_uncertainty_px,
        minimum_uncertainty_px=0.5):
    """Refine SDD from ring observations aggregated by ring index."""
    aggregated = _aggregate_ring_observations(
        peak_positions=peak_positions,
        beam_center_px=beam_center_px,
        ring_indices=ring_indices,
        peak_position_uncertainty_px=peak_position_uncertainty_px,
        minimum_uncertainty_px=minimum_uncertainty_px,
    )
    refined_sdd_cm, standard_uncertainty_cm = _refine_sdd_with_fixed_orders(
        peak_positions=aggregated["aggregated_peak_positions"],
        beam_center_px=beam_center_px,
        orders=aggregated["aggregated_ring_indices"],
        wavelength_nm=wavelength_nm,
        pixel_size_um=pixel_size_um,
        pitch_nm=pitch_nm,
        sdd_initial_cm=sdd_initial_cm,
        peak_position_uncertainty_px=(
            aggregated["aggregated_radial_uncertainty_px"]
        ),
    )

    return refined_sdd_cm, standard_uncertainty_cm, aggregated


def _refine_sdd_with_fixed_orders(
        peak_positions,
        beam_center_px,
        orders,
        wavelength_nm,
        pixel_size_um,
        pitch_nm,
    sdd_initial_cm,
    peak_position_uncertainty_px):
    """Refine SDD for fixed beam center and diffraction orders."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    beam_center_px = np.asarray(beam_center_px, dtype=float)
    orders = np.asarray(orders, dtype=float)

    radial_distance_cm = np.linalg.norm(
        peak_positions - beam_center_px,
        axis=1,
    ) * pixel_size_um * 1e-4
    q_spacing = 2 * np.pi / pitch_nm
    q_target = np.abs(orders) * q_spacing

    sigma_r = np.asarray(peak_position_uncertainty_px, dtype=float)
    sigma_r = sigma_r * pixel_size_um * 1e-4
    theta_initial = np.arctan2(radial_distance_cm, float(sdd_initial_cm))
    dq_dr = (
        (2 * np.pi / wavelength_nm)
        * np.cos(theta_initial / 2)
        * float(sdd_initial_cm)
        / (radial_distance_cm**2 + float(sdd_initial_cm)**2)
    )
    sigma_q = np.maximum(np.abs(dq_dr) * sigma_r, 1e-12)

    def q_model(radial_distance, sdd_cm):
        theta = np.arctan2(radial_distance, sdd_cm)
        return 4 * np.pi * np.sin(theta / 2) / wavelength_nm

    popt, pcov = curve_fit(
        q_model,
        radial_distance_cm,
        q_target,
        p0=[float(sdd_initial_cm)],
        bounds=(0, np.inf),
        sigma=sigma_q,
        absolute_sigma=True,
    )
    sdd_refined_cm = float(popt[0])
    if pcov.size == 0 or not np.isfinite(pcov[0, 0]):
        standard_uncertainty_cm = np.nan
    else:
        standard_uncertainty_cm = float(np.sqrt(pcov[0, 0]))

    return sdd_refined_cm, standard_uncertainty_cm


def _refine_beam_center_and_sdd_with_fixed_orders(
        peak_positions,
        beam_center_initial_px,
        orders,
        wavelength_nm,
        pixel_size_um,
        pitch_nm,
        sdd_initial_cm,
        peak_position_uncertainty_px,
        beam_center_bounds_px=None,
        sdd_bounds_cm=None):
    """Refine beam center and SDD for fixed diffraction orders."""
    peak_positions = np.asarray(peak_positions, dtype=float)
    beam_center_initial_px = np.asarray(beam_center_initial_px, dtype=float)
    orders = np.asarray(orders, dtype=float)
    sigma_r_px = np.asarray(peak_position_uncertainty_px, dtype=float)

    if peak_positions.ndim != 2 or peak_positions.shape[1] != 2:
        raise ValueError(
            "peak_positions must be a 2D array with shape (n_peaks, 2)."
        )
    if beam_center_initial_px.shape != (2,):
        raise ValueError(
            "beam_center_initial_px must contain [row, column]."
        )
    if peak_positions.shape[0] != orders.shape[0]:
        raise ValueError(
            "peak_positions and orders must contain the same number of "
            "observations."
        )
    if sigma_r_px.shape[0] != peak_positions.shape[0]:
        raise ValueError(
            "peak_position_uncertainty_px must match peak_positions."
        )

    q_spacing = 2 * np.pi / pitch_nm
    q_target = np.abs(orders) * q_spacing
    sigma_r_cm = np.maximum(sigma_r_px, 1e-12) * pixel_size_um * 1e-4

    def q_model(coordinates, row_center, col_center, sdd_cm):
        beam_center_px = np.array([row_center, col_center], dtype=float)
        radial_distance_cm = np.linalg.norm(
            coordinates - beam_center_px,
            axis=1,
        ) * pixel_size_um * 1e-4
        theta = np.arctan2(radial_distance_cm, sdd_cm)
        return 4 * np.pi * np.sin(theta / 2) / wavelength_nm

    radial_distance_initial_cm = np.linalg.norm(
        peak_positions - beam_center_initial_px,
        axis=1,
    ) * pixel_size_um * 1e-4
    theta_initial = np.arctan2(radial_distance_initial_cm, float(sdd_initial_cm))
    dq_dr = (
        (2 * np.pi / wavelength_nm)
        * np.cos(theta_initial / 2)
        * float(sdd_initial_cm)
        / (radial_distance_initial_cm**2 + float(sdd_initial_cm)**2)
    )
    sigma_q = np.maximum(np.abs(dq_dr) * sigma_r_cm, 1e-12)

    if beam_center_bounds_px is None:
        lower_bounds = [-np.inf, -np.inf]
        upper_bounds = [np.inf, np.inf]
    else:
        lower_bounds = [
            float(beam_center_bounds_px[0][0]),
            float(beam_center_bounds_px[1][0]),
        ]
        upper_bounds = [
            float(beam_center_bounds_px[0][1]),
            float(beam_center_bounds_px[1][1]),
        ]
    if sdd_bounds_cm is None:
        sdd_lower_cm, sdd_upper_cm = 0.0, np.inf
    else:
        sdd_lower_cm = float(sdd_bounds_cm[0])
        sdd_upper_cm = float(sdd_bounds_cm[1])

    popt, pcov = curve_fit(
        q_model,
        peak_positions,
        q_target,
        p0=[
            float(beam_center_initial_px[0]),
            float(beam_center_initial_px[1]),
            float(sdd_initial_cm),
        ],
        bounds=(
            lower_bounds + [sdd_lower_cm],
            upper_bounds + [sdd_upper_cm],
        ),
        sigma=sigma_q,
        absolute_sigma=True,
    )
    refined_beam_center_px = np.asarray(popt[:2], dtype=float)
    refined_sdd_cm = float(popt[2])
    if pcov.size == 0 or not np.all(np.isfinite(np.diag(pcov))):
        parameter_uncertainty = np.full(3, np.nan, dtype=float)
    else:
        parameter_uncertainty = np.sqrt(np.diag(pcov))

    return refined_beam_center_px, refined_sdd_cm, parameter_uncertainty


def _build_search_values(center_value, half_width, points, lower_bound=None,
                         upper_bound=None):
    """Create a bounded linear search grid around a center value."""
    start = center_value - half_width
    stop = center_value + half_width
    if lower_bound is not None:
        start = max(start, lower_bound)
    if upper_bound is not None:
        stop = min(stop, upper_bound)
    if points <= 1 or np.isclose(start, stop):
        return np.array([(start + stop) / 2])
    return np.linspace(start, stop, int(points))


def _ring_radius_px_from_geometry(
        ring_index,
        sdd_cm,
        pitch_nm,
        wavelength_nm,
        pixel_size_um):
    """Predict detector radius in pixels for a diffraction ring index."""
    q_spacing = 2 * np.pi / pitch_nm
    q_value = float(ring_index) * q_spacing
    argument = q_value * wavelength_nm / (4 * np.pi)
    if argument <= 0 or argument >= 1:
        return np.nan
    theta = 2 * np.arcsin(argument)
    radial_distance_cm = float(sdd_cm) * np.tan(theta)
    return radial_distance_cm * 1e4 / float(pixel_size_um)


def _detector_polar_coordinates(image_shape, beam_center_px):
    """Return detector radius and azimuth arrays for a beam center."""
    row_grid, col_grid = np.indices(image_shape, dtype=float)
    delta_row = row_grid - float(beam_center_px[0])
    delta_col = col_grid - float(beam_center_px[1])
    radius_px = np.sqrt(delta_row**2 + delta_col**2)
    azimuth_deg = np.rad2deg(np.arctan2(delta_row, delta_col))
    azimuth_deg = np.mod(azimuth_deg, 360.0)
    return radius_px, azimuth_deg


def _extract_sector_radial_profile(
        image,
        beam_center_px,
        sector_center_deg,
        sector_width_deg,
        radial_bin_size_px=1.0,
        exclude_within_radius_px=None):
    """Average image intensity radially within an azimuthal sector."""
    image = np.asarray(image, dtype=float)
    radius_px, azimuth_deg = _detector_polar_coordinates(
        image.shape,
        beam_center_px,
    )
    angle_delta = ((azimuth_deg - sector_center_deg + 180.0) % 360.0) - 180.0
    sector_mask = np.abs(angle_delta) <= (float(sector_width_deg) / 2.0)
    if exclude_within_radius_px is not None:
        sector_mask &= radius_px >= float(exclude_within_radius_px)

    finite_mask = np.isfinite(image)
    sector_mask &= finite_mask
    if not np.any(sector_mask):
        return np.array([], dtype=float), np.array([], dtype=float)

    radii = radius_px[sector_mask]
    intensities = image[sector_mask]
    radial_bin_size_px = max(float(radial_bin_size_px), 0.25)
    radial_bins = np.floor(radii / radial_bin_size_px).astype(int)
    if radial_bins.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    radial_sum = np.bincount(radial_bins, weights=intensities)
    radial_count = np.bincount(radial_bins)
    valid_bins = radial_count > 0
    radial_centers_px = (
        np.nonzero(valid_bins)[0].astype(float) + 0.5
    ) * radial_bin_size_px
    radial_profile = radial_sum[valid_bins] / radial_count[valid_bins]
    return radial_centers_px, radial_profile


def _fit_sector_ring_radius(
        radial_centers_px,
        radial_profile,
        expected_radius_px,
        radial_window_px,
        minimum_uncertainty_px=0.5):
    """Fit a 1D Gaussian to a sector radial profile near an expected ring."""
    radial_centers_px = np.asarray(radial_centers_px, dtype=float)
    radial_profile = np.asarray(radial_profile, dtype=float)
    if radial_centers_px.size < 4 or radial_profile.size < 4:
        return None

    fit_mask = np.abs(radial_centers_px - float(expected_radius_px)) <= float(
        radial_window_px
    )
    fit_mask &= np.isfinite(radial_profile)
    if np.count_nonzero(fit_mask) < 4:
        return None

    x_fit = radial_centers_px[fit_mask]
    y_fit = radial_profile[fit_mask]
    if np.nanmax(y_fit) <= np.nanmin(y_fit):
        return None

    sigma_guess = max(float(radial_window_px) / 3.0, 1.0)
    p0 = [
        float(expected_radius_px),
        sigma_guess,
        float(np.nanmax(y_fit) - np.nanmin(y_fit)),
        float(np.nanmin(y_fit)),
    ]
    try:
        popt, pcov = curve_fit(
            gaussian,
            x_fit,
            y_fit,
            p0=p0,
            bounds=(
                [x_fit.min(), 0.25, 0.0, -np.inf],
                [
                    x_fit.max(),
                    max(float(radial_window_px), 1.0),
                    np.inf,
                    np.inf,
                ],
            ),
        )
    except (RuntimeError, ValueError):
        return None

    fitted_radius_px = float(popt[0])
    if not np.isfinite(fitted_radius_px):
        return None
    if pcov.size == 0 or not np.isfinite(pcov[0, 0]):
        radius_uncertainty_px = float(minimum_uncertainty_px)
    else:
        radius_uncertainty_px = float(
            max(np.sqrt(pcov[0, 0]), float(minimum_uncertainty_px))
        )

    return {
        "radius_px": fitted_radius_px,
        "radius_uncertainty_px": radius_uncertainty_px,
        "fit_parameters": np.asarray(popt, dtype=float),
    }


def _plot_ring_sector_fit_diagnostics(
        image,
        beam_center_px,
        ring_indices,
        sector_angles_deg,
        fitted_peak_positions_px,
        pitch_nm,
        wavelength_nm,
        pixel_size_um,
        sdd_cm,
        radial_window_px,
        title=None):
    """Plot sector-fit search windows and fitted ring positions."""
    image = np.asarray(image, dtype=float)
    beam_center_px = np.asarray(beam_center_px, dtype=float)
    ring_indices = np.asarray(ring_indices, dtype=int)
    sector_angles_deg = np.asarray(sector_angles_deg, dtype=float)
    fitted_peak_positions_px = np.asarray(fitted_peak_positions_px, dtype=float)

    finite_positive = image[np.isfinite(image) & (image > 0)]
    if finite_positive.size == 0:
        plotting_image = np.where(np.isfinite(image), image, 1.0)
        plotting_image = np.maximum(plotting_image, 1.0)
        norm = None
    else:
        plotting_image = np.where(np.isfinite(image), image, np.nan)
        plotting_image = np.maximum(plotting_image, np.min(finite_positive))
        norm = LogNorm(
            vmin=float(np.min(finite_positive)),
            vmax=float(np.max(finite_positive)),
        )

    fig, ax = plt.subplots()
    image_artist = ax.imshow(
        plotting_image,
        origin='upper',
        cmap='viridis',
        norm=norm,
    )
    fig.colorbar(image_artist, ax=ax, label='Intensity (log scale)')

    unique_ring_indices = np.unique(ring_indices)
    unique_sector_angles_deg = np.unique(sector_angles_deg)
    for ring_index in unique_ring_indices:
        expected_radius_px = _ring_radius_px_from_geometry(
            ring_index=ring_index,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
        )
        if not np.isfinite(expected_radius_px):
            continue
        inner_radius_px = max(expected_radius_px - float(radial_window_px), 0.0)
        outer_radius_px = expected_radius_px + float(radial_window_px)
        for radius_px, linestyle in (
                (inner_radius_px, '--'),
                (outer_radius_px, '--')):
            ring_patch = plt.Circle(
                (beam_center_px[1], beam_center_px[0]),
                radius_px,
                fill=False,
                color='white',
                linewidth=0.8,
                linestyle=linestyle,
                alpha=0.5,
            )
            ax.add_patch(ring_patch)

    for sector_angle_deg in unique_sector_angles_deg:
        angle_rad = np.deg2rad(sector_angle_deg)
        direction = np.array([
            np.sin(angle_rad),
            np.cos(angle_rad),
        ])
        max_radius_px = np.max([
            np.linalg.norm([beam_center_px[0], beam_center_px[1]]),
            np.linalg.norm([
                beam_center_px[0],
                image.shape[1] - beam_center_px[1],
            ]),
            np.linalg.norm([
                image.shape[0] - beam_center_px[0],
                beam_center_px[1],
            ]),
            np.linalg.norm([
                image.shape[0] - beam_center_px[0],
                image.shape[1] - beam_center_px[1],
            ]),
        ])
        endpoint = beam_center_px + max_radius_px * direction
        ax.plot(
            [beam_center_px[1], endpoint[1]],
            [beam_center_px[0], endpoint[0]],
            color='white',
            linewidth=0.4,
            alpha=0.15,
        )

    ax.scatter(
        fitted_peak_positions_px[:, 1],
        fitted_peak_positions_px[:, 0],
        s=18,
        c='red',
        marker='o',
        label='Fitted sector peak radii',
    )
    ax.scatter(
        [beam_center_px[1]],
        [beam_center_px[0]],
        s=60,
        c='cyan',
        marker='x',
        linewidths=2.0,
        label='Beam center',
    )
    ax.set_xlabel('Column (px)')
    ax.set_ylabel('Row (px)')
    ax.set_title(title or 'Ring sector fit diagnostics')
    ax.legend(loc='upper right')
    fig.tight_layout()

    return fig


def _collect_ring_sector_observations(
        image,
        beam_center_px,
        sdd_cm,
        pitch_nm,
        wavelength_nm,
        pixel_size_um,
        sector_step_deg=10.0,
        sector_width_deg=10.0,
        ring_indices=None,
        radial_window_px=8.0,
        radial_bin_size_px=1.0,
        exclude_within_beamstop_radius_px=None,
        minimum_ring_uncertainty_px=0.5,
        progress_bar=None):
    """Collect ring observations from sector radial averages."""
    image = np.asarray(image, dtype=float)
    beam_center_px = np.asarray(beam_center_px, dtype=float)
    if ring_indices is None:
        max_radius_px = np.max([
            np.linalg.norm([beam_center_px[0], beam_center_px[1]]),
            np.linalg.norm([
                beam_center_px[0],
                image.shape[1] - beam_center_px[1],
            ]),
            np.linalg.norm([
                image.shape[0] - beam_center_px[0],
                beam_center_px[1],
            ]),
            np.linalg.norm([
                image.shape[0] - beam_center_px[0],
                image.shape[1] - beam_center_px[1],
            ]),
        ])
        inferred_ring_indices = []
        ring_index = 1
        while True:
            radius_px = _ring_radius_px_from_geometry(
                ring_index=ring_index,
                sdd_cm=sdd_cm,
                pitch_nm=pitch_nm,
                wavelength_nm=wavelength_nm,
                pixel_size_um=pixel_size_um,
            )
            if not np.isfinite(radius_px) or radius_px > max_radius_px:
                break
            inferred_ring_indices.append(ring_index)
            ring_index += 1
        ring_indices = inferred_ring_indices

    sector_angles_deg = np.arange(0.0, 360.0, float(sector_step_deg))
    fitted_positions = []
    fitted_ring_indices = []
    fitted_uncertainties_px = []
    fitted_sector_angles_deg = []
    fitted_radii_px = []

    for sector_angle_deg in sector_angles_deg:
        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"angle={sector_angle_deg:.1f} deg",
                refresh=False,
            )
        radial_centers_px, radial_profile = _extract_sector_radial_profile(
            image=image,
            beam_center_px=beam_center_px,
            sector_center_deg=sector_angle_deg,
            sector_width_deg=sector_width_deg,
            radial_bin_size_px=radial_bin_size_px,
            exclude_within_radius_px=exclude_within_beamstop_radius_px,
        )
        if progress_bar is not None:
            progress_bar.update(1)
        if radial_centers_px.size < 4:
            continue

        for ring_index in ring_indices:
            expected_radius_px = _ring_radius_px_from_geometry(
                ring_index=ring_index,
                sdd_cm=sdd_cm,
                pitch_nm=pitch_nm,
                wavelength_nm=wavelength_nm,
                pixel_size_um=pixel_size_um,
            )
            if not np.isfinite(expected_radius_px):
                continue
            fit_result = _fit_sector_ring_radius(
                radial_centers_px=radial_centers_px,
                radial_profile=radial_profile,
                expected_radius_px=expected_radius_px,
                radial_window_px=radial_window_px,
                minimum_uncertainty_px=minimum_ring_uncertainty_px,
            )
            if fit_result is None:
                continue

            angle_rad = np.deg2rad(sector_angle_deg)
            direction = np.array([
                np.sin(angle_rad),
                np.cos(angle_rad),
            ])
            fitted_position = (
                beam_center_px + fit_result["radius_px"] * direction
            )
            fitted_positions.append(fitted_position)
            fitted_ring_indices.append(int(ring_index))
            fitted_uncertainties_px.append(fit_result["radius_uncertainty_px"])
            fitted_sector_angles_deg.append(float(sector_angle_deg))
            fitted_radii_px.append(fit_result["radius_px"])

    if not fitted_positions:
        return {
            "peak_positions": np.empty((0, 2), dtype=float),
            "ring_indices": np.empty((0,), dtype=int),
            "peak_position_uncertainty_px": np.empty((0,), dtype=float),
            "sector_angles_deg": np.empty((0,), dtype=float),
            "radii_px": np.empty((0,), dtype=float),
        }

    return {
        "peak_positions": np.asarray(fitted_positions, dtype=float),
        "ring_indices": np.asarray(fitted_ring_indices, dtype=int),
        "peak_position_uncertainty_px": np.asarray(
            fitted_uncertainties_px,
            dtype=float,
        ),
        "sector_angles_deg": np.asarray(fitted_sector_angles_deg, dtype=float),
        "radii_px": np.asarray(fitted_radii_px, dtype=float),
    }


def _score_ring_sector_candidate(
        image,
        beam_center_px,
        sdd_cm,
        pitch_nm,
        wavelength_nm,
        pixel_size_um,
        sector_step_deg=10.0,
        sector_width_deg=10.0,
        ring_indices=None,
        radial_window_px=8.0,
        radial_bin_size_px=1.0,
        exclude_within_beamstop_radius_px=None,
        minimum_ring_uncertainty_px=0.5,
        progress_bar=None):
    """Score a beam-center/SDD candidate using sector-fitted ring radii."""
    observations = _collect_ring_sector_observations(
        image=image,
        beam_center_px=beam_center_px,
        sdd_cm=sdd_cm,
        pitch_nm=pitch_nm,
        wavelength_nm=wavelength_nm,
        pixel_size_um=pixel_size_um,
        sector_step_deg=sector_step_deg,
        sector_width_deg=sector_width_deg,
        ring_indices=ring_indices,
        radial_window_px=radial_window_px,
        radial_bin_size_px=radial_bin_size_px,
        exclude_within_beamstop_radius_px=exclude_within_beamstop_radius_px,
        minimum_ring_uncertainty_px=minimum_ring_uncertainty_px,
        progress_bar=progress_bar,
    )
    if observations["peak_positions"].shape[0] < 2:
        return np.inf, observations

    observed_radii_px = np.linalg.norm(
        observations["peak_positions"]
        - np.asarray(beam_center_px, dtype=float),
        axis=1,
    )
    expected_radii_px = np.asarray([
        _ring_radius_px_from_geometry(
            ring_index=ring_index,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
        )
        for ring_index in observations["ring_indices"]
    ], dtype=float)
    sigma_px = np.maximum(
        observations["peak_position_uncertainty_px"],
        float(minimum_ring_uncertainty_px),
    )
    residual = (observed_radii_px - expected_radii_px) / sigma_px
    score = float(np.mean(residual**2))

    expected_total = max(
        len(np.arange(0.0, 360.0, float(sector_step_deg)))
        * max(len(ring_indices or []), 1),
        1,
    )
    missing_fraction = 1.0 - (
        observations["peak_positions"].shape[0] / expected_total
    )
    score += missing_fraction
    return score, observations


def estimate_sample_detector_distance(
    image,
        peak_positions,
        pitch_nm,
        wavelength_nm,
        pixel_size_um,
        beam_center_guess_px=None,
        beam_center_search_radius_px=None,
        sdd_search_range_cm=(1.0, 1000.0),
        coarse_grid_points=31,
        fine_grid_points=31,
        return_details=False):
    """
    Estimate sample-to-detector distance from diffraction peak positions.

    Parameters
    ----------
    image : ndarray
        Source detector image used to estimate local peak-position
        uncertainties from 2D Gaussian fits around the provided peaks.
    peak_positions : ndarray
        Peak coordinates with shape (n, 2) where each row is
        [row_position, column_position].
    pitch_nm : float
        Sample pitch in nanometers.
    wavelength_nm : float
        X-ray wavelength in nanometers.
    pixel_size_um : float
        Detector pixel size in microns.
    beam_center_guess_px : tuple | list | ndarray, optional
        Initial guess for the beam center as [row, column] in pixels.
        If omitted, the mean of the peak positions is used.
    beam_center_search_radius_px : float | tuple, optional
        Search half-width in pixels around the beam center guess.
        If a scalar is provided, the same radius is used for row and
        column. If omitted, the search radius is based on the peak span.
    sdd_search_range_cm : tuple, optional
        Inclusive search range for sample-to-detector distance in cm as
        (minimum, maximum).
    coarse_grid_points : int, optional
        Number of grid points per dimension for the coarse search.
    fine_grid_points : int, optional
        Number of grid points per dimension for the fine search.
    return_details : bool, optional
        If set to True, also return a dictionary with the fitted beam
        center, assigned diffraction orders, fitted q values, and fit
        score.

    Returns
    -------
    sdd_cm : float
        Estimated sample-to-detector distance in cm.
    uncertainty_cm : float
        Estimated standard uncertainty in sample-to-detector distance in
        cm from a post-order nonlinear refit using image-based radial
        peak-position uncertainties.
    details : dict, optional
        Returned only when return_details is True. Contains
        ``beam_center_px``, ``orders``, ``q_values_nm_inverse``,
        ``score``, ``grid_sdd_cm``, ``grid_uncertainty_cm``,
        ``standard_uncertainty_cm``, ``peak_position_uncertainty_px``,
        and ``fitted_peak_positions_px``.

    Examples
    --------
    >>> sdd_cm, uncertainty_cm = estimate_sample_detector_distance(
    ...     image=image,
    ...     peak_positions=peaks,
    ...     pitch_nm=80.0,
    ...     wavelength_nm=0.1,
    ...     pixel_size_um=75.0,
    ... )
    >>> sdd_cm, uncertainty_cm, details = estimate_sample_detector_distance(
    ...     image=image,
    ...     peak_positions=peaks,
    ...     pitch_nm=80.0,
    ...     wavelength_nm=0.1,
    ...     pixel_size_um=75.0,
    ...     return_details=True,
    ... )
    >>> details["beam_center_px"]
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")

    peak_positions = np.asarray(peak_positions, dtype=float)
    if peak_positions.ndim != 2 or peak_positions.shape[1] != 2:
        raise ValueError(
            "peak_positions must be a 2D array with shape (n_peaks, 2)."
        )
    if peak_positions.shape[0] < 2:
        raise ValueError("At least two peak positions are required.")
    if pitch_nm <= 0 or wavelength_nm <= 0 or pixel_size_um <= 0:
        raise ValueError(
            "pitch_nm, wavelength_nm, and pixel_size_um must be positive."
        )

    sdd_min_cm, sdd_max_cm = map(float, sdd_search_range_cm)
    if sdd_min_cm <= 0 or sdd_max_cm <= sdd_min_cm:
        raise ValueError(
            "sdd_search_range_cm must contain positive increasing values."
        )

    if beam_center_guess_px is None:
        beam_center_guess_px = np.mean(peak_positions, axis=0)
    else:
        beam_center_guess_px = np.asarray(beam_center_guess_px, dtype=float)
        if beam_center_guess_px.shape != (2,):
            raise ValueError(
                "beam_center_guess_px must contain [row, column]."
            )

    peak_span = np.ptp(peak_positions, axis=0)
    default_radius = max(float(np.max(peak_span)) / 2, 2.0)
    if beam_center_search_radius_px is None:
        beam_center_search_radius_px = (default_radius, default_radius)
    elif np.isscalar(beam_center_search_radius_px):
        beam_center_search_radius_px = (
            float(beam_center_search_radius_px),
            float(beam_center_search_radius_px),
        )
    else:
        beam_center_search_radius_px = tuple(beam_center_search_radius_px)
        if len(beam_center_search_radius_px) != 2:
            raise ValueError(
                "beam_center_search_radius_px must be a scalar or length 2."
            )

    row_bounds = (
        float(np.min(peak_positions[:, 0]) - peak_span[0]),
        float(np.max(peak_positions[:, 0]) + peak_span[0]),
    )
    col_bounds = (
        float(np.min(peak_positions[:, 1]) - peak_span[1]),
        float(np.max(peak_positions[:, 1]) + peak_span[1]),
    )
    diffraction_direction = _fit_diffraction_direction(peak_positions)

    best = None
    search_specs = [
        (
            coarse_grid_points,
            beam_center_search_radius_px,
            (sdd_min_cm, sdd_max_cm),
        ),
        (
            fine_grid_points,
            (
                max(beam_center_search_radius_px[0] / 4, 0.5),
                max(beam_center_search_radius_px[1] / 4, 0.5),
            ),
            None,
        ),
    ]

    for grid_points, center_radius, sdd_bounds in search_specs:
        if best is None:
            center_seed = beam_center_guess_px
            sdd_seed = (sdd_min_cm + sdd_max_cm) / 2
            sdd_half_width = (sdd_max_cm - sdd_min_cm) / 2
        else:
            center_seed = best["beam_center_px"]
            sdd_seed = best["sdd_cm"]
            sdd_half_width = max(best["sdd_half_width_cm"] / 4, 0.25)

        if sdd_bounds is None:
            sdd_bounds = (
                max(sdd_min_cm, sdd_seed - sdd_half_width),
                min(sdd_max_cm, sdd_seed + sdd_half_width),
            )

        row_values = _build_search_values(
            center_value=center_seed[0],
            half_width=center_radius[0],
            points=grid_points,
            lower_bound=row_bounds[0],
            upper_bound=row_bounds[1],
        )
        col_values = _build_search_values(
            center_value=center_seed[1],
            half_width=center_radius[1],
            points=grid_points,
            lower_bound=col_bounds[0],
            upper_bound=col_bounds[1],
        )
        sdd_values = np.linspace(
            sdd_bounds[0],
            sdd_bounds[1],
            int(grid_points),
        )

        for row_center in row_values:
            for col_center in col_values:
                beam_center_px = np.array(
                    [row_center, col_center],
                    dtype=float,
                )
                for sdd_cm in sdd_values:
                    score, orders, q = _score_sdd_candidate(
                        peak_positions=peak_positions,
                        beam_center_px=beam_center_px,
                        sdd_cm=sdd_cm,
                        pixel_size_um=pixel_size_um,
                        wavelength_nm=wavelength_nm,
                        pitch_nm=pitch_nm,
                        diffraction_direction=diffraction_direction,
                    )
                    if best is None or score < best["score"]:
                        best = {
                            "score": score,
                            "beam_center_px": beam_center_px,
                            "sdd_cm": float(sdd_cm),
                            "orders": orders,
                            "q": q,
                            "diffraction_direction": diffraction_direction,
                            "sdd_half_width_cm": max(
                                (sdd_bounds[1] - sdd_bounds[0]) / 2,
                                0.25,
                            ),
                        }

    sdd_probe_half_width = max(best["sdd_half_width_cm"] / 4, 0.25)
    sdd_probe_values = _build_search_values(
        center_value=best["sdd_cm"],
        half_width=sdd_probe_half_width,
        points=max(fine_grid_points, 11),
        lower_bound=sdd_min_cm,
        upper_bound=sdd_max_cm,
    )
    sdd_scores = []
    for sdd_cm in sdd_probe_values:
        score, _, _ = _score_sdd_candidate(
            peak_positions=peak_positions,
            beam_center_px=best["beam_center_px"],
            sdd_cm=sdd_cm,
            pixel_size_um=pixel_size_um,
            wavelength_nm=wavelength_nm,
            pitch_nm=pitch_nm,
            diffraction_direction=diffraction_direction,
        )
        sdd_scores.append(score)
    sdd_scores = np.asarray(sdd_scores)
    min_score = float(np.min(sdd_scores))
    threshold = min_score + max(min_score * 0.1, 1e-12)
    within_threshold = sdd_probe_values[sdd_scores <= threshold]
    if within_threshold.size >= 2:
        grid_uncertainty_cm = float(
            (within_threshold[-1] - within_threshold[0]) / 2
        )
    else:
        step_size = (
            np.min(np.diff(sdd_probe_values)) / 2
            if sdd_probe_values.size > 1 else 0.25
        )
        grid_uncertainty_cm = float(
            max(
                step_size,
                0.01,
            )
        )

    fitted_peak_positions, peak_position_uncertainty_px, _ = (
        _estimate_peak_radial_uncertainty_from_image(
            image=image,
            peak_positions=peak_positions,
            beam_center_px=best["beam_center_px"],
        )
    )
    refined_sdd_cm, standard_uncertainty_cm = _refine_sdd_with_fixed_orders(
        peak_positions=fitted_peak_positions,
        beam_center_px=best["beam_center_px"],
        orders=best["orders"],
        wavelength_nm=wavelength_nm,
        pixel_size_um=pixel_size_um,
        pitch_nm=pitch_nm,
        sdd_initial_cm=best["sdd_cm"],
        peak_position_uncertainty_px=peak_position_uncertainty_px,
    )
    uncertainty_cm = standard_uncertainty_cm

    if not return_details:
        return refined_sdd_cm, uncertainty_cm

    details = {
        "beam_center_px": tuple(best["beam_center_px"]),
        "orders": best["orders"].copy(),
        "q_values_nm_inverse": best["q"].copy(),
        "score": best["score"],
        "diffraction_direction": best["diffraction_direction"].copy(),
        "grid_sdd_cm": best["sdd_cm"],
        "grid_uncertainty_cm": grid_uncertainty_cm,
        "standard_uncertainty_cm": standard_uncertainty_cm,
        "peak_position_uncertainty_px": peak_position_uncertainty_px.copy(),
        "fitted_peak_positions_px": fitted_peak_positions.copy(),
    }

    return refined_sdd_cm, uncertainty_cm, details


def estimate_sample_detector_distance_from_rings(
    image,
        peak_positions,
        pitch_nm,
        wavelength_nm,
        pixel_size_um,
        beam_center_guess_px=None,
        exclude_within_beamstop_radius_px=None,
        beam_center_search_radius_px=None,
        sdd_search_range_cm=(1.0, 1000.0),
        coarse_grid_points=31,
        fine_grid_points=31,
        minimum_ring_uncertainty_px=0.5,
        return_details=False):
    """
    Estimate sample-to-detector distance from points sampled on rings.

    Parameters
    ----------
    image : ndarray
        Source detector image used to estimate local peak-position
        uncertainties from 2D Gaussian fits around the supplied ring
        samples.
    peak_positions : ndarray
        Ring sample coordinates with shape (n, 2) where each row is
        [row_position, column_position].
    pitch_nm : float
        Sample pitch in nanometers.
    wavelength_nm : float
        X-ray wavelength in nanometers.
    pixel_size_um : float
        Detector pixel size in microns.
    beam_center_guess_px : tuple | list | ndarray, optional
        Initial guess for the beam center as [row, column] in pixels.
        If omitted, the mean of the supplied positions is used.
    exclude_within_beamstop_radius_px : float, optional
        Exclude supplied ring points whose distance from the beam center
        guess is smaller than this radius in pixels. This can be used to
        ignore points obscured by the beamstop. If omitted, no exclusion
        is applied.
    beam_center_search_radius_px : float | tuple, optional
        Search half-width in pixels around the beam center guess.
        If a scalar is provided, the same radius is used for row and
        column. If omitted, the search radius is based on the point span.
    sdd_search_range_cm : tuple, optional
        Inclusive search range for sample-to-detector distance in cm as
        (minimum, maximum).
    coarse_grid_points : int, optional
        Number of grid points per dimension for the coarse search.
    fine_grid_points : int, optional
        Number of grid points per dimension for the fine search.
    minimum_ring_uncertainty_px : float, optional
        Lower bound for the radial uncertainty assigned to each supplied
        ring point.
    return_details : bool, optional
        If set to True, also return a dictionary with the fitted beam
        center, assigned ring indices, fitted q values, and fit score.

    Returns
    -------
    sdd_cm : float
        Estimated sample-to-detector distance in cm.
    uncertainty_cm : float
        Estimated standard uncertainty in sample-to-detector distance in
        cm from a post-index nonlinear refit using image-based radial
        peak-position uncertainties aggregated by ring index.
    details : dict, optional
        Returned only when return_details is True. Contains
        beam_center_px, ring_indices, q_values_nm_inverse, score,
        radial_distances_px, peak_position_uncertainty_px,
        fitted_peak_positions_px, aggregated_ring_indices,
        aggregated_radial_distances_px,
        aggregated_peak_position_uncertainty_px, grid_sdd_cm,
        grid_uncertainty_cm, and standard_uncertainty_cm.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")

    peak_positions = np.asarray(peak_positions, dtype=float)
    if peak_positions.ndim != 2 or peak_positions.shape[1] != 2:
        raise ValueError(
            "peak_positions must be a 2D array with shape (n_peaks, 2)."
        )
    if peak_positions.shape[0] < 2:
        raise ValueError("At least two peak positions are required.")
    if pitch_nm <= 0 or wavelength_nm <= 0 or pixel_size_um <= 0:
        raise ValueError(
            "pitch_nm, wavelength_nm, and pixel_size_um must be positive."
        )
    if minimum_ring_uncertainty_px <= 0:
        raise ValueError("minimum_ring_uncertainty_px must be positive.")
    if exclude_within_beamstop_radius_px is not None:
        exclude_within_beamstop_radius_px = float(
            exclude_within_beamstop_radius_px
        )
        if exclude_within_beamstop_radius_px < 0:
            raise ValueError(
                "exclude_within_beamstop_radius_px must be non-negative."
            )

    sdd_min_cm, sdd_max_cm = map(float, sdd_search_range_cm)
    if sdd_min_cm <= 0 or sdd_max_cm <= sdd_min_cm:
        raise ValueError(
            "sdd_search_range_cm must contain positive increasing values."
        )

    if beam_center_guess_px is None:
        beam_center_guess_px = np.mean(peak_positions, axis=0)
    else:
        beam_center_guess_px = np.asarray(beam_center_guess_px, dtype=float)
        if beam_center_guess_px.shape != (2,):
            raise ValueError(
                "beam_center_guess_px must contain [row, column]."
            )

    if exclude_within_beamstop_radius_px is not None:
        radial_distance_from_guess_px = np.linalg.norm(
            peak_positions - beam_center_guess_px,
            axis=1,
        )
        keep_mask = (
            radial_distance_from_guess_px >= exclude_within_beamstop_radius_px
        )
        peak_positions = peak_positions[keep_mask]
        if peak_positions.shape[0] < 2:
            raise ValueError(
                "At least two peak positions must remain after "
                "beamstop exclusion."
            )

    peak_span = np.ptp(peak_positions, axis=0)
    default_radius = max(float(np.max(peak_span)) / 2, 2.0)
    if beam_center_search_radius_px is None:
        beam_center_search_radius_px = (default_radius, default_radius)
    elif np.isscalar(beam_center_search_radius_px):
        beam_center_search_radius_px = (
            float(beam_center_search_radius_px),
            float(beam_center_search_radius_px),
        )
    else:
        beam_center_search_radius_px = tuple(beam_center_search_radius_px)
        if len(beam_center_search_radius_px) != 2:
            raise ValueError(
                "beam_center_search_radius_px must be a scalar or length 2."
            )

    row_bounds = (
        float(np.min(peak_positions[:, 0]) - peak_span[0]),
        float(np.max(peak_positions[:, 0]) + peak_span[0]),
    )
    col_bounds = (
        float(np.min(peak_positions[:, 1]) - peak_span[1]),
        float(np.max(peak_positions[:, 1]) + peak_span[1]),
    )

    best = None
    search_specs = [
        (
            coarse_grid_points,
            beam_center_search_radius_px,
            (sdd_min_cm, sdd_max_cm),
        ),
        (
            fine_grid_points,
            (
                max(beam_center_search_radius_px[0] / 4, 0.5),
                max(beam_center_search_radius_px[1] / 4, 0.5),
            ),
            None,
        ),
    ]

    for grid_points, center_radius, sdd_bounds in search_specs:
        if best is None:
            center_seed = beam_center_guess_px
            sdd_seed = (sdd_min_cm + sdd_max_cm) / 2
            sdd_half_width = (sdd_max_cm - sdd_min_cm) / 2
        else:
            center_seed = best["beam_center_px"]
            sdd_seed = best["sdd_cm"]
            sdd_half_width = max(best["sdd_half_width_cm"] / 4, 0.25)

        if sdd_bounds is None:
            sdd_bounds = (
                max(sdd_min_cm, sdd_seed - sdd_half_width),
                min(sdd_max_cm, sdd_seed + sdd_half_width),
            )

        row_values = _build_search_values(
            center_value=center_seed[0],
            half_width=center_radius[0],
            points=grid_points,
            lower_bound=row_bounds[0],
            upper_bound=row_bounds[1],
        )
        col_values = _build_search_values(
            center_value=center_seed[1],
            half_width=center_radius[1],
            points=grid_points,
            lower_bound=col_bounds[0],
            upper_bound=col_bounds[1],
        )
        sdd_values = np.linspace(
            sdd_bounds[0],
            sdd_bounds[1],
            int(grid_points),
        )

        for row_center in row_values:
            for col_center in col_values:
                beam_center_px = np.array(
                    [row_center, col_center],
                    dtype=float,
                )
                for sdd_cm in sdd_values:
                    score, ring_indices, q, supported_mask = (
                        _score_sdd_ring_candidate(
                            peak_positions=peak_positions,
                            beam_center_px=beam_center_px,
                            sdd_cm=sdd_cm,
                            pixel_size_um=pixel_size_um,
                            wavelength_nm=wavelength_nm,
                            pitch_nm=pitch_nm,
                        )
                    )
                    if best is None or score < best["score"]:
                        best = {
                            "score": score,
                            "beam_center_px": beam_center_px,
                            "sdd_cm": float(sdd_cm),
                            "ring_indices": ring_indices,
                            "q": q,
                            "supported_mask": supported_mask,
                            "sdd_half_width_cm": max(
                                (sdd_bounds[1] - sdd_bounds[0]) / 2,
                                0.25,
                            ),
                        }

    sdd_probe_half_width = max(best["sdd_half_width_cm"] / 4, 0.25)
    sdd_probe_values = _build_search_values(
        center_value=best["sdd_cm"],
        half_width=sdd_probe_half_width,
        points=max(fine_grid_points, 11),
        lower_bound=sdd_min_cm,
        upper_bound=sdd_max_cm,
    )
    sdd_scores = []
    for sdd_cm in sdd_probe_values:
        score, _, _, _ = _score_sdd_ring_candidate(
            peak_positions=peak_positions,
            beam_center_px=best["beam_center_px"],
            sdd_cm=sdd_cm,
            pixel_size_um=pixel_size_um,
            wavelength_nm=wavelength_nm,
            pitch_nm=pitch_nm,
        )
        sdd_scores.append(score)
    sdd_scores = np.asarray(sdd_scores)
    min_score = float(np.min(sdd_scores))
    threshold = min_score + max(min_score * 0.1, 1e-12)
    within_threshold = sdd_probe_values[sdd_scores <= threshold]
    if within_threshold.size >= 2:
        grid_uncertainty_cm = float(
            (within_threshold[-1] - within_threshold[0]) / 2
        )
    else:
        step_size = (
            np.min(np.diff(sdd_probe_values)) / 2
            if sdd_probe_values.size > 1 else 0.25
        )
        grid_uncertainty_cm = float(max(step_size, 0.01))

    fitted_peak_positions, peak_position_uncertainty_px, radial_fit_success = (
        _estimate_ring_peak_radial_uncertainty_from_image(
            image=image,
            peak_positions=peak_positions[best["supported_mask"]],
            beam_center_px=best["beam_center_px"],
            minimum_uncertainty_px=minimum_ring_uncertainty_px,
        )
    )
    refined_sdd_cm, standard_uncertainty_cm, aggregated_ring_observations = (
        _refine_sdd_from_ring_observations(
            peak_positions=fitted_peak_positions,
            beam_center_px=best["beam_center_px"],
            ring_indices=best["ring_indices"][best["supported_mask"]],
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            pitch_nm=pitch_nm,
            sdd_initial_cm=best["sdd_cm"],
            peak_position_uncertainty_px=peak_position_uncertainty_px,
            minimum_uncertainty_px=minimum_ring_uncertainty_px,
        )
    )
    radial_distances_px = aggregated_ring_observations["radial_distances_px"]
    uncertainty_cm = standard_uncertainty_cm

    if not return_details:
        return refined_sdd_cm, uncertainty_cm

    details = {
        "beam_center_px": tuple(best["beam_center_px"]),
        "ring_indices": best["ring_indices"].copy(),
        "supported_ring_mask": best["supported_mask"].copy(),
        "q_values_nm_inverse": best["q"].copy(),
        "score": best["score"],
        "radial_distances_px": radial_distances_px.copy(),
        "grid_sdd_cm": best["sdd_cm"],
        "grid_uncertainty_cm": grid_uncertainty_cm,
        "standard_uncertainty_cm": standard_uncertainty_cm,
        "peak_position_uncertainty_px": peak_position_uncertainty_px.copy(),
        "fitted_peak_positions_px": fitted_peak_positions.copy(),
        "radial_fit_success": radial_fit_success.copy(),
        "aggregated_ring_indices": (
            aggregated_ring_observations["aggregated_ring_indices"].copy()
        ),
        "aggregated_radial_distances_px": (
            aggregated_ring_observations[
                "aggregated_radial_distances_px"
            ].copy()
        ),
        "aggregated_peak_position_uncertainty_px": (
            aggregated_ring_observations[
                "aggregated_radial_uncertainty_px"
            ].copy()
        ),
    }

    return refined_sdd_cm, uncertainty_cm, details


def estimate_sample_detector_distance_from_ring_sectors(
        image,
        beam_center_guess_px,
        sdd_guess_cm,
        pitch_nm,
        wavelength_nm,
        pixel_size_um,
        ring_indices=None,
        sector_step_deg=10.0,
        sector_width_deg=10.0,
        radial_window_px=8.0,
        radial_bin_size_px=1.0,
        exclude_within_beamstop_radius_px=None,
        beam_center_search_radius_px=5.0,
        sdd_search_half_width_cm=None,
        sdd_search_range_cm=None,
        coarse_grid_points=11,
        fine_grid_points=11,
        minimum_ring_uncertainty_px=0.5,
        show_progress=False,
        show_diagnostic_plot=False,
        return_details=False):
    """Estimate beam center and SDD from sector radial averages of rings.

    Parameters
    ----------
    show_progress : bool, optional
        If True, display a tqdm progress bar while sector angles are
        processed for each beam-center/SDD candidate. The current angle
        is shown in the progress bar postfix.
    show_diagnostic_plot : bool, optional
        If True, create a matplotlib figure showing the log-scale image,
        ring search windows, fitted sector peak locations, and beam
        center.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if pitch_nm <= 0 or wavelength_nm <= 0 or pixel_size_um <= 0:
        raise ValueError(
            "pitch_nm, wavelength_nm, and pixel_size_um must be positive."
        )
    beam_center_guess_px = np.asarray(beam_center_guess_px, dtype=float)
    if beam_center_guess_px.shape != (2,):
        raise ValueError("beam_center_guess_px must contain [row, column].")
    sdd_guess_cm = float(sdd_guess_cm)
    if sdd_guess_cm <= 0:
        raise ValueError("sdd_guess_cm must be positive.")
    if sector_step_deg <= 0 or sector_width_deg <= 0:
        raise ValueError(
            "sector_step_deg and sector_width_deg must be positive."
        )
    if radial_window_px <= 0 or radial_bin_size_px <= 0:
        raise ValueError(
            "radial_window_px and radial_bin_size_px must be positive."
        )
    if minimum_ring_uncertainty_px <= 0:
        raise ValueError("minimum_ring_uncertainty_px must be positive.")
    if exclude_within_beamstop_radius_px is not None:
        exclude_within_beamstop_radius_px = float(
            exclude_within_beamstop_radius_px
        )
        if exclude_within_beamstop_radius_px < 0:
            raise ValueError(
                "exclude_within_beamstop_radius_px must be non-negative."
            )

    if np.isscalar(beam_center_search_radius_px):
        beam_center_search_radius_px = (
            float(beam_center_search_radius_px),
            float(beam_center_search_radius_px),
        )
    else:
        beam_center_search_radius_px = tuple(beam_center_search_radius_px)
        if len(beam_center_search_radius_px) != 2:
            raise ValueError(
                "beam_center_search_radius_px must be a scalar or length 2."
            )

    if sdd_search_range_cm is None:
        if sdd_search_half_width_cm is None:
            sdd_search_half_width_cm = max(0.1 * sdd_guess_cm, 1.0)
        sdd_search_range_cm = (
            max(sdd_guess_cm - float(sdd_search_half_width_cm), 0.1),
            sdd_guess_cm + float(sdd_search_half_width_cm),
        )
    sdd_min_cm, sdd_max_cm = map(float, sdd_search_range_cm)
    if sdd_min_cm <= 0 or sdd_max_cm <= sdd_min_cm:
        raise ValueError(
            "sdd_search_range_cm must contain positive increasing values."
        )

    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=len(np.arange(0.0, 360.0, float(sector_step_deg))),
            desc="Sector fit from initial guess",
            leave=False,
        )
    try:
        observations = _collect_ring_sector_observations(
            image=image,
            beam_center_px=beam_center_guess_px,
            sdd_cm=sdd_guess_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            sector_step_deg=sector_step_deg,
            sector_width_deg=sector_width_deg,
            ring_indices=ring_indices,
            radial_window_px=radial_window_px,
            radial_bin_size_px=radial_bin_size_px,
            exclude_within_beamstop_radius_px=exclude_within_beamstop_radius_px,
            minimum_ring_uncertainty_px=minimum_ring_uncertainty_px,
            progress_bar=progress_bar,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if observations["peak_positions"].shape[0] < 2:
        raise RuntimeError(
            "Could not identify enough sector ring observations to refine SDD."
        )

    row_bounds = (
        max(0.0, beam_center_guess_px[0] - beam_center_search_radius_px[0]),
        min(float(image.shape[0] - 1),
            beam_center_guess_px[0] + beam_center_search_radius_px[0]),
    )
    col_bounds = (
        max(0.0, beam_center_guess_px[1] - beam_center_search_radius_px[1]),
        min(float(image.shape[1] - 1),
            beam_center_guess_px[1] + beam_center_search_radius_px[1]),
    )
    refined_beam_center_px, refined_sdd_cm, parameter_uncertainty = (
        _refine_beam_center_and_sdd_with_fixed_orders(
            peak_positions=observations["peak_positions"],
            beam_center_initial_px=beam_center_guess_px,
            orders=observations["ring_indices"],
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            pitch_nm=pitch_nm,
            sdd_initial_cm=sdd_guess_cm,
            peak_position_uncertainty_px=(
                observations["peak_position_uncertainty_px"]
            ),
            beam_center_bounds_px=(row_bounds, col_bounds),
            sdd_bounds_cm=(sdd_min_cm, sdd_max_cm),
        )
    )

    refined_sdd_cm, standard_uncertainty_cm, aggregated_ring_observations = (
        _refine_sdd_from_ring_observations(
            peak_positions=observations["peak_positions"],
            beam_center_px=refined_beam_center_px,
            ring_indices=observations["ring_indices"],
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            pitch_nm=pitch_nm,
            sdd_initial_cm=refined_sdd_cm,
            peak_position_uncertainty_px=(
                observations["peak_position_uncertainty_px"]
            ),
            minimum_uncertainty_px=minimum_ring_uncertainty_px,
        )
    )
    uncertainty_cm = standard_uncertainty_cm
    diagnostic_figure = None
    if show_diagnostic_plot:
        diagnostic_figure = _plot_ring_sector_fit_diagnostics(
            image=image,
            beam_center_px=refined_beam_center_px,
            ring_indices=observations["ring_indices"],
            sector_angles_deg=observations["sector_angles_deg"],
            fitted_peak_positions_px=observations["peak_positions"],
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            sdd_cm=refined_sdd_cm,
            radial_window_px=radial_window_px,
            title='Ring sector fit diagnostics',
        )

    if not return_details:
        return refined_sdd_cm, uncertainty_cm

    details = {
        "beam_center_px": tuple(refined_beam_center_px),
        "grid_sdd_cm": sdd_guess_cm,
        "standard_uncertainty_cm": standard_uncertainty_cm,
        "score": np.nan,
        "ring_indices": observations["ring_indices"].copy(),
        "sector_angles_deg": observations["sector_angles_deg"].copy(),
        "fitted_peak_positions_px": (
            observations["peak_positions"].copy()
        ),
        "peak_position_uncertainty_px": (
            observations["peak_position_uncertainty_px"].copy()
        ),
        "fitted_radii_px": observations["radii_px"].copy(),
        "beam_center_standard_uncertainty_px": parameter_uncertainty[:2].copy(),
        "sdd_standard_uncertainty_cm": float(parameter_uncertainty[2]),
        "aggregated_ring_indices": (
            aggregated_ring_observations["aggregated_ring_indices"].copy()
        ),
        "aggregated_radial_distances_px": (
            aggregated_ring_observations[
                "aggregated_radial_distances_px"
            ].copy()
        ),
        "aggregated_peak_position_uncertainty_px": (
            aggregated_ring_observations[
                "aggregated_radial_uncertainty_px"
            ].copy()
        ),
    }
    if diagnostic_figure is not None:
        details["diagnostic_figure"] = diagnostic_figure

    return refined_sdd_cm, uncertainty_cm, details


def default_mask(data):

    """
    Generate a boolean mask for invalid numeric values.

    Parameters
    ----------
    data : array-like
        Input numeric data.

    Returns
    -------
    mask : ndarray
        Boolean mask that is True where data contains nan, inf, or -inf.
    """
    data = np.array(data)
    mask = np.isnan(data)
    mask += np.isinf(data)
    mask += np.isneginf(data)

    return mask


def find_gaussian_peakloc(x, y, p0=None):
    """
    Find the peak location of a one-dimensional spectra using a Gaussian
    fit.

    Parameters
    ----------
    x : ndarray, float
        One-dimensional array of independent variable.
    y : ndarray, float
        One-dimensional array of dependent variable.
    p0 : list, optional
        Initial guess of the mean, std_dev, scale, and offset parameters
        of the Gaussian function.

    Returns
    -------
    peak_x : float
        Peak location based on the Gaussian fit.
    popt : list
        Optimized parameters mean, std_dev, scale, and offset from the
        Gaussian fit.
    """

    mask = default_mask(y)
    x_fit = np.array(x)[~mask]
    y_fit = np.array(y)[~mask]

    order_by_x = np.argsort(x_fit)
    x_fit = x_fit[order_by_x]
    y_fit = y_fit[order_by_x]

    if x_fit.shape[0] < 4:
        raise TypeError(
            "More than 4 data points need to be provided."
        )

    if p0 is None:
        try:
            offset = np.min(y_fit)
            amplitude = np.max(y_fit) - offset
            half_max = amplitude/2
            above_half = np.where(y_fit > half_max)[0]
            try:
                stdev = (x_fit[np.max(above_half)] - x_fit[np.min(above_half)])/2.355
            except:
                stdev = 1.0
            scale = amplitude * np.sqrt(2*np.pi)*stdev
            p0 = [x[np.argmax(y_fit)], stdev, scale, offset]
        except:
            p0 = None

    popt, _ = curve_fit(
        gaussian,
        x_fit, y_fit,
        p0=p0,
    )
    peak_x = popt[0]

    return float(peak_x), popt


def line_fit(x, y, force_intercept=None):
    """
    Fit a line to x and y data and return angle, slope, and intercept.

    The angle is defined counterclockwise from the x-axis. If the fit is
    constrained with force_intercept, the returned line passes through that
    point. In the case of a vertical line, slope and intercept are returned
    as nan.

    Parameters
    ----------
    x : array-like
        x coordinates of the data points.
    y : array-like
        y coordinates of the data points.
    force_intercept : tuple | None, optional
        Point (x, y) through which the fitted line must pass.

    Returns
    -------
    angle : float
        Line angle in degrees.
    slope : float
        Fitted slope.
    intercept : float
        Fitted intercept.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    if force_intercept is not None:
        x = x - force_intercept[0]
        y = y - force_intercept[1]

    if len(x) == 1:
        warnings.warn(
            "Only one point was provided for a line fit. Two or more" \
            "are required. Assuming a horizontal line."
        )
    try:
        model = LinearRegression(
            fit_intercept=False if force_intercept is not None else True)
        model.fit(x, y)
        slope = float(model.coef_[0][0])
        intercept = np.asarray(model.intercept_).item() # convert to scalar

        if force_intercept is not None:
            intercept = force_intercept[1] - slope * force_intercept[0]
        angle = np.rad2deg(np.arctan(slope))
    except ValueError:
        # vertical line
        angle = 90
        slope = np.nan
        intercept = np.nan

    return angle, slope, intercept


def gaussian_refine_peak_2D(image):

    """
    Refine a two-dimensional peak position with one Gaussian fit per axis.

    Parameters
    ----------
    image : ndarray
        Two-dimensional image region containing a single dominant peak.

    Returns
    -------
    a_opt : float
        Refined peak position along axis 0.
    b_opt : float
        Refined peak position along axis 1.
    """

    try:
        a_opt, _ = find_gaussian_peakloc(
            np.arange(0, image.shape[0]),
            np.nansum(image, axis=1))
    except RuntimeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 0;"
            "assuming peak is at the pixel with the highest value.")
        a_opt = int(np.nanargmax(np.nansum(image, axis=1)))
    except TypeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 0;"
            "assuming peak is at the pixel with the highest value.")
        a_opt = int(np.nanargmax(np.nansum(image, axis=1)))

    try:
        b_opt, _ = find_gaussian_peakloc(
            np.arange(0, image.shape[1]),
            np.nansum(image, axis=0))
    except RuntimeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 1;"
            "assuming peak is at the pixel with the highest value.")
        b_opt = int(np.nanargmax(np.nansum(image, axis=0)))
    except TypeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 1;"
            "assuming peak is at the pixel with the highest value.")
        b_opt = int(np.nanargmax(np.nansum(image, axis=0)))

    return float(a_opt), float(b_opt)


def rotate_image(image,
                 degrees, rotation_center,
                 resampling_mode="bilinear",
                 fill_mode="constant",
                 fill_constant=np.nan,
                 log_scale=False,
                 **kwargs):
    """

    Rotates an image by a specified number of degrees counterclockwise
    about the rotation center.

    Parameters
    ----------
    image : ndarray
        Two-dimensional image for rotation.
    rotation_center : list
        Center of rotation. Indices should be provided as [row, column]
        keeping in mind that numpy index orders rows from top to
        bottom and columns from left to right.
    resampling_mode : str, optional
        Set the resampling method used during the rotation.
        The box rotation works by rotating the image underneath then
        extracting the box for integration. Resampling modes are
        chosen from the sklearn.transform.warp method. Options are:
            nearest_neighbor
            bilinear (default)
            biquadratic
            bicubic
            biquartic
            biquintic
        Default value is 'bilinear'.
    fill_mode : str, optional
        Determine how pixels outside the boundaries of the input image
        are filled after the rotation. Options match those from np.pad.
        Options are:
            constant (default)
            edge
            symmetric
            reflect
            wrap
        Default value is "constant".
    fill_constant : float, optional
        Specifies the constant value used to fill pixels outside the
        image boundaries after rotation. Only applies when resampling_mode
        is set to 'constant'.
    log_scale : bool, optional
        Rotate the log-scale of your image. This could help resolve
        some artifacts caused by certain rotation sampling algorithms
        but you will lose any pixels that are negative (turned to nan).
        Deafult value is False.

    Other Parameters
    ----------------
    **kwargs
        Other keyword arguments for skimage.transform.rotate are
        accepted. These include:
            resize
            clip
            preserve_range

    Returns
    -------
    image : ndarray
        Two-dimensional rotated image of same dimensions as 'image'.
    """

    sklearn_resampling_modes = {
        "nearest_neighbor": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5,
    }

    resampling_mode = ''.join(filter(str.isalpha, resampling_mode.lower()))
    resampling_order = sklearn_resampling_modes[resampling_mode]

    fill_mode = fill_mode.lower()

    image = np.array(image)
    if log_scale:
        image = np.log10(image)
    image[default_mask(image)] = np.nan

    image = transform.rotate(
        image,
        angle=degrees,
        center=(rotation_center[1], rotation_center[0]),  # needs (col, row)
        order=resampling_order,
        mode=fill_mode,
        cval=fill_constant,
        **kwargs
    )

    if log_scale:
        image = np.power(10, image)

    return image


def rotate_image_pillow(
        image, degrees, rotation_center, resampling_mode="bilinear",
        log_scale=False, fillcolor=-9999):
    """

    Rotates an image by a specified number of degrees counterclockwise
    about the rotation center.

    Parameters
    ----------
    image : ndarray
        Two-dimensional image for rotation.
    rotation_center : list
        Center of rotation. Indices should be provided as [row, column]
        keeping in mind that numpy index orders rows from top to
        bottom and columns from left to right.
    resampling_mode : str, optional
        Set the resampling method used during the rotation.
        The box rotation works by rotating the image underneath then
        extracting the box for integration. Resampling of the
        image intensities can be performed with the 'nearest',
        'bilinear', or 'bicubic' methods in the PILLOW package.
        We encourage the user to look into the rotation sampling modes
        as this may result in 'features' in your data due to sharp
        log-scale peaks.
        Default value is 'bilinear'.
    log_scale : bool, optional
        Rotate the log-scale of your image. This could help resolve
        some artifacts caused by certain rotation sampling algorithms
        but you will lose any pixels that are negative (turned to nan).
        Deafult value is False.
    fillcolor : float, optional
        A temporary value used to fill pixels that are outside of the
        original image after rotation. These pixels will
        be replaced with NAN after the rotation is complete. A float
        is used to comply with the keyword argument requirements of
        the pillow package rotate() function used to perform the
        rotation.
        Default value is -9999.

    Returns
    -------
    image : ndarray
        Two-dimensional rotated image of same dimensions as 'image'.
    """

    if resampling_mode == 'nearest':
        resample = Image.Resampling.NEAREST
    elif resampling_mode == 'bilinear':
        resample = Image.Resampling.BILINEAR
    else:
        resample = Image.Resampling.BICUBIC

    image = np.array(image)
    if log_scale:
        image = np.log10(image)
    image[default_mask(image)] = np.nan
    # convert to PILLOW Image for the rotation
    image = Image.fromarray(image)
    image = image.rotate(
        degrees,
        resample=resample,
        # pillow calls for (x, y) of beam center
        center=(rotation_center[1], rotation_center[0]),
        fillcolor=fillcolor)
    image = np.array(image)
    image[image == fillcolor] = np.nan
    if log_scale:
        image = np.power(10, image)

    return image


def find_peaks_1D(data, log_scale=True, refinement_size=7, mask=None,
                  algorithm='scikit', refinement=True, **kwargs):
    """
    Find peaks across one-dimensional data using scikit-image.feature
    peak_local_max() function. The peak location is then further refined
    with a Gaussian fit along the single axis. Refinement is required for
    more accurate peak positions as the peak_local_max() only returns
    the positions to the nearest pixel.

    Parameters
    ----------
    data : NDArray, list
        One-dimensional data as a numpy array or list.
    log_scale : bool, optional
        If set to True, the data will be passed to the peak finding
        algorithm on a log sale of intensity. If set to False, the image
        will be sent to the peak finding algorithm with its original
        values.
        Default value is True.
    refinement_size : int
        Define the pixel range centered on the peaks in which to peform
        Gaussian refinement.
        Default value is 7. Minimum value is 4.
    mask : NDArray, list
        One-dimensional boolean array or list of points to mask during
        the peak finding operation.
    algorithm: str
        Specify which peak finding algorithm is used. Default value is
        'scikit' which uses scikit-image.feature peak_local_max() to
        locate the peaks. If set instead to 'scipy', the scipy.signal
        find_peaks() algorithm will be used instead.

    Other Parameters
    ----------------
    **kwargs
        The keyword arguments for the specified peak finding algorithm
        can be passed through.
        If using scikit-image's peak_local_max() function (algorithm set
        to 'scikit'), keyword arguments include:
            min_distance
            threshold_abs
            threshold_rel
            exclude_border
            num_peaks
            footprint
            labels
            num_peaks_per_label
            p_norm
        The threshold_abs keyword will always be set to 0 if no other
        value is provided by the user. This is to account for the -inf
        values after the log transform of the image.

        If using scipy's find_peaks() function (algorithm set to
        'scipy'), keyword arguments include:
            height
            threshold
            distance
            prominence
            width
            wlen
            rel_height
            pleateau_size

        Note that these argument lists are not always kept up to date
        and we encourage the user to reference the scikit-image or
        scipy documentation directly.

    Returns
    -------
    coordinates : NDArray
        An n x 2 array of peak coordinate positions will be returned for
        n number of peaks found.
    coordinates_px : NDArray
        An n x 2 array of peak coordinate positions rounded to the
        nearest pixels will be returned for n number of peaks found.
    """
    if algorithm not in ['scikit', 'scipy']:
        raise ValueError(
            f"The algorithm {algorithm} is not recognized. Use either "
            "'scikit' or 'scipy'."
        )

    # check the threshold_abs
    if algorithm == 'scikit':
        value = kwargs.get("threshold_abs")
        if value is None:
            kwargs["threshold_abs"] = 0

    data_fed = np.array(data)
    if mask is not None:
        data_fed[mask] = np.nan
    if log_scale:
        data_fed = np.log10(data_fed)
        data_fed[np.isneginf(data_fed)] = np.nan

    if algorithm == 'scikit':
        accepted_kwargs = [
            param.name for param in inspect.signature(
                peak_local_max).parameters.values()
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY)
            and param.default is not inspect.Parameter.empty
        ]
        coordinates_px = peak_local_max(
            data_fed,
            **{x: y for x, y in kwargs.items() if x in accepted_kwargs})
        coordinates_px = coordinates_px.tolist()
        coordinates_px = [x[0] for x in coordinates_px]
    elif algorithm == 'scipy':
        accepted_kwargs = [
            param.name for param in inspect.signature(
                find_peaks).parameters.values()
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY)
            and param.default is not inspect.Parameter.empty
        ]
        coordinates_px, _ = find_peaks(
            data_fed,
            **{x: y for x, y in kwargs.items() if x in accepted_kwargs})

    coordinates = []

    refinement_size = max(refinement_size, 4)
    if data_fed.shape[0] < 4:
        warnings.warn(
            "Data does not have enough points for refinement."
            "Using pixel location."
            )
        return np.array([x[0] for x in coordinates_px])

    for x in coordinates_px:
        x_min = max(0, x - int(refinement_size/2))
        x_max = min(x + (refinement_size - int(refinement_size/2)),
                    data_fed.shape[0])

        if x_max-x_min < 4:
            while x_max < data_fed.shape[0] and x_max-x_min < 4:
                x_max += 1
            while x_min > 0 and x_max-x_min < 4:
                x_min -= 1

        data_refine = data_fed[x_min:x_max]

        try:
            x_opt, _ = find_gaussian_peakloc(
                np.arange(0, data_refine.shape[0]),
                data_refine)
        except RuntimeError:
            warnings.warn(
                "Could not fit Gaussian to the peak location;"
                "assuming peak is at the pixel with the highest value.")
            x_opt = int(np.nanargmax(data_refine))

        x_opt += x_min

        coordinates.append(x_opt)

    coordinates = [x for x in coordinates if x < len(data) and x >= 0]

    return np.array(coordinates)


def find_peaks_2D(image, log_scale=True, refinement_size=7, mask=None,
                  **kwargs):
    """
    Find peaks across a two-dimensional image using scikit-image.feature
    peak_local_max() function and then further refined with local
    Gaussian fits across the two axes. Refinement is required for more
    accurate peak positions as the peak_local_max() only returns the
    positions to the nearest pixel.

    Parameters
    ----------
    image : NDArray
        Two-dimensional image as a numpy array.
    log_scale : bool, optional
        If set to True, the image will be passed to the peak finding
        algorithm on a log sale of intensity. If set to False, the image
        will be sent to the peak finding algorithm with its original
        values.
        Default value is True.
    refinement_size : int | None
        Define the box size around the peaks in which to peform the
        Gaussian refinement. If set to None, the pixel coordinates from
        peak_local_max() are returned without Gaussian refinement.
        Default value is 7. Minimum value is 4 when refinement is used.
    mask : NDArray
        Two-dimensional boolean array of pixels to mask during the
        peak finding operation.

    Other Parameters
    ----------------
    **kwargs
        The keyword arguments for scikit-image's peak_local_max()
        function can be passed through. Please refer to the scikit-image
        documentation for detailed information on the parameters.
        A brief list is provided here:
            min_distance
            threshold_abs
            threshold_rel
            exclude_border
            num_peaks
            footprint
            labels
            num_peaks_per_label
            p_norm
        The threshold_abs keyword will always be set to 0 if no other
        value is provided by the user. This is to account for the -inf
        values after the log transform of the image.

    Returns
    -------
    coordinates : NDArray
        An n x 2 array of peak coordinate positions will be returned for
        n number of peaks found.
    """
    # check the threshold_abs
    value = kwargs.pop("threshold_abs", 0)
    kwargs["threshold_abs"] = value

    image_fed = np.copy(image)
    if mask is not None:
        image_fed[mask] = np.nan
    if log_scale:
        image_fed = np.log10(image_fed)
        image_fed[np.isneginf(image_fed)] = np.nan


    accepted_kwargs = [
        param.name for param in inspect.signature(peak_local_max).parameters.values()
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY)
        and param.default is not inspect.Parameter.empty
    ]
    coordinates_px = peak_local_max(
        image_fed,
        **{x: y for x, y in kwargs.items() if x in accepted_kwargs})
    coordinates_px = coordinates_px.tolist()

    if refinement_size is None:
        return np.array(coordinates_px)

    coordinates = []

    refinement_size = max(refinement_size, 4)
    if min(image.shape) < 4:
        warnings.warn(
            "Image is not large enough for refinement. Using pixel location."
            )
        return np.array(coordinates_px)

    for (y, x) in coordinates_px:
        y_min = max(0, y - int(refinement_size/2))
        y_max = min(y + (refinement_size - int(refinement_size/2)),
                    image.shape[0])

        x_min = max(0, x - int(refinement_size/2))
        x_max = min(x + (refinement_size - int(refinement_size/2)),
                    image.shape[1])

        if y_max-y_min < 4:
            while y_max < image.shape[0] and y_max-y_min < 4:
                y_max += 1
            while y_min > 0 and y_max-y_min < 4:
                y_min -= 1
        if x_max-x_min < 4:
            while x_max < image.shape[1] and x_max-x_min < 4:
                x_max += 1
            while x_min > 0 and x_max-x_min < 4:
                x_min -= 1

        image_refine = image_fed[y_min:y_max, x_min:x_max]
        y_opt, x_opt = gaussian_refine_peak_2D(image_refine)
        y_opt += y_min
        x_opt += x_min

        coordinates.append([y_opt, x_opt])

    coordinates = [(y, x) for (y, x) in coordinates
                   if y < image.shape[0] and y >= 0
                   and x < image.shape[1] and x >= 0]

    return np.array(coordinates)


def find_peaks_2D_one_axis(
        image, peak_axis, integration_mode='sum', log_scale=True,
        refinement_size=7, mask=None, algorithm='scikit', **kwargs):
    """
    Find peaks along one axis of a two-dimensional image using the
    scikit-image.feature peak_local_max() function. The peaks are
    further refined with local Gaussian fits across the two axes at the
    peak locations. Refinement is required for more accurate peak
    positions as the peak_local_max() only returns the positions to the
    nearest pixel.

    The old version of this function used scipy.signal find_peaks()
    to determine the initial peak positions. It is possible to use this
    algorithm by switching the 'algorithm' keyword argument to 'scipy'.

    This function differs from find_peaks_2D() in that it only allows
    for the primary peaks to be found along a single axis. For example,
    if axis 1 is chosen as the peak axis, the image provided will be
    integrated along axis 0 (summed or averaged) to find the primary
    peak location along axis 1. Then the peak location in axis 0 will
    be determined as the highest intensity pixel at each peak location
    along axis 1. This is then refined by the Gaussian fits.

    Parameters
    ----------
    image : NDArray
        Two-dimensional image as a numpy array.
    peak_axis : int
        The axis along which the peaks should be found. If axis 0 is
        selected, the image will be integrated along axis 1. This means
        peaks found will likely be vertical in your image. If axis 1 is
        selected, the image will be integrated along axis 0. This means
        peaks found will likely be horizontal in your image.
    integration_mode : str, optional
        Integration mode to be performed along the axis not set as
        peak_axis. Options are 'mean' and 'sum'.
        Default value is 'sum'.
    log_scale : bool, optional
        If set to True, the image will be passed to the peak finding
        algorithm on a log sale of intensity. If set to False, the image
        will be sent to the peak finding algorithm with its original
        values.
        Default value is True.
    refinement_size : int
        Define the box size around the peaks in which to peform the
        Gaussian refinement.
        Default value is 7.
    algorithm: str
        Specify which peak finding algorithm is used. Default value is
        'scikit' which uses scikit-image.feature peak_local_max() to
        locate the peaks. If set instead to 'scipy', the scipy.signal
        find_peaks() algorithm will be used instead.

    Other Parameters
    ----------------
    **kwargs
        The keyword arguments for the specified peak finding algorithm
        can be passed through.
        If using scikit-image's peak_local_max() function (algorithm set
        to 'scikit'), keyword arguments include:
            min_distance
            threshold_abs
            threshold_rel
            exclude_border
            num_peaks
            footprint
            labels
            num_peaks_per_label
            p_norm
        The threshold_abs keyword will always be set to 0 if no other
        value is provided by the user. This is to account for the -inf
        values after the log transform of the image.

        If using scipy's find_peaks() function (algorithm set to
        'scipy'), keyword arguments include:
            height
            threshold
            distance
            prominence
            width
            wlen
            rel_height
            pleateau_size

        Note that these argument lists are not always kept up to date
        and we encourage the user to reference the scikit-image or
        scipy documentation directly.

    Returns
    -------
    coordinates : NDArray
        An n x 2 array of peak coordinate positions will be returned for
        n number of peaks found.
    """
    image_fed = np.copy(image)
    if mask is not None:
        image_fed[mask] = np.nan
    drop_if_any_nan = np.isnan(image_fed).any(axis=1-peak_axis)
    if integration_mode == 'sum':
        image_fed = np.nansum(image_fed, axis=1-peak_axis)
    elif integration_mode == 'mean':
        image_fed = np.nanmean(image_fed, axis=1-peak_axis)
    else:
        raise ValueError(
            f"Integration mode {integration_mode} not recognized."
            "Use 'mean' or 'sum'."
        )
    image_fed[drop_if_any_nan] = np.nan

    refinement_size = max(refinement_size, 4)
    coordinates_peak_axis = find_peaks_1D(image_fed, log_scale=log_scale,
                                          refinement_size=refinement_size,
                                          algorithm=algorithm,
                                          **kwargs)
    coordinates_peak_axis = np.round(
        np.array(coordinates_peak_axis), 0).astype(int)

    if peak_axis == 1:
        peaks_other = np.argmax(image[:, coordinates_peak_axis], axis=0)
        coordinates_px = [
            (y, x) for x, y in zip(coordinates_peak_axis, peaks_other)]
    else:
        peaks_other = np.argmax(image[coordinates_peak_axis, :], axis=1)
        coordinates_px = [
            (y, x) for y, x in zip(coordinates_peak_axis, peaks_other)]

    coordinates_px = []
    for a in coordinates_peak_axis:
        a = int(np.round(a, 0))
        if peak_axis == 1:
            coordinates_px.append([int(np.argmax(image[:, a])), a])
        elif peak_axis == 0:
            coordinates_px.append([a, int(np.argmax(image[a, :]))])

    if min(image.shape) < 4:
        warnings.warn(
            "Image is not large enough for refinement. Using pixel location."
            )
        return np.array(coordinates_px)

    coordinates = []
    if log_scale:
        image = np.log10(image)
        image[np.isneginf(image)] = np.nan
    else:
        image = np.array(image)

    for (y, x) in coordinates_px:
        y_min = max(0, y - int(refinement_size/2))
        y_max = min(y + (refinement_size - int(refinement_size/2)),
                    image.shape[0])

        x_min = max(0, x - int(refinement_size/2))
        x_max = min(x + (refinement_size - int(refinement_size/2)),
                    image.shape[1])

        if y_max-y_min < 4:
            while y_max < image.shape[0] and y_max-y_min < 4:
                y_max += 1
            while y_min > 0 and y_max-y_min < 4:
                y_min -= 1
        if x_max-x_min < 4:
            while x_max < image.shape[1] and x_max-x_min < 4:
                x_max += 1
            while x_min > 0 and x_max-x_min < 4:
                x_min -= 1

        image_refine = image[y_min:y_max, x_min:x_max]
        y_opt, x_opt = gaussian_refine_peak_2D(image_refine)
        y_opt += y_min
        x_opt += x_min

        coordinates.append([y_opt, x_opt])

    coordinates = [(y, x) for (y, x) in coordinates
                   if y < image.shape[0] and y >= 0
                   and x < image.shape[1] and x >= 0]

    return np.array(coordinates)


def find_maximum_rectangular_roi(data):
    """
    Find the largest rectangular region of valid pixels in a mask.

    The input should be a boolean array, or an array of 1 and 0 values,
    where True or 1 marks pixels that belong to the candidate region of
    interest.

    Parameters
    ----------
    data : ndarray
        Two-dimensional boolean or binary array describing valid pixels.

    Returns
    -------
    row_bounds : tuple
        Row bounds of the maximum rectangular region as (min_row, max_row).
    col_bounds : tuple
        Column bounds of the maximum rectangular region as (min_col, max_col).
    height : ndarray
        Height map used internally for the rectangle search.
    width : ndarray
        Width map used internally for the rectangle search.
    area : ndarray
        Area map used internally for the rectangle search.
    """

    data = data.astype(int)

    height = np.flipud(np.cumsum(np.flipud(data), axis=0))*data
    where_ones_start = np.where((data[1:, :] - data[:-1, :]) == 1, 1, 0)
    adjustments = height[1:, :] * where_ones_start
    adjustments = np.flipud(np.maximum.accumulate(np.flipud(adjustments), axis=0))
    height[:-1, :] = height[:-1, :] - adjustments
    height[height < 0] = 0

    width = np.fliplr(np.cumsum(np.fliplr(data), axis=1))*data
    where_ones_start = np.where((data[:, 1:] - data[:, :-1]) == 1, 1, 0)
    adjustments = width[:, 1:] * where_ones_start
    adjustments = np.fliplr(np.maximum.accumulate(np.fliplr(adjustments), axis=1))
    width[:, :-1] = width[:, :-1] - adjustments
    width[width < 0] = 0

    area = height * width

    found_it = False
    counter = 0
    while not found_it and counter <= 1e5:
        min0, min1 = np.unravel_index(np.argmax(area), area.shape)

        # take the test area assuming first row and column of
        # continuous ones sets the boundaries
        test_area = data[min0:min0+height[min0, min1], min1:min1+width[min0, min1]]

        # there could still be zeros anywhere else in test area
        # find the area for different size boxes here after excluding
        # those pixels
        areas = []
        px_x = np.tile(np.arange(0, test_area.shape[1]), test_area.shape[0])+1
        px_y = np.tile(np.arange(0, test_area.shape[0]).reshape(-1, 1), test_area.shape[1]).reshape(-1)+1
        for x, y in zip(px_x, px_y):
            if np.min(test_area[:y, :x])==0:
                areas.append(0)
            else:
                areas.append(x*y)
        x, y = px_x[np.argmax(areas)], px_y[np.argmax(areas)]
        actual_area = x*y

        # new box that's actually the biggest without zeros
        test_area = test_area[:y, :x]
        area[min0, min1] = actual_area
        new_max = np.max(area)
        if new_max == actual_area or new_max == 1:
            found_it = True

        # ideally it will find the answer quick but set a break point
        # just in case for this while loop
        counter += 1

    max0 = min0 + test_area.shape[0]
    max1 = min1 + test_area.shape[1]

    return (min0, max0), (min1, max1), height, width, area

