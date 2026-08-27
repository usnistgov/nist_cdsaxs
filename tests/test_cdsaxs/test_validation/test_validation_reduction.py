import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from cdsaxs.loaders.load_data import LoadDataset
from cdsaxs.reduction import slice_reduced_dataset


VALIDATION_DIR = Path(__file__).resolve().parents[2] / 'data' / 'test_validation'
DATA_DIR = VALIDATION_DIR / 'data_W204_F2'
LEGACY_SLICES_CSV = VALIDATION_DIR / 'LegacyGUISlices_LargerWidthSlices0p001_20250509.csv'

FILENAME_PATTERN = (
    'W204_F2measure1_{sdd_cm}m_{energy_ev}keV_num{num}_'
    '{sample_phi_deg}deg_bpm{bpm}_id{id}_combined.tif'
)

Q_VALUES = -1 * np.array([
    0.006441, 0.01275, 0.01904, 0.02532, 0.03156, 0.03788, 0.04412, 0.05044,
    0.05672, 0.06294, 0.06923, 0.07553, 0.08187, 0.08805, 0.09442, 0.1007,
    -0.006155, -0.01249, -0.01871, -0.02496, -0.03124, -0.03751, -0.04373,
    -0.05005, -0.0563, -0.06248, -0.06883, -0.07513, -0.08133, -0.08767,
    -0.09397, -0.1003,
], dtype=float)

# Tolerance choices for the legacy-vs-current code validation:
# Q_MATCH_RTOL: 0.1% relative q matching tolerance, chosen to mirror the 
#       original notebook's nearest-neighbor acceptance rule.
# Q_MATCH_ATOL: small absolute q matching floor for stability near q = 0.
# REL_WARN_THRESHOLD: relative intensity error that triggers a warning.
# REL_FAIL_THRESHOLD: relative intensity error that fails the test; this is 
#       intentionally looser than the warning threshold to avoid brittle failures.
# ABS_INTENSITY_FLOOR: below this legacy intensity magnitude, the test stops 
#       using relative error because percent error becomes unstable near zero.
# ABS_ERROR_TOLERANCE: absolute intensity tolerance used for those near-zero points.
# MIN_MATCH_FRACTION: minimum fraction of legacy q points that must find a 
#       valid counterpart so the test cannot pass by comparing only a small subset.
Q_MATCH_RTOL = 1e-3
Q_MATCH_ATOL = 1e-6
REL_WARN_THRESHOLD = 1e-6
REL_FAIL_THRESHOLD = 1e-5
ABS_INTENSITY_FLOOR = 1e-12
ABS_ERROR_TOLERANCE = 1e-10
MIN_MATCH_FRACTION = 0.95


def _load_legacy_slices():
    with LEGACY_SLICES_CSV.open(newline='') as handle:
        rows = list(csv.reader(handle))

    header = rows[0]
    data_rows = rows[1:]
    data = np.array(data_rows, dtype=str)

    legacy_slices = {}
    for index in range(0, len(header) // 2):
        qz = data[:, index * 2]
        iqz = data[:, index * 2 + 1]

        selection = np.where(qz != '')
        qz = qz[selection].astype(np.float64)
        iqz = iqz[selection].astype(np.float64)

        qx_text = header[index * 2 + 1].split('=')[-1].strip()
        qx = float(np.round(np.float64(qx_text), 4))

        legacy_slices[qx] = {
            'q': qz,
            'Iq': iqz,
        }

    return legacy_slices


def _build_refactored_slices():
    dataset = LoadDataset(
        'validation set',
        str(DATA_DIR),
        metadata_pattern=FILENAME_PATTERN,
        metadata_scales={'energy_ev': 1000},
        metadata={'exposure_time_s': 2},
        filetype='tif',
        verbose=False,
    )
    dataset.update_all_metadata(
        {'sdd_cm': 504.982, 'center_px': [738, 492], 'pixel_size_um': 172},
        overwrite=True,
        verbose=False,
    )
    dataset.normalize_all_data_by_metadata(['exposure_time_s'])

    integrated_dataset = dataset.integrate_dataset(
        'sum',
        width_qdy_px=5,
        width_qdx_px=884,
    )
    reduced_slices, _ = slice_reduced_dataset(
        integrated_dataset,
        q_values=Q_VALUES,
        q_widths=0.001,
        show_plot=False,
    )

    refactored_slices = {}
    for data in reduced_slices.data:
        qx = float(np.round(-1 * float(data.qsx), 4))
        refactored_slices[qx] = {
            'q': -1 * np.asarray(data.q, dtype=float),
            'Iq': np.asarray(data.Iq, dtype=float),
        }

    return refactored_slices


def _match_curve_points(reference_q, candidate_q, q_rtol=Q_MATCH_RTOL, q_atol=Q_MATCH_ATOL):
    matched_reference = []
    matched_candidate = []

    for reference_index, q_value in enumerate(reference_q):
        candidate_index = int(np.argmin(np.abs(candidate_q - q_value)))
        if np.isclose(q_value, candidate_q[candidate_index], rtol=q_rtol, atol=q_atol):
            matched_reference.append(reference_index)
            matched_candidate.append(candidate_index)

    return np.array(matched_reference, dtype=int), np.array(matched_candidate, dtype=int)


@pytest.mark.slow
def test_validation_reduction_matches_legacy_gui_with_hybrid_tolerances():
    legacy_slices = _load_legacy_slices()
    refactored_slices = _build_refactored_slices()

    assert set(refactored_slices) == set(legacy_slices)

    warning_messages = []

    for qx, legacy_slice in legacy_slices.items():
        refactored_slice = refactored_slices[qx]
        legacy_indices, refactored_indices = _match_curve_points(
            legacy_slice['q'],
            refactored_slice['q'],
        )

        assert legacy_indices.size > 0, f'No matched q points found for qx={qx}.'
        assert legacy_indices.size >= int(np.ceil(MIN_MATCH_FRACTION * legacy_slice['q'].size)), (
            f'Insufficient q-point coverage for qx={qx}: '
            f'{legacy_indices.size}/{legacy_slice["q"].size} matched.'
        )

        legacy_iq = legacy_slice['Iq'][legacy_indices]
        refactored_iq = refactored_slice['Iq'][refactored_indices]

        near_zero_mask = np.abs(legacy_iq) < ABS_INTENSITY_FLOOR
        non_zero_mask = ~near_zero_mask

        if np.any(near_zero_mask):
            absolute_error = np.abs(refactored_iq[near_zero_mask] - legacy_iq[near_zero_mask])
            assert np.all(absolute_error <= ABS_ERROR_TOLERANCE), (
                f'Absolute error exceeded tolerance for near-zero points at qx={qx}. '
                f'Max absolute error was {absolute_error.max():.3e}.'
            )

        if np.any(non_zero_mask):
            relative_error = np.abs(
                (refactored_iq[non_zero_mask] - legacy_iq[non_zero_mask]) / legacy_iq[non_zero_mask]
            )
            max_relative_error = float(np.max(relative_error))

            assert max_relative_error <= REL_FAIL_THRESHOLD, (
                f'Relative error exceeded tolerance for qx={qx}. '
                f'Max relative error was {max_relative_error:.3e}.'
            )

            if max_relative_error > REL_WARN_THRESHOLD:
                warning_messages.append(
                    f'Validation slice qx={qx} exceeded warning threshold with '
                    f'max relative error {max_relative_error:.3e}.'
                )

    for message in warning_messages:
        warnings.warn(message)