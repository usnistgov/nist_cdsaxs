"""
Handle accepted metadata.
"""

import warnings

METADATA_KEYWORDS = [
    "sample_phi_deg", "sample_phi_offset_deg",
    "sample_omega_deg", "sample_chi_deg",
    "energy_ev", "wavelength_nm",
    "exposure_time_s", "sdd_cm", "pixel_size_um",
    "scaling_factor", "I0", "beam_current",
    "data_directory", "filename",
    "name", "center_px", 'sample_phi_offset_deg'
]

SAMPLE_METADATA_KEYWORDS = [
    'sample_size_mm',
    'substrate_thickness_um',
    'substrate_attenuation_coeff_um-1',
]

ACCEPTED_Q_AXES = [
    "qdy", "qdx", "qdz", "qd",
    "qsy", "qsx", "qsz", "qs",
    "qby", "qbx", "qbz", "qb"
]

FLOATS = [
    "sample_phi_deg", "sample_phi_offset_deg",
    "sample_omega_deg", "sample_chi_deg",
    "energy_ev", "wavelength_nm",
    "exposure_time_s", "sdd_cm", "pixel_size_um",
    "scaling_factor", "I0", "beam_current",
]

INTEGERS = [
]

STRINGS = [
    "data_directory", "filename", "name"
]


def correct_metadata_dtype(name, value):
    """
    Correct the data type for metadata values.
    """

    if name in FLOATS:
        return float(value)
    elif name in INTEGERS:
        return int(value)
    elif name in STRINGS:
        return str(value)
    else:
        warnings.warn(
            f"No data type known for {name}."
        )
        return value
