"""
Handle accepted metadata.
"""

import warnings

METADATA_KEYWORDS = [
    "sample_phi_deg", "sample_phi_offset_deg",
    "sample_omega_deg", "sample_omega_offset_deg",
    "sample_chi_deg", "sample_chi_offset_deg",
    "energy_ev", "wavelength_nm",
    "exposure_time_s", "sdd_cm", "pixel_size_um",
    "scaling_factor", "I0", "beam_current",
    "data_directory", "filename",
    "name", "center_px", "center_px_detector",
    "image_rotation_angle",
    "sample_size_mm", "beam_center_mm", "beam_fwhm_mm",
    "substrate_thickness_um", "substrate_attenuation_coeff_um-1",
    "footprint_factor", "sample_size_factor",
    "substrate_absorption_factor",
    "detector_phi_deg", "detector_phi0_deg", "detector_phi_scale",
    "detector_y_mm", "detector_y0_mm"

]

SAMPLE_METADATA_KEYWORDS = [
    'sample_size_mm',
    'substrate_thickness_um',
    'substrate_attenuation_coeff_um-1',
    'pitch_nm',
    'q_peak_positions_Ang-1'
]

ACCEPTED_Q_AXES = [
    "qdy", "qdx", "qdz", "qd",
    "qsy", "qsx", "qsz", "qs", "qsr",
    "qby", "qbx", "qbz", "qb", "qbr",
]

FLOATS = [
    "sample_phi_deg", "sample_phi_offset_deg",
    "sample_omega_deg", "sample_omega_offset_deg",
    "sample_chi_deg", "sample_chi_offset_deg",
    "energy_ev", "wavelength_nm",
    "exposure_time_s", "sdd_cm", "pixel_size_um",
    "scaling_factor", "I0", "beam_current",
    "sample_size_mm", "substrate_thickness_um",
    "substrate_attenuation_coeff_um-1",
    "pitch_nm", "image_rotation_angle",
    "detector_phi_deg", "detector_phi0_deg", "detector_phi_scale",
    "detector_y_mm", "detector_y0_mm",
    "beam_center_mm", "beam_fwhm_mm",
    "footprint_factor", "sample_size_factor", "substrate_absorption_factor"
]

INTEGERS = [
]

STRINGS = [
    "data_directory", "filename", "name"
]


def correct_metadata_dtype(name, value):
    """
    Convert a metadata value to the expected type for a recognized key.

    Parameters
    ----------
    name : str
        Metadata key to validate and convert.
    value : object
        Metadata value associated with name.

    Returns
    -------
    corrected_value : object
        Converted metadata value. If name is not a recognized metadata key,
        the original value is returned unchanged after issuing a warning.

    Raises
    ------
    ValueError
        Raised when value cannot be converted to the expected type for name.
    """

    if name in FLOATS:
        try:
            return float(value)
        except:
            raise ValueError(
                f"The datatype for metadata {name} should be a float."
            )
    elif name in INTEGERS:
        try:
            return int(value)
        except:
            raise ValueError(
                f"The datatype for metadata {name} should be an integer."
            )
    elif name in STRINGS:
        try:
            return str(value)
        except:
            raise ValueError(
                f"The datatype for metadata {name} should be a string."
            )
    else:
        warnings.warn(
            f"No data type known for {name}."
        )
        return value


def check_metadata(metadata, sample_mode=False):
    """
    Validate metadata keys and check for overspecified 
    wavelengths/energy inputs (only one should be set).

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to validate.
    sample_mode : bool, optional
        If True, validate metadata against sample-specific metadata keys.
        If False, validate against the full accepted metadata key list.

    Returns
    -------
    is_valid : bool
        True when metadata passes validation.

    Raises
    ------
    ValueError
        Raised when metadata contains unsupported keys or when both
        energy_ev and wavelength_nm are provided.
    """
    if sample_mode:
        unaccepted_keywords = [
            x for x in metadata.keys() if x not in SAMPLE_METADATA_KEYWORDS
        ]

        if len(unaccepted_keywords) > 0:
            raise ValueError(
                "The following metadata keywords are not accepted:\n" +
                f"{unaccepted_keywords}\n" +
                "The following are accepted sample metadata keywords:\n" +
                f"{SAMPLE_METADATA_KEYWORDS}"
            )
        return True

    else:
        unaccepted_keywords = [
            x for x in metadata.keys() if x not in METADATA_KEYWORDS
        ]

        if len(unaccepted_keywords) > 0:
            raise ValueError(
                "The following metadata keywords are not accepted:\n" +
                f"{unaccepted_keywords}\n" +
                "The following are accepted metadata keywords:\n" +
                f"{METADATA_KEYWORDS}"
            )

        if "energy_ev" in metadata.keys() and\
                "wavelength_nm" in metadata.keys():
            raise ValueError(
                "You have specified both the source energy and wavelength. "
                "Only one of these can be specified and the other is "
                "calculated. To avoid over-specifying or conflicting values, "
                "please only use one of these values. "
            )
        return True
