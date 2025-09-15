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
    "name", "center_px", 'sample_phi_offset_deg',
    "image_rotation_angle",
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
    "qsy", "qsx", "qsz", "qs",
    "qby", "qbx", "qbz", "qb"
]

FLOATS = [
    "sample_phi_deg", "sample_phi_offset_deg",
    "sample_omega_deg", "sample_chi_deg",
    "energy_ev", "wavelength_nm",
    "exposure_time_s", "sdd_cm", "pixel_size_um",
    "scaling_factor", "I0", "beam_current",
    "sample_size_mm", "substrate_thickness_um",
    "substrate_attenuation_coeff_um-1",
    "pitch_nm", "image_rotation_angle"
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


def check_metadata(metadata, sample_mode=False):
    """
    Check the metadata dictionary for:
    - unaccepted metadata keywords
    - overspecified wavelength/energy (onle one should be set)

    If sample_mode is set to True, it will check the metadata against
    the accepted sample metadata keywords.
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
