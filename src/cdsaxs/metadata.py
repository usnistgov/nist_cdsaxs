"""
Handle accepted metadata.
"""

METADATA_KEYWORDS = [
    "sample_kappa_deg", "sample_phi_deg", "sample_omega_deg", "energy_ev",
    "wavelength_nm", "exposure_time_s", "sdd_cm", "pixel_size_um",
    "scaling_factor", "I0", "beam_current", "data_directory", "filename",
    "name", "center_px"
]

SAMPLE_METADATA_KEYWORDS = [
    'sample_size_mm',
    'substrate_thickness_um',
    'substrate_attenuation_coeff_um-1',
]

FLOATS = [
    "sample_kappa_deg", "sample_phi_deg", "sample_omega_deg", "energy_ev",
    "wavelength_nm", "exposure_time_s", "sdd_cm", "pixel_size_um",
    "scaling_factor", "I0", "beam_current"
]

INTEGERS = [
]

STRINGS = [
    "data_directory", "filename", "name"
]


def correct_dtype(name, value):
    """
    Correct the units for metadata values.
    """

    if name in FLOATS:
        return float(value)
    elif name in INTEGERS:
        return int(value)
    elif name in STRINGS:
        return str(value)
    else:
        raise KeyError(
            f"No data type known for {name}."
        )
