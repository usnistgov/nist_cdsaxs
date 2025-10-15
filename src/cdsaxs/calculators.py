"""
Simple calculators and frequently used functions.
"""

import numpy as np
from scipy.special import erf

# Global constants
EV_NM = 1239.8420452127


def wavelength_to_energy(wavelength_nm: float) -> float:
    """
    Returns energy in eV from wavelength provided in nm.
    """
    return EV_NM/wavelength_nm


def energy_to_wavelength(energy_ev: float) -> float:
    """
    Returns wavelength in nm from energy provided in eV.
    """
    return EV_NM/energy_ev


def gaussian(x, mean, std_dev, scale, offset):
    """
    Calculates a Gaussian distribution following:

    f(x) = (scale / (sqrt(2*pi)*std_dev)) * exp(-(x-mean)^2/(2*std_dev^2)) + offset
    TODO: format above equation to render correctly in docs
    """

    x = np.array(x)

    a = scale * np.reciprocal(np.sqrt(2*np.pi)*std_dev)
    b = -((x-mean)**2)/(2*std_dev**2)

    return a*np.exp(b) + offset


def footprint_correction(sample_phi_deg):

    cos_sample_phi = np.cos(np.deg2rad(sample_phi_deg))

    return float(cos_sample_phi)


def sample_size_correction(sample_phi_deg, fwhm_mm, center_mm, sample_size_mm):
    cos_sample_phi = np.cos(np.deg2rad(sample_phi_deg))

    sigma_times_sqrt2 = 1e-99 + fwhm_mm / (2 * np.sqrt(np.log(2)))
    sample_size_factor = (
        (erf((center_mm + sample_size_mm / 2) / sigma_times_sqrt2) -
            erf((center_mm - sample_size_mm / 2) / sigma_times_sqrt2)) /
        (erf((center_mm + sample_size_mm / 2)
                * cos_sample_phi / sigma_times_sqrt2) -
            erf((center_mm - sample_size_mm / 2)
                * cos_sample_phi / sigma_times_sqrt2) + 1e-99))

    return sample_size_factor


def substrate_absorption_correction(sample_phi_deg,
                                    substrate_thickness_um,
                                    substrate_attenuation_coeff_um_m1):
    """
    Substrate attenuation coefficient in units of 1/um.
    """
    cos_sample_phi = np.cos(np.deg2rad(sample_phi_deg))

    substrate_absorption_factor = np.exp(
        -substrate_thickness_um * substrate_attenuation_coeff_um_m1 * (
            1 - 1 / cos_sample_phi))

    return substrate_absorption_factor
