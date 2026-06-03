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
    """
    This function calculates the footprint correction based on the
    sample rotation angle about the y-axis. When integrated into the
    data classes, any offset to the nominal sample_phi_deg is accounted
    for but if using this function directly, the user should be sure
    to provide the actual rotation angle for sample_phi_deg.

    Notes from legacy gui:
    footprint_factor = I(path length = sample thickness) / I (path length)
                     = (t/cos(0))/ (t/cos(sample phi)) = cos(sample phi)

    Parameters
    ----------
    sample_phi_deg : float | array-like
        Sample rotation angle about the postive y-axis in sample
        coordinate space.
        Units are degrees.

    Returns
    -------
    float | array-like
        Footprint correction value at each provided rotation angle.
        To apply the correction to the scattering intensity, multiply
        by the returned value.
    """
    sample_phi_deg = np.array(sample_phi_deg)
    cos_sample_phi = np.cos(np.deg2rad(sample_phi_deg))

    return cos_sample_phi


def sample_size_correction(sample_phi_deg, fwhm_mm, center_mm, sample_size_mm):
    """
    Calculate the sample size correction.
    TODO: flush out this doc string from legacy notes

    Parameters
    ----------
    sample_phi_deg : float | array-like
        Sample rotation angle about the positive y-axis in sample
        coordinate space.
        Units are degrees.
    fwhm_mm : float
    center_mm : float
    sample_size_mm : float

    Returns
    -------
    float | array-like
        Sample size correction value at each provided rotation angle.
        To apply the correction to the scattering intensity, multiply
        by the returned value.
    """
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
    Calculate the substrate absorption correction at each sample
    rotation angle with a known substrate thickenss and attenuation
    coefficient.

    Note from legacy gui:
    abs_factor = I(path length =substrate thickness) / I(path length)
     = (I0 exp(-mu*t)) / (I0 exp(-mu*t/cos(phi))) = exp(-mu*t*(1-1/cos(phi)))

    Parameters
    ----------
    sample_phi_deg : float | array-like
        Sample rotation angle about the positive y-axis in sample
        coordinate space.
        Units are degrees.
    substrate_thickness_um : float
        Thickness of the substrate in units of um.
    substrate_attenuation_coeff_um_m1 : float
        Substrate attenuation coefficient in units of 1/um.

    Returns
    -------
    float | array-like
        Substrate absorption correction at each provided rotation angle.
        To apply the correction to the scattering intensity, multiply
        by the returned value.
    """
    cos_sample_phi = np.cos(np.deg2rad(sample_phi_deg))

    substrate_absorption_factor = np.exp(
        -substrate_thickness_um * substrate_attenuation_coeff_um_m1 * (
            1 - 1 / cos_sample_phi))

    return substrate_absorption_factor


def polarization_factor(scattering_angle_deg, polarization):
    """
    Calculates the lorentz factor for p-polarization as:
        lorentz_factor = cos(scattering_angle)^2
    The lorentz factor for s-polarized systems is equal to 1.

    Parameters
    ----------
    scattering_angle_deg : float | array-like
        Scattering angle in units of degrees.
    polarization : str
        Specify whether the system is p-polarized or s-polarized.
        Accepted strings are "p" and "s".

    Returns
    -------
    float | array-like
        Polarization factor at each scattering angle provided. To correct
        your measured intensity, divide by this factor. If you are
        applying this factor in your model to fit the measured data,
        mulitply your model by this factor.
    """
    if polarization in ["p", "P"]:
        polar = np.cos(np.deg2rad(scattering_angle_deg))**2
    elif polarization in ["s", "S"]:
        polar = np.ones_like(scattering_angle_deg).astype(float)
    else:
        raise ValueError(f"The polarization {polarization} is not recognized. "
                         "Use 'p' or 's'.")

    return polar


def lorentz_factor(scattering_angle_deg, sample_phi_deg):
    """
    Calculates the lorentz factor for p-polarization as:
        lorentz_factor = cos(scattering_angle)^2
    The lorentz factor for s-polarized systems is equal to 1.

    Parameters
    ----------
    scattering_angle_deg : float | array-like
        Scattering angle in units of degrees.
    sample_phi_deg : float
        Rotation angle of the sample.

    Returns
    -------
    float | array-like
        Lorentz factor at each scattering angle provided. To correct
        your measured intensity, multiply by this factor. If you are
        applying this factor in your model to fit the measured data,
        divide your model by this factor.
    """
    sample_phi_deg = float(sample_phi_deg)
    scattering_angle_deg = np.array(scattering_angle_deg)
    lorentz = np.cos(np.deg2rad(scattering_angle_deg - sample_phi_deg))

    return lorentz
