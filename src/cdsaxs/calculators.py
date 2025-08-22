"""
Simple calculators and frequently used functions.
"""

import numpy as np

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
