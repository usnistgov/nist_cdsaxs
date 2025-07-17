"""
Simple calculators and frequently used functions.
"""

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
