"""
Simple calculators.
"""

# Global constants
EV_NM = 1239.84


def wavelength_to_energy(wavelength_nm):
    """
    Returns energy in eV from wavelength provided in nm.
    """
    return EV_NM/wavelength_nm


def energy_to_wavelength(energy_ev):
    """
    Returns wavelength in nm from energy provided in eV.
    """
    return EV_NM/energy_ev
