import unittest

from cdsaxs.calculators import wavelength_to_energy, energy_to_wavelength


class TestCalculators(unittest.TestCase):

    planck_constant = 4.1356679e-15  # eV*s
    speed_of_light = 299792458  # m/s

    def setUp(self):
        self.wavelengths_nm = [12., 2., 1., 0.5, 0.1, 0.01, 0.001]
        self.energy_eV = [103.3201704344,
                          619.9210226063,
                          1239.8420452127,
                          2479.6840904254,
                          12398.4204521270,
                          123984.2045212700,
                          1239842.0452127000]

    def test_wavelength_to_energy(self):
        """
        Energy = hc / wavelength
            h = Planck constant
            c = speed of light
        """
        for energy, wavelength in zip(self.energy_eV, self.wavelengths_nm):
            energy_actual = wavelength_to_energy(wavelength)
            self.assertAlmostEqual(energy_actual, energy)

    def test_energy_to_wavelength(self):
        """
        Energy = hc / wavelength
            h = Planck constant
            c = speed of light
        """
        for energy, wavelength in zip(self.energy_eV, self.wavelengths_nm):
            wavelength_actual = energy_to_wavelength(energy)
            self.assertAlmostEqual(wavelength_actual, wavelength)
