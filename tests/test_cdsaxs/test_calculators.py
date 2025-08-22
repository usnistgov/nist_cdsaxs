import unittest

import numpy as np

from cdsaxs.calculators import wavelength_to_energy, energy_to_wavelength
from cdsaxs.calculators import gaussian


class TestCalculatorsEnergyWavelength(unittest.TestCase):

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


class TestGaussian(unittest.TestCase):

    def test_values(self):

        x = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3]
        y = [
            2.0474927095, 2.1699622264, 2.4737009498, 3.0282115523,
            3.7381493166, 4.2883268928, 4.3462561639, 3.8735236002,
            3.1651163299, 2.5642944643, 2.2128475571, 2.0625256089,
            2.0143045292
        ]
        y = np.array(y)

        mean = 1.4
        std_dev = 0.5
        scale = 3
        offset = 2

        test_y = gaussian(x, mean, std_dev, scale, offset)

        np.testing.assert_array_almost_equal(test_y, y)
