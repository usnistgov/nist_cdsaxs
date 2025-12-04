import unittest

import numpy as np

from cdsaxs.calculators import (
    wavelength_to_energy,
    energy_to_wavelength,
    gaussian,
    footprint_correction,
    sample_size_correction,
    substrate_absorption_correction,
)


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


class TestFootprintCorrection(unittest.TestCase):

    def test_values(self):

        sample_phis = [
            -40, -30, -25, -15, -10, 1, 0.1, 0, 0.1, 1, 10, 15, 25, 30, 40
        ]
        desired_footprints = [
            0.766044443119,
            0.866025403784,
            0.906307787037,
            0.965925826289,
            0.984807753012,
            0.999847695156,
            0.999998476913,
            1.000000000000,
            0.999998476913,
            0.999847695156,
            0.984807753012,
            0.965925826289,
            0.906307787037,
            0.866025403784,
            0.766044443119,
        ]

        actual_footprints = footprint_correction(sample_phis)
        np.testing.assert_almost_equal(actual_footprints, desired_footprints)


class TestSampleSizeCorrection(unittest.TestCase):

    def test_values(self):
        sample_phis = [
            -40, -30, -25, -15, -10, 1, 0.1, 0, 0.1, 1, 10, 15, 25, 30, 40
        ]
        desired_corrections = [
            1.08826179634,
            1.04471392041,
            1.03001804999,
            1.01035010565,
            1.00454429986,
            1.00004502376,
            1.00000045020,
            1.00000000000,
            1.00000045020,
            1.00004502376,
            1.00454429986,
            1.01035010565,
            1.03001804999,
            1.04471392041,
            1.08826179634,
        ]

        actual_corrections = sample_size_correction(
            sample_phis, 0.2, 0.1, 0.3
        )

        np.testing.assert_allclose(actual_corrections, desired_corrections)


class TestSubstrateAbsorptionCorrection(unittest.TestCase):

    def test_values(self):
        sample_phis = [
            -40, -30, -25, -15, -10, 1, 0.1, 0, 0.1, 1, 10, 15, 25, 30, 40
        ]
        desired_corrections = [
            1.0575293661658,
            1.0287386415604,
            1.0191140636150,
            1.0064817551438,
            1.0028293819593,
            1.0000278992983,
            1.0000002789541,
            1.0000000000000,
            1.0000002789541,
            1.0000278992983,
            1.0028293819593,
            1.0064817551438,
            1.0191140636150,
            1.0287386415604,
            1.0575293661658,
        ]

        actual_corrections = substrate_absorption_correction(
            sample_phis, 100, 1/546
        )

        np.testing.assert_allclose(actual_corrections, desired_corrections)
