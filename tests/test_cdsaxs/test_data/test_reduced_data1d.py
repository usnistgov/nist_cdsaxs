import numpy as np
import unittest

from cdsaxs.data.reduced_data1d import ReducedData1D


class TestReducedData1D(unittest.TestCase):

    def setUp(self):
        self.q = np.array(
            [0.001, 0.002, 0.003, 0.004,
             0.005, 0.006, 0.007, 0.01, 0.02]).astype(float)
        self.Iq = np.array(
            [1, 11, 100, 0, 100, 10, 1000, 1001, np.nan]).astype(float)
        self.dIq = np.array(
            [0.001, 0.001, 0.001, 0.001, 0.002,
             0.002, 0.002, 0.002, np.nan]).astype(float)
        self.q_axis = 'qsx'
        self.qsy = np.ones_like(self.q, dtype=float)*0
        self.custom_mask = np.array([True, False, False, False, False, False,
                                     False, False, False,])
        self.wavelength_nm = 0.1
        self.sample_phi_deg = 2
        self.sample_chi_deg = 3
        self.sample_omega_deg = 4

        self.reduced_data1d = ReducedData1D(
            q=self.q,
            Iq=self.Iq,
            q_axis=self.q_axis,
            dIq=self.dIq,
            mask=self.custom_mask,
            wavelength_nm=self.wavelength_nm,
            sample_chi_deg=self.sample_chi_deg,
            sample_omega_deg=self.sample_omega_deg,
            sample_phi_deg=self.sample_phi_deg
        )

    def test_data1d_init_params_unique_to_reduced(self):
        self.assertEqual(self.reduced_data1d.wavelength_nm,
                         self.wavelength_nm)
        self.assertEqual(self.reduced_data1d.sample_chi_deg,
                         self.sample_chi_deg)
        self.assertEqual(self.reduced_data1d.sample_omega_deg,
                         self.sample_omega_deg)
        self.assertEqual(self.reduced_data1d.sample_phi_deg,
                         self.sample_phi_deg)
