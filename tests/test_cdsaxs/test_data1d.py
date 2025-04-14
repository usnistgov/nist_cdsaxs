import numpy as np
import unittest

from cdsaxs.data1d import Data1D


class TestData1D(unittest.TestCase):

    def setUp(self):
        self.q = np.array(
            [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.01])
        self.Iq = np.array(
            [1, 11, 100, 0, 100, 10, 1000, 1001])
        self.dIq = np.array(
            [0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002])
        self.dq = np.array(
            [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.006, 0.007, 0.008])
        self.q_axis = 'qsx'

        self.data1d = Data1D(self.q, self.Iq, self.q_axis,
                             dIq=self.dIq, dq=self.dq)

    def test_data1d_init_params(self):
        np.testing.assert_array_equal(self.data1d.q, self.q)
        np.testing.assert_array_equal(self.data1d.Iq, self.Iq)
        np.testing.assert_array_equal(self.data1d.dq, self.dq)
        np.testing.assert_array_equal(self.data1d.dIq, self.dIq)
        self.assertEqual(self.data1d.q_axis, self.q_axis)

    def test_data1d_init_lengths(self):
        with self.assertRaises(ValueError):
            Data1D(q=[1, 2, 3], Iq=[1, 2], q_axis='qsx')
            Data1D(q=[1, 2, 3], Iq=[1, 2, 3], q_axis='qsx', dIq=[1])
            Data1D(q=[1, 2, 3], Iq=[1, 2, 3], q_axis='qsx', dq=[1])

    def test_data1d_init_q_axis(self):
        with self.assertRaises(ValueError):
            Data1D(q=[1], Iq=[1], q_axis='not_an_axis')

    def test_data1d_interpolate_linear(self):
        test_interpolated_q = np.array([0.0042, 0.0017])
        test_interpolated_Iq = np.array([20., 8.])

        interpolated_q, interpolated_Iq = self.data1d.interpolate(
            interpolated_q=test_interpolated_q,
            mode='linear'
        )

        np.testing.assert_array_almost_equal(
            interpolated_Iq, test_interpolated_Iq)
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q)


class TestIntegratedQSlice(unittest.TestCase):

    # TODO: implement tests for IntegratedQSlice with additional functionality

    def setUp(self):
        pass
