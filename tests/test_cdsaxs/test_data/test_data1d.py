import numpy as np
import unittest

from cdsaxs._dtypes import REAL_DTYPE
from cdsaxs.data.data1d import Data1D


class TestData1D(unittest.TestCase):

    def setUp(self):
        self.q = np.array(
            [0.001, 0.002, 0.003, 0.004,
             0.005, 0.006, 0.007, 0.01, 0.02]).astype(REAL_DTYPE)
        self.Iq = np.array(
            [1, 11, 100, 0, 100, 10, 1000, 1001, np.nan]).astype(REAL_DTYPE)
        self.dIq = np.array(
            [0.001, 0.001, 0.001, 0.001, 0.002,
             0.002, 0.002, 0.002, np.nan]).astype(REAL_DTYPE)
        self.q_axis = 'qsx'
        self.qsy = np.zeros_like(self.q, dtype=REAL_DTYPE)
        self.custom_mask = np.array([True, False, False, False, False, False,
                                     False, False, False,])

        self.data1d = Data1D(self.q, self.Iq, self.q_axis,
                             dIq=self.dIq, mask=self.custom_mask,
                             qsy=self.qsy)

    def test_data1d_init_params(self):
        np.testing.assert_array_equal(self.data1d.q, self.q)
        np.testing.assert_array_equal(self.data1d.qsx, self.q)
        np.testing.assert_array_equal(self.data1d.Iq, self.Iq)
        np.testing.assert_array_equal(self.data1d.dIq, self.dIq)
        np.testing.assert_array_equal(self.data1d.qsy, self.qsy)
        np.testing.assert_array_equal(
            self.data1d.mask,
            self.custom_mask + np.isnan(self.data1d.Iq))
        self.assertEqual(self.data1d.q_axis, self.q_axis)

    def test_data1d_init_lengths(self):
        with self.assertRaises(ValueError):
            Data1D(q=[1, 2, 3], Iq=[1, 2], q_axis='qsx')
        with self.assertRaises(ValueError):
            Data1D(q=[1, 2, 3], Iq=[1, 2, 3], q_axis='qsx', dIq=[1])
        with self.assertRaises(ValueError):
            Data1D(q=[1, 2, 3], Iq=[1, 2, 3], q_axis='qsx',
                   mask=np.array([True]))
        with self.assertRaises(ValueError):
            Data1D(q=[1, 2, 3], Iq=[1, 2, 3], q_axis='qsx',
                   qsy=np.array([1, 2]))

    def test_data1d_init_q_axis(self):
        with self.assertRaises(ValueError):
            Data1D(q=[1], Iq=[1], q_axis='not_an_axis')

    def test_data1d_interpolate_linear(self):
        test_interpolated_q = np.array([0.0042, 0.0017])
        test_interpolated_Iq = np.array([20., 8.])

        interpolated_q, interpolated_Iq = self.data1d.linear_interpolation(
            q_points=test_interpolated_q,
            mode='linear'
        )

        # remove the last point because it should be filtered out by
        # the interpolation function (it is outside of the data range
        # after the masked points have been removed!)
        np.testing.assert_allclose(
            interpolated_Iq, test_interpolated_Iq[:-1], rtol=1e-6, atol=2e-5)
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q[:-1])

    def test_data1d_interpolate_log(self):
        test_interpolated_q = np.array([0.0042, 0.0017])
        test_interpolated_Iq = np.array([0, 6.2693476668])

        interpolated_q, interpolated_Iq = self.data1d.linear_interpolation(
            q_points=test_interpolated_q,
            mode='log'
        )

        # remove the last point because it should be filtered out by
        # the interpolation function (it is outside of the data range
        # after the masked points have been removed!)
        np.testing.assert_allclose(
            interpolated_Iq, test_interpolated_Iq[:-1], rtol=5e-6, atol=1e-4)
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q[:-1])

    def test_data1d_interpolate_log_q(self):
        test_interpolated_q = np.array([0.0042, 0.0017])
        test_interpolated_Iq = np.array([21.864922326, 8.655347464])

        interpolated_q, interpolated_Iq = self.data1d.linear_interpolation(
            q_points=test_interpolated_q,
            mode='log_q'
        )

        # remove the last point because it should be filtered out by
        # the interpolation function (it is outside of the data range
        # after the masked points have been removed!)
        np.testing.assert_allclose(
            interpolated_Iq, test_interpolated_Iq[:-1], rtol=5e-6, atol=1e-4)
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q[:-1])

    def test_data1d_interpolate_log_Iq(self):
        # unexpected behavior will occur where 0 values occur
        test_interpolated_q = np.array([0.0042, 0.0017])
        test_interpolated_Iq = np.array([0., 5.357657656669])

        interpolated_q, interpolated_Iq = self.data1d.linear_interpolation(
            q_points=test_interpolated_q,
            mode='log_Iq'
        )

        np.testing.assert_array_almost_equal(
            interpolated_Iq, test_interpolated_Iq[:-1])
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q[:-1])

    def test_scale_data_value(self):
        scale_value = 20.2
        test_scaled_Iq = np.array([
            20.2, 222.2, 2020, 0, 2020, 202, 20200, 20220.2, np.nan
        ])

        self.data1d.scale_data(scale_value)

        np.testing.assert_allclose(
            self.data1d.Iq, test_scaled_Iq, rtol=1e-7, atol=2e-3
        )
        self.assertTupleEqual(
            self.data1d._data_transformations[0],
            ("scale", scale_value)
        )

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)
        self.assertListEqual(self.data1d._data_transformations, [])

    def test_scale_data_array(self):
        scale_value = np.array([1, 1, 1, 1, 20.2, 20.2, 20.2, 20.2, 1])
        test_scaled_Iq = np.array([
            1, 11, 100, 0, 2020, 202, 20200, 20220.2, np.nan
        ])

        self.data1d.scale_data(scale_value)

        np.testing.assert_array_almost_equal(
            self.data1d.Iq, test_scaled_Iq
        )
        self.assertTupleEqual(
            self.data1d._data_transformations[0],
            ("scale", scale_value)
        )

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)
        self.assertListEqual(self.data1d._data_transformations, [])

    def test_normalize_data_value(self):
        norm_value = 20.2
        test_scaled_Iq = np.array([
            0.049504950, 0.544554455, 4.950495050, 0.00,
            4.950495050, 0.495049505, 49.504950495, 49.554455446, np.nan
        ])

        self.data1d.normalize_data(norm_value)

        np.testing.assert_array_almost_equal(
            self.data1d.Iq, test_scaled_Iq
        )
        self.assertTupleEqual(
            self.data1d._data_transformations[0],
            ("normalize", norm_value)
        )

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)
        self.assertListEqual(self.data1d._data_transformations, [])

    def test_normalize_data_array(self):
        norm_value = np.array([1, 1, 1, 1, 20.2, 20.2, 20.2, 20.2, 1])
        test_scaled_Iq = np.array([
            1, 11, 100, 0,
            4.950495050, 0.495049505, 49.504950495, 49.554455446, np.nan
        ])

        self.data1d.normalize_data(norm_value)

        np.testing.assert_array_almost_equal(
            self.data1d.Iq, test_scaled_Iq
        )
        print(self.data1d._data_transformations)
        self.assertTupleEqual(
            self.data1d._data_transformations[0],
            ("normalize", norm_value)
        )

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)
        self.assertListEqual(self.data1d._data_transformations, [])

    def test_subtract_data_value(self):
        value = 5.
        test_scaled_Iq = np.array([
            -4, 6, 95, -5, 95, 5, 995, 996, np.nan
        ])

        self.data1d.subtract_from_data(value)

        np.testing.assert_array_almost_equal(
            self.data1d.Iq, test_scaled_Iq
        )
        self.assertTupleEqual(
            self.data1d._data_transformations[0],
            ("subtract", value)
        )

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)
        self.assertListEqual(self.data1d._data_transformations, [])

    def test_subtract_data_array(self):
        value = np.array([0, 0, 0, 0, 5, 5, 5, 5, 1])
        test_scaled_Iq = np.array([
            1, 11, 100, 0, 95, 5, 995, 996, np.nan
        ])

        self.data1d.subtract_from_data(value)

        np.testing.assert_array_almost_equal(
            self.data1d.Iq, test_scaled_Iq
        )
        print(self.data1d._data_transformations)
        self.assertTupleEqual(
            self.data1d._data_transformations[0],
            ("subtract", value)
        )

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)
        self.assertListEqual(self.data1d._data_transformations, [])

    def test_add_data_value(self):
        value = 5.
        test_scaled_Iq = np.array([
            6, 16, 105, 5, 105, 15, 1005, 1006, np.nan
        ])

        self.data1d.add_to_data(value)

        np.testing.assert_array_almost_equal(
            self.data1d.Iq, test_scaled_Iq
        )
        self.assertTupleEqual(
            self.data1d._data_transformations[0],
            ("add", value)
        )

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)
        self.assertListEqual(self.data1d._data_transformations, [])

    def test_add_data_array(self):
        value = np.array([0, 0, 0, 0, 5, 5, 5, 5, 1])
        test_scaled_Iq = np.array([
            1, 11, 100, 0, 105, 15, 1005, 1006, np.nan
        ])

        self.data1d.add_to_data(value)

        np.testing.assert_array_almost_equal(
            self.data1d.Iq, test_scaled_Iq
        )
        print(self.data1d._data_transformations)
        self.assertTupleEqual(
            self.data1d._data_transformations[0],
            ("add", value)
        )

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)
        self.assertListEqual(self.data1d._data_transformations, [])

    def test_reset_data_transformations(self):

        value = 5

        self.data1d.scale_data(value)
        self.assertRaises(AssertionError, np.testing.assert_almost_equal,
                          self.data1d.Iq, self.Iq)

        self.data1d.add_to_data(value*100)
        self.assertRaises(AssertionError, np.testing.assert_almost_equal,
                          self.data1d.Iq, self.Iq)
        self.data1d.normalize_data(value/2)
        self.assertRaises(AssertionError, np.testing.assert_almost_equal,
                          self.data1d.Iq, self.Iq)
        self.data1d.subtract_from_data(value*1.06)
        self.assertRaises(AssertionError, np.testing.assert_almost_equal,
                          self.data1d.Iq, self.Iq)

        self.data1d.reset_data_transformations()
        np.testing.assert_array_almost_equal(self.data1d.Iq, self.Iq)

    def test_abs_q(self):
        q = np.array([-1, -2, 0, 0.5, 3, 2.5])
        Iq = np.array([10, 20, 100, 200, 300, np.nan])
        qsy = np.array([0, 0, 1, 1, 1, 0])

        data1d = Data1D(q, Iq, "qsx", qsy=qsy)
        data1d.abs_q()

        np.testing.assert_array_equal(
            data1d.q,
            np.array([0, 0.5, 1, 2, 2.5, 3])
            )
        np.testing.assert_array_equal(
            data1d.qsx,
            np.array([0, 0.5, 1, 2, 2.5, 3])
            )
        np.testing.assert_array_equal(
            data1d.Iq,
            np.array([100, 200, 10, 20, np.nan, 300])
            )
        np.testing.assert_array_equal(
            data1d.qsy,
            np.array([1, 1, 0, 0, 0, 1])
            )

    def test_abs_q_no_resort(self):
        q = np.array([-1, -2, 0, 0.5, 3, 2.5])
        Iq = np.array([10, 20, 100, 200, 300, np.nan])
        qsy = np.array([0, 0, 1, 1, 1, 0])

        data1d = Data1D(q, Iq, "qsx", qsy=qsy)
        data1d.abs_q(resort=False)

        np.testing.assert_array_equal(
            data1d.q,
            np.abs(q)
            )
        np.testing.assert_array_equal(
            data1d.qsx,
            np.abs(q)
            )
        np.testing.assert_array_equal(
            data1d.Iq,
            Iq
            )
        np.testing.assert_array_equal(
            data1d.qsy,
            qsy
            )

    def test_abs_q_different_axis(self):
        qsy = np.array([-1, -2, 0, 0.5, 3, 2.5])
        Iq = np.array([10, 20, 100, 200, 300, np.nan])
        q = np.array([0, 0, 1, 1, 1, 0])

        data1d = Data1D(q, Iq, "qsx", qsy=qsy)
        data1d.abs_q(q_axis="qsy")

        np.testing.assert_array_equal(
            data1d.q,
            np.array([1, 1, 0, 0, 0, 1])
            )
        np.testing.assert_array_equal(
            data1d.qsx,
            np.array([1, 1, 0, 0, 0, 1])
            )
        np.testing.assert_array_equal(
            data1d.Iq,
            np.array([100, 200, 10, 20, np.nan, 300])
            )
        np.testing.assert_array_equal(
            data1d.qsy,
            np.array([0, 0.5, 1, 2, 2.5, 3])
            )

    def test_mask_points(self):
        qsy = np.array([-1, -2, 0, 0.5, 3, 2.5])
        Iq = np.array([10, 20, 100, 200, 300, np.nan])
        q = np.array([0, 0, 1, 1, 1, 0])

        data1d = Data1D(q, Iq, "qsx", qsy=qsy)

        data1d.mask_points(Iq < 100)

        np.testing.assert_array_equal(
            data1d.mask,
            np.array([True, True, False, False, False, True])
        )

    def test_overwrite_mask(self):
        qsy = np.array([-1, -2, 0, 0.5, 3, 2.5])
        Iq = np.array([10, 20, 100, 200, 300, np.nan])
        q = np.array([0, 0, 1, 1, 1, 0])

        data1d = Data1D(q, Iq, "qsx", qsy=qsy)

        data1d.overwrite_mask(Iq < 100)
        np.testing.assert_array_equal(
            data1d.mask,
            np.array([True, True, False, False, False, False])
        )

    def test_reset_mask(self):
        qsy = np.array([-1, -2, 0, 0.5, 3, 2.5])
        Iq = np.array([10, 20, 100, 200, 300, np.nan])
        q = np.array([0, 0, 1, 1, 1, 0])

        data1d = Data1D(q, Iq, "qsx", qsy=qsy)

        data1d.overwrite_mask(Iq < 100)
        data1d.reset_mask()

        np.testing.assert_array_equal(
            data1d.mask,
            np.array([False, False, False, False, False, True])
        )
