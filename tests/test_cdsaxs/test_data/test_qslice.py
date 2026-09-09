import numpy as np
import unittest
from unittest.mock import patch

from cdsaxs.data.reduced_data1d import ReducedData1D


class TestIntegratedQSlice(unittest.TestCase):

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

        self.limits_axis0 = (2, 7)
        self.limits_axis1 = (17, 26)

        self.integration_mode = 'sum'
        self.data2d = "test data"

        self.integration_axis = 0
        self.image_roi = np.ones((5, 9), dtype=float)
        self.image_mask = np.isnan(self.image_roi)
        self.background = 1

        self.qslice = ReducedData1D(
            q=self.q,
            Iq=self.Iq,
            q_axis=self.q_axis,
            mask=self.custom_mask,
            data2d=self.data2d,
            limits_axis0=self.limits_axis0,
            limits_axis1=self.limits_axis1,
            integration_mode=self.integration_mode,
            integration_axis=self.integration_axis,
            image_roi=self.image_roi,
            image_mask=self.image_mask,
            background=self.background,
            dIq=self.dIq
            )

    def test_init_params(self):
        np.testing.assert_array_equal(self.qslice.q, self.q)
        np.testing.assert_array_equal(self.qslice.qsx, self.q)
        np.testing.assert_array_equal(self.qslice.Iq, self.Iq)
        np.testing.assert_array_equal(self.qslice.dIq, self.dIq)
        self.assertEqual(self.qslice.q_axis, self.q_axis)
        np.testing.assert_array_equal(self.qslice.mask,
                                      self.custom_mask+np.isnan(self.Iq))

        self.assertEqual(self.qslice.data2d, self.data2d)
        self.assertTupleEqual(self.qslice.limits_axis0,
                              self.limits_axis0)
        self.assertTupleEqual(self.qslice.limits_axis1,
                              self.limits_axis1)
        self.assertEqual(self.qslice.integration_mode, self.integration_mode)
        self.assertEqual(self.qslice.integration_axis,
                         self.integration_axis)
        np.testing.assert_array_equal(self.qslice.image_roi, self.image_roi)
        np.testing.assert_array_equal(self.qslice.image_mask, self.image_mask)
        np.testing.assert_array_equal(self.qslice.background, np.array([self.background]))

    def test_init_bad_inputs(self):

        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=[1],
                q_axis=self.q_axis,
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                integration_mode=self.integration_mode,
                integration_axis=self.integration_axis,
                image_roi=self.image_roi,
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis='no',
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                integration_mode=self.integration_mode,
                integration_axis=self.integration_axis,
                image_roi=self.image_roi,
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis=self.q_axis,
                mask=np.array([True]),
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                integration_mode=self.integration_mode,
                integration_axis=self.integration_axis,
                image_roi=self.image_roi,
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis=self.q_axis,
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=(1, 2),
                limits_axis1=self.limits_axis1,
                integration_mode=self.integration_mode,
                integration_axis=self.integration_axis,
                image_roi=self.image_roi,
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis=self.q_axis,
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=(1, 2),
                integration_mode=self.integration_mode,
                integration_axis=self.integration_axis,
                image_roi=self.image_roi,
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis=self.q_axis,
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                integration_mode='something',
                integration_axis=self.integration_axis,
                image_roi=self.image_roi,
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis=self.q_axis,
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                integration_mode=self.integration_mode,
                integration_axis=3,
                image_roi=self.image_roi,
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis=self.q_axis,
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                integration_mode=self.integration_mode,
                integration_axis=self.integration_axis,
                image_roi=self.image_roi[:-1, :-1],
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis=self.q_axis,
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                integration_mode=self.integration_mode,
                integration_axis=self.integration_axis,
                image_roi=self.image_roi,
                image_mask=self.image_mask[:-1, :-1],
                background=np.array([1, 2]),
                dIq=self.dIq
            )
        with self.assertRaises(ValueError):
            ReducedData1D(
                q=self.q,
                Iq=self.Iq,
                q_axis=self.q_axis,
                mask=self.custom_mask,
                data2d=self.data2d,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                integration_mode=self.integration_mode,
                integration_axis=self.integration_axis,
                image_roi=self.image_roi,
                image_mask=self.image_mask,
                background=self.background,
                dIq=self.dIq[:-1]
            )

    def test_integrated_q_slice_mirror(self):
        with patch("cdsaxs.data.reduced_data1d.Data1D.abs_q") as mock_abs_q:
            result = self.qslice.mirror_q(q_axis="qsx", resort=False)

        self.assertIsNone(result)
        mock_abs_q.assert_called_once_with(q_axis="qsx", resort=False)
