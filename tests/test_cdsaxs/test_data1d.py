import numpy as np
import unittest

from cdsaxs.data1d import Data1D, IntegratedQSlice


class TestData1D(unittest.TestCase):

    def setUp(self):
        self.q = np.array(
            [0.001, 0.002, 0.003, 0.004,
             0.005, 0.006, 0.007, 0.01]).astype(float)
        self.Iq = np.array(
            [1, 11, 100, 0, 100, 10, 1000, 1001]).astype(float)
        self.dIq = np.array(
            [0.001, 0.001, 0.001, 0.001, 0.002,
             0.002, 0.002, 0.002]).astype(float)
        self.dq = np.array(
            [0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
             0.006, 0.007, 0.008]).astype(float)
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

        interpolated_q, interpolated_Iq = self.data1d.linear_interpolation(
            interpolated_q_points=test_interpolated_q,
            mode='linear'
        )

        np.testing.assert_array_almost_equal(
            interpolated_Iq, test_interpolated_Iq)
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q)

    def test_data1d_interpolate_log(self):
        test_interpolated_q = np.array([0.0042, 0.0017])
        test_interpolated_Iq = np.array([0, 6.2693476668])

        interpolated_q, interpolated_Iq = self.data1d.linear_interpolation(
            interpolated_q_points=test_interpolated_q,
            mode='log'
        )

        np.testing.assert_array_almost_equal(
            interpolated_Iq, test_interpolated_Iq)
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q)

    def test_data1d_interpolate_log_q(self):
        test_interpolated_q = np.array([0.0042, 0.0017])
        test_interpolated_Iq = np.array([21.864922326, 8.655347464])

        interpolated_q, interpolated_Iq = self.data1d.linear_interpolation(
            interpolated_q_points=test_interpolated_q,
            mode='log_q'
        )

        np.testing.assert_array_almost_equal(
            interpolated_Iq, test_interpolated_Iq)
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q)

    def test_data1d_interpolate_log_Iq(self):
        # unexpected behavior will occur where 0 values occur
        test_interpolated_q = np.array([0.0042, 0.0017])
        test_interpolated_Iq = np.array([0., 5.357657656669])

        interpolated_q, interpolated_Iq = self.data1d.linear_interpolation(
            interpolated_q_points=test_interpolated_q,
            mode='log_Iq'
        )

        np.testing.assert_array_almost_equal(
            interpolated_Iq, test_interpolated_Iq)
        np.testing.assert_array_almost_equal(
            interpolated_q, test_interpolated_q)

    def test_scale_data_value(self):
        scale_value = 20.2
        test_scaled_Iq = np.array([
            20.2, 222.2, 2020, 0, 2020, 202, 20200, 20220.2
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

    def test_scale_data_array(self):
        scale_value = np.array([1, 1, 1, 1, 20.2, 20.2, 20.2, 20.2])
        test_scaled_Iq = np.array([
            1, 11, 100, 0, 2020, 202, 20200, 20220.2
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
            4.950495050, 0.495049505, 49.504950495, 49.554455446
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
        norm_value = np.array([1, 1, 1, 1, 20.2, 20.2, 20.2, 20.2])
        test_scaled_Iq = np.array([
            1, 11, 100, 0, 4.950495050, 0.495049505, 49.504950495, 49.554455446
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
            -4, 6, 95, -5, 95, 5, 995, 996
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
        value = np.array([0, 0, 0, 0, 5, 5, 5, 5])
        test_scaled_Iq = np.array([
            1, 11, 100, 0, 95, 5, 995, 996
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
            6, 16, 105, 5, 105, 15, 1005, 1006
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
        value = np.array([0, 0, 0, 0, 5, 5, 5, 5])
        test_scaled_Iq = np.array([
            1, 11, 100, 0, 105, 15, 1005, 1006
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


class TestIntegratedQSlice(unittest.TestCase):

    def setUp(self):
        self.q = np.array(
            [-0.2, -0.1, -0.007, -0.009,
             0.005, 0.006, 0.008, 0.01]).astype(float)
        self.Iq = np.array(
            [1, 11, 100, 0, 100, 10, 1000, 1001]).astype(float)
        self.dIq = np.array(
            [0.001, 0.001, 0.001, 0.001, 0.002,
             0.002, 0.002, 0.002]).astype(float)
        self.dq = np.array(
            [0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
             0.006, 0.007, 0.008]).astype(float)
        self.q_axis = 'qsx'

        self.limits_axis0 = (2, 7)
        self.limits_axis1 = (17, 25)

        self.mode = 'sum'
        self.name = "test data"

        self.integration_axis = 0

        self.rotation_angle = 5
        self.rotation_center = [50, 50]
        self.rotation_sampling_mode = 'nearest'
        self.rotated_image = np.ones([100, 200]).astype(float)

        self.integrated_q_slice = IntegratedQSlice(
            self.q,
            self.Iq,
            self.q_axis,
            dIq=self.dIq,
            dq=self.dq,
            name=self.name,
            limits_axis0=self.limits_axis0,
            limits_axis1=self.limits_axis1,
            mode=self.mode,
            integration_axis=self.integration_axis,
            rotation_angle=self.rotation_angle,
            rotation_sampling_mode=self.rotation_sampling_mode,
            rotation_center=self.rotation_center,
            rotated_image=self.rotated_image,
            )

    def test_init_params(self):
        np.testing.assert_array_equal(self.integrated_q_slice.q, self.q)
        np.testing.assert_array_equal(self.integrated_q_slice.Iq, self.Iq)
        np.testing.assert_array_equal(self.integrated_q_slice.dq, self.dq)
        np.testing.assert_array_equal(self.integrated_q_slice.dIq, self.dIq)
        self.assertEqual(self.integrated_q_slice.q_axis, self.q_axis)

        self.assertEqual(self.integrated_q_slice.name, self.name)
        self.assertTupleEqual(self.integrated_q_slice.limits_axis0,
                              self.limits_axis0)
        self.assertTupleEqual(self.integrated_q_slice.limits_axis1,
                              self.limits_axis1)
        self.assertEqual(self.integrated_q_slice.mode, self.mode)
        self.assertEqual(self.integrated_q_slice.integration_axis,
                         self.integration_axis)
        self.assertEqual(self.integrated_q_slice.rotation_angle,
                         self.rotation_angle)
        self.assertEqual(self.integrated_q_slice.rotation_sampling_mode,
                         self.rotation_sampling_mode)
        self.assertListEqual(self.integrated_q_slice.rotation_center,
                             self.rotation_center)
        np.testing.assert_array_equal(self.integrated_q_slice.rotated_image,
                                      self.rotated_image)

    def test_init_bad_inputs(self):

        with self.assertRaises(ValueError):
            # bad limits
            IntegratedQSlice(
                self.q,
                self.Iq,
                self.q_axis,
                name=self.name,
                limits_axis0=(0, 1, 2),
                limits_axis1=self.limits_axis1,
                mode=self.mode,
                integration_axis=self.integration_axis,
                rotation_angle=self.rotation_angle,
                rotation_sampling_mode=self.rotation_sampling_mode,
                rotation_center=self.rotation_center,
                rotated_image=self.rotated_image,
                )
            IntegratedQSlice(
                self.q,
                self.Iq,
                self.q_axis,
                name=self.name,
                limits_axis0=self.limits_axis0,
                limits_axis1=(0, 1, 2),
                mode=self.mode,
                integration_axis=self.integration_axis,
                rotation_angle=self.rotation_angle,
                rotation_sampling_mode=self.rotation_sampling_mode,
                rotation_center=self.rotation_center,
                rotated_image=self.rotated_image,
                )

            # unknown mode
            IntegratedQSlice(
                self.q,
                self.Iq,
                self.q_axis,
                name=self.name,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                mode="random test",
                integration_axis=self.integration_axis,
                rotation_angle=self.rotation_angle,
                rotation_sampling_mode=self.rotation_sampling_mode,
                rotation_center=self.rotation_center,
                rotated_image=self.rotated_image,
                )

            # mismatched limits length with non-integrated axis
            IntegratedQSlice(
                self.q,
                self.Iq,
                self.q_axis,
                name=self.name,
                limits_axis0=self.limits_axis0,
                limits_axis1=[0, 10],
                mode=self.mode,
                integration_axis=self.integration_axis,
                rotation_angle=self.rotation_angle,
                rotation_sampling_mode=self.rotation_sampling_mode,
                rotation_center=self.rotation_center,
                rotated_image=self.rotated_image,
                )
            IntegratedQSlice(
                self.q,
                self.Iq,
                self.q_axis,
                name=self.name,
                limits_axis0=[0, 10],
                limits_axis1=self.limits_axis1,
                mode=self.mode,
                integration_axis=1,
                rotation_angle=self.rotation_angle,
                rotation_sampling_mode=self.rotation_sampling_mode,
                rotation_center=self.rotation_center,
                rotated_image=self.rotated_image,
                )

            # unknown integration axis
            IntegratedQSlice(
                self.q,
                self.Iq,
                self.q_axis,
                name=self.name,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                mode=self.mode,
                integration_axis="three",
                rotation_angle=self.rotation_angle,
                rotation_sampling_mode=self.rotation_sampling_mode,
                rotation_center=self.rotation_center,
                rotated_image=self.rotated_image,
                )

            # missing rotated image
            IntegratedQSlice(
                self.q,
                self.Iq,
                self.q_axis,
                name=self.name,
                limits_axis0=self.limits_axis0,
                limits_axis1=self.limits_axis1,
                mode=self.mode,
                integration_axis=self.integration_axis,
                rotation_angle=self.rotation_angle,
                rotation_sampling_mode=self.rotation_sampling_mode,
                rotation_center=self.rotation_center,
                )
    
    def test_integrated_q_slice_mirror(self):
        
        mirror_q = np.array([
            0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.1, 0.2
        ])
        mirror_Iq = np.array([
            100, 10, 100, 1000, 0, 1001, 11, 1
        ])
        mirror_dIq = np.array([
            0.002, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.001
        ])
        mirror_dq = np.array([
            0.0005, 0.006, 0.0003, 0.007, 0.0004, 0.008, 0.0002, 0.0001
        ])

        self.integrated_q_slice.mirror_q()

        np.testing.assert_array_almost_equal(
            self.integrated_q_slice.q,
            mirror_q
        )
        np.testing.assert_array_almost_equal(
            self.integrated_q_slice.Iq,
            mirror_Iq
        )

        np.testing.assert_array_almost_equal(
            self.integrated_q_slice.dIq,
            mirror_dIq
        )
        np.testing.assert_array_almost_equal(
            self.integrated_q_slice.dq,
            mirror_dq
        )
        np.testing.assert_array_almost_equal(
            self.integrated_q_slice._data_before_mirror[0],
            self.q
        )
        np.testing.assert_array_almost_equal(
            self.integrated_q_slice._data_before_mirror[1],
            self.Iq
        )
        np.testing.assert_array_almost_equal(
            self.integrated_q_slice._data_before_mirror[2],
            self.dIq
        )
        np.testing.assert_array_almost_equal(
            self.integrated_q_slice._data_before_mirror[3],
            self.dq
        )

        self.integrated_q_slice.reset_mirrored_q()

    def test_integrated_q_slice_mirror_reset(self):

        self.integrated_q_slice.mirror_q()
        self.integrated_q_slice.reset_mirrored_q()

        np.testing.assert_array_equal(
            self.integrated_q_slice.q, self.q
        )
        np.testing.assert_array_equal(
            self.integrated_q_slice.Iq, self.Iq
        )
        np.testing.assert_array_equal(
            self.integrated_q_slice.dIq, self.dIq
        )
        np.testing.assert_array_equal(
            self.integrated_q_slice.dq, self.dq
        )