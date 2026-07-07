import numpy as np
import unittest
from unittest.mock import patch

from cdsaxs.calculators import wavelength_to_energy
from cdsaxs.data.data2d import Data2D, combine_data2d


class TestCombineData2D(unittest.TestCase):
    def _make_metadata(self, exposure_time_s=None):
        metadata = {
            'center_px': (1, 1),
            'wavelength_nm': 0.07,
            'pixel_size_um': 172,
            'sdd_cm': 542,
            'sample_phi_deg': 0,
            'sample_phi_offset_deg': 0,
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
        }
        if exposure_time_s is not None:
            metadata['exposure_time_s'] = exposure_time_s
        return metadata

    def _make_data(self, name, image, mask=None, exposure_time_s=None, **user_params):
        return Data2D(
            image=np.array(image, dtype=np.float64),
            name=name,
            mask=np.array(mask, dtype=bool) if mask is not None else None,
            hide_q_warnings=True,
            **self._make_metadata(exposure_time_s=exposure_time_s),
            **user_params,
        )

    def test_combine_data2d_sums_images_unions_masks_and_exposure_time(self):
        data1 = self._make_data(
            'first',
            image=[[1.0, np.nan], [3.0, 4.0]],
            mask=[[False, True], [False, False]],
            exposure_time_s=2,
            sample_id='A',
        )
        data2 = self._make_data(
            'second',
            image=[[10.0, 20.0], [30.0, 40.0]],
            mask=[[False, False], [True, False]],
            exposure_time_s=5,
            sample_id='B',
        )

        combined = combine_data2d(data1, data2)

        np.testing.assert_array_equal(
            combined.image,
            np.array([[11.0, 20.0], [33.0, 44.0]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            combined.mask,
            np.array([[False, True], [True, False]], dtype=bool),
        )
        self.assertEqual(combined.metadata['exposure_time_s'], 7)
        self.assertEqual(combined.user_params['sample_id'], 'A')
        self.assertEqual(combined.name, 'first')

    def test_combine_data2d_uses_explicit_name_override(self):
        data1 = self._make_data('first', image=np.ones((3, 3)), exposure_time_s=1)
        data2 = self._make_data('second', image=np.full((3, 3), 2.0), exposure_time_s=2)

        combined = combine_data2d(data1, data2, name='combined-data')

        self.assertEqual(combined.name, 'combined-data')

    def test_combine_data2d_keeps_first_metadata_when_exposure_missing(self):
        data1 = self._make_data('first', image=np.ones((3, 3)), exposure_time_s=3)
        data2 = self._make_data('second', image=np.full((3, 3), 2.0))

        combined = combine_data2d(data1, data2)

        self.assertEqual(combined.metadata['exposure_time_s'], 3)

    def test_combine_data2d_raises_for_mismatched_image_shapes(self):
        data1 = self._make_data('first', image=np.ones((3, 3)))
        data2 = self._make_data('second', image=np.ones((2, 3)))

        with self.assertRaisesRegex(ValueError, 'All Data2D images must have the same shape'):
            combine_data2d(data1, data2)


class TestData2D(unittest.TestCase):

    def setUp(self):
        self.image = np.array(
            [[44386., 20215., 48676., 20338.],
             [64811., 49702., 45711., 77210.],
             [ 4701.,  9052., 61100., 46325.],
             [60019., 70069.,   np.nan, 65941.],
             [18439., 82922., 99999., 91001.],
             [10592., 19847., 10893., 41683.],
             [15954.,  3109., 35295., 61517.]]).astype(np.float64)
        self.custom_mask = np.isnan(self.image)
        self.custom_mask[0, 0] = True
        self.custom_mask[3, 2] = False

        self.metadata = {
            'center_px': (4, 2),
            'wavelength_nm': 0.07,
            'pixel_size_um': 172,
            'sdd_cm': 542,
            'sample_phi_deg': 0,
            'sample_phi_offset_deg': 0,
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
        }
        self.user_params = {
            'custom_param': 20
        }

        self.dataqdyqdx = Data2D(
            image=self.image, name='test data', mask=self.custom_mask,
            hide_q_warnings=True, **self.metadata, **self.user_params)

        self.metadata["energy_ev"] = wavelength_to_energy(
            self.metadata["wavelength_nm"])

    def test_init(self):
        np.testing.assert_array_equal(self.dataqdyqdx.image, self.image)
        np.testing.assert_array_equal(self.dataqdyqdx.mask,
                                      self.custom_mask + np.isnan(self.image))

    def test_metadata(self):
        key = 'center_px'
        self.assertTupleEqual(self.dataqdyqdx.metadata[key], self.metadata[key])

        key = 'wavelength_nm'
        self.assertEqual(self.dataqdyqdx.metadata[key], self.metadata[key])

        key = "energy_ev"
        self.assertAlmostEqual(self.dataqdyqdx.metadata[key], self.metadata[key])

        key = 'sdd_cm'
        self.assertEqual(self.dataqdyqdx.metadata[key], self.metadata[key])

        key = 'sample_phi_offset_deg'
        self.assertEqual(self.dataqdyqdx.metadata[key], 0)

    def test_user_params(self):
        key = 'custom_param'
        self.assertEqual(self.dataqdyqdx.user_params[key], self.user_params[key])

    def test_update_metadata(self):
        self.dataqdyqdx.update_metadata({'sample_phi_deg': 20}, hide_q_warnings=True)
        key = 'sample_phi_deg'
        self.assertEqual(self.dataqdyqdx.metadata[key], 20)

    def test_update_metadata_overwrite(self):
        self.dataqdyqdx.update_metadata({'wavelength_nm': 0.1},
                                        overwrite=True, hide_q_warnings=False)
        self.assertEqual(self.dataqdyqdx.metadata['wavelength_nm'], 0.1)
        self.assertEqual(self.dataqdyqdx.metadata['energy_ev'],
                         wavelength_to_energy(0.1))

    def test_update_metadata_dont_overwrite(self):
        self.dataqdyqdx.update_metadata({'wavelength_nm': 0.1},
                                        overwrite=False, hide_q_warnings=False)
        self.assertEqual(self.dataqdyqdx.metadata['wavelength_nm'],
                         self.metadata['wavelength_nm'])
        self.assertEqual(self.dataqdyqdx.metadata['energy_ev'],
                         self.metadata['energy_ev'])

    def test_update_metadata_not_a_keyword(self):
        with self.assertRaises(ValueError):
            self.dataqdyqdx.update_metadata({'not_a_keyword': 2},
                                            hide_q_warnings=True)
            
    def test_update_user_params(self):
        key = 'new_param'
        self.dataqdyqdx.update_user_params({key: 10})
        self.assertEqual(self.dataqdyqdx.user_params[key], 10)
    
    def test_update_user_params_overwrite(self):
        key = 'custom_param'
        self.dataqdyqdx.update_user_params({key: 10}, overwrite=True)
        self.assertEqual(self.dataqdyqdx.user_params[key], 10)

    def test_update_user_params_dont_overwrite(self):
        key = 'custom_param'
        self.dataqdyqdx.update_user_params({key: 10}, overwrite=False)
        self.assertEqual(self.dataqdyqdx.user_params[key],
                         self.user_params[key])

    def test_dataqdyqdx_calculate_q(self):
        """
        we are using qb for the main detector q in preparation of ALS data
        we also now create 2d q components so we need to reference
        the 1d components that we use for the plots that form the
        crosshairs of the beam center position
        """
        qdy = np.array([0.001139386259, 0.000854539696, 0.000569693132,
                        0.000284846566, 0., -0.000284846566,
                        -0.000569693132])
        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, qdy, decimal=8)

        qdx = np.array([0.000569693132, 0.000284846566, 0., -0.000284846566])
        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, qdx, decimal=8)

    def test_data2d_rotate_ccw90(self):
        """test 90 degree rotation counter-clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   np.nan, 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.001139386259, 0.000854539696,
                                  0.000569693132, 0.000284846566, 0.,
                                  -0.000284846566, -0.000569693132])
        expected_qys = np.array([0.000284846566, 0., -0.000284846566,
                                 -0.000569693132])
        expected_center_px = (1, 4)

        self.dataqdyqdx.rotate_image_ccw(1)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "wrong image.")

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The counter-clockwise 90 degree rotation "
                              "resulted in the wrong center pixel location.")

    def test_data2d_rotate_ccw180(self):
        """test 180 degree rotation counter-clockwise"""
        expected_image = np.array(
            [[61517., 35295.,  3109., 15954.],
             [41683., 10893., 19847., 10592.],
             [91001., 99999., 82922., 18439.],
             [65941.,   np.nan, 70069., 60019.],
             [46325., 61100.,  9052.,  4701.],
             [77210., 45711., 49702., 64811.],
             [20338., 48676., 20215., 44386.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.000284846566, 0., -0.000284846566,
                                 -0.000569693132])
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132,
                                 -0.000854539696, -0.001139386259]
                                )

        expected_center_px = (2, 1)

        self.dataqdyqdx.rotate_image_ccw(2)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The counter-clockwise 180 degree rotation "
                              "resulted in the wrong center pixel location.")

    def test_data2d_rotate_ccw270(self):
        """test 270 degree rotation counter-clockwise"""
        expected_image = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   np.nan, 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.000569693132, 0.000284846566, 0.,
                                  -0.000284846566, -0.000569693132,
                                  -0.000854539696, -0.001139386259])
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566]
                                )
        expected_center_px = (2, 2)

        self.dataqdyqdx.rotate_image_ccw(3)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The counter-clockwise 182700 degree rotation"
                              " resulted in the wrong center pixel location.")

    def test_data2d_rotate_ccw_m90(self):
        """test -90 degree rotation counter-clockwise (90 degree clockwise)"""
        expected_image = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   np.nan, 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132,
                                 -0.000854539696, -0.001139386259])
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566]
                                )
        expected_center_px = (2, 2)

        self.dataqdyqdx.rotate_image_ccw(-1)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The counter-clockwise -90 degree rotation "
                              "resulted in the wrong center pixel location.")

    def test_data2d_rotate_cw180(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[61517., 35295.,  3109., 15954.],
             [41683., 10893., 19847., 10592.],
             [91001., 99999., 82922., 18439.],
             [65941.,   np.nan, 70069., 60019.],
             [46325., 61100.,  9052.,  4701.],
             [77210., 45711., 49702., 64811.],
             [20338., 48676., 20215., 44386.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.000284846566, 0., -0.000284846566,
                                 -0.000569693132])
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132,
                                 -0.000854539696, -0.001139386259]
                                )

        expected_center_px = (2, 1)

        self.dataqdyqdx.rotate_image_ccw(-2)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The clockwise 180 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The clockwise 180 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The clockwise 180 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The clockwise 180 degree rotation "
                              "resulted in the wrong center pixel location.")

    def test_data2d_rotate_cw270(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   np.nan, 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.001139386259, 0.000854539696,
                                  0.000569693132, 0.000284846566, 0.,
                                  -0.000284846566, -0.000569693132])
        expected_qys = np.array([0.000284846566, 0., -0.000284846566,
                                 -0.000569693132])
        expected_center_px = (1, 4)

        self.dataqdyqdx.rotate_image_ccw(-3)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The clockwise 270 degree rotation resulted in the "
            "wrong image.")

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The clockwise 270 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The clockwise 270 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The clockwise 270 degree rotation "
                              "resulted in the wrong center pixel location.")

    def test_data2d_rotate_cw630(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   np.nan, 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.001139386259, 0.000854539696,
                                  0.000569693132, 0.000284846566, 0.,
                                  -0.000284846566, -0.000569693132])
        expected_qys = np.array([0.000284846566, 0., -0.000284846566,
                                 -0.000569693132])
        expected_center_px = (1, 4)

        self.dataqdyqdx.rotate_image_ccw(-7)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The clockwise 630 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The clockwise 630 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The clockwise 630 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The clockwise 630 degree rotation "
                              "resulted in the wrong center pixel location.")

    def test_flip_horizontally(self):
        expected_image = np.array(
            [[20338., 48676., 20215., 44386.],
             [77210., 45711., 49702., 64811.],
             [46325., 61100.,  9052.,  4701.],
             [65941.,   np.nan, 70069., 60019.],
             [91001., 99999., 82922., 18439.],
             [41683., 10893., 19847., 10592.],
             [61517., 35295.,  3109., 15954.]]
        ).astype(np.float64)
        expected_qys = np.array([0.001139386259, 0.000854539696,
                                 0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132])
        expected_qxzs = np.array([0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132])

        expected_center_px = (4, 1)

        self.dataqdyqdx.flip_horizontally()

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The horizontal flip resulted in the wrong image."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The horizontal flip resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The horizontal flip resulted in the"
            " incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The horizontal flip "
                              "resulted in the wrong center pixel location.")

    def test_flip_vertically(self):
        expected_image = np.array(
            [[15954.,  3109., 35295., 61517.],
             [10592., 19847., 10893., 41683.],
             [18439., 82922., 99999., 91001.],
             [60019., 70069.,   np.nan, 65941.],
             [ 4701.,  9052., 61100., 46325.],
             [64811., 49702., 45711., 77210.],
             [44386., 20215., 48676., 20338.]]
        ).astype(np.float64)
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132,
                                 -0.000854539696, -0.001139386259])
        expected_qxzs = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566])

        expected_center_px = (2, 2)

        self.dataqdyqdx.flip_vertically()

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_image,
            err_msg="The vertical flip resulted in the wrong image."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qbx_1d, expected_qxzs, decimal=8,
            err_msg="The horizontal flip resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qby_1d, expected_qys, decimal=8,
            err_msg="The horizontal flip resulted in the"
            " incorrect qxzs array."
        )

        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'],
                              expected_center_px,
                              msg="The horizontal flip "
                              "resulted in the wrong center pixel location.")

    def test_data2d_reset_image_orientation(self):
        """test the reset_image"""

        self.dataqdyqdx.rotate_image_ccw(1)
        self.dataqdyqdx.rotate_image_ccw(1)
        self.dataqdyqdx.flip_horizontally()
        self.dataqdyqdx.rotate_image_ccw(-2)
        self.dataqdyqdx.scale_data(20)
        self.dataqdyqdx.flip_vertically()
        self.dataqdyqdx.rotate_image_ccw(3)
        self.dataqdyqdx.rotate_image_ccw(3)
        self.dataqdyqdx.flip_horizontally()
        self.dataqdyqdx.flip_vertically()
        self.dataqdyqdx.rotate_image_ccw(-1)

        self.dataqdyqdx.reset_image()
        np.testing.assert_array_equal(
            self.dataqdyqdx.image, self.image)
        self.assertTupleEqual(self.dataqdyqdx.metadata['center_px'], (0, 0))

    def test_add_to_data_value(self):
        value = 2
        new_image = np.array([
            [44388, 20217, 48678, 20340],
            [64813, 49704, 45713, 77212],
            [4703, 9054, 61102, 46327],
            [60021, 70071, np.nan, 65943],
            [18441, 82924, 100001, 91003],
            [10594, 19849, 10895, 41685],
            [15956, 3111, 35297, 61519],
        ])
        self.dataqdyqdx.add_to_data(value)
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("add", value, None))

    def test_add_to_data_array(self):
        value = 2
        value = np.ones_like(self.dataqdyqdx.image, dtype=float)*value
        new_image = np.array([
            [44388, 20217, 48678, 20340],
            [64813, 49704, 45713, 77212],
            [4703, 9054, 61102, 46327],
            [60021, 70071, np.nan, 65943],
            [18441, 82924, 100001, 91003],
            [10594, 19849, 10895, 41685],
            [15956, 3111, 35297, 61519],
        ])
        self.dataqdyqdx.add_to_data(value)
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("add", value, None))

    def test_subtract_from_data_value(self):
        value = 2
        new_image = np.array([
            [44384, 20213, 48674, 20336],
            [64809, 49700, 45709, 77208],
            [4699, 9050, 61098, 46323],
            [60017, 70067, np.nan, 65939],
            [18437, 82920, 99997, 90999],
            [10590, 19845, 10891, 41681],
            [15952, 3107, 35293, 61515],
        ])
        self.dataqdyqdx.subtract_from_data(value)
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("subtract", value, None))

    def test_subtract_from_data_array(self):
        value = 2
        value = np.ones_like(self.dataqdyqdx.image, dtype=float)*value
        new_image = np.array([
            [44384, 20213, 48674, 20336],
            [64809, 49700, 45709, 77208],
            [4699, 9050, 61098, 46323],
            [60017, 70067, np.nan, 65939],
            [18437, 82920, 99997, 90999],
            [10590, 19845, 10891, 41681],
            [15952, 3107, 35293, 61515],
        ])
        self.dataqdyqdx.subtract_from_data(value)
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("subtract", value, None))

    def test_scale_data_value(self):
        value = 2
        new_image = np.array([
            [88772, 40430, 97352, 40676],
            [129622, 99404, 91422, 154420],
            [9402, 18104, 122200, 92650],
            [120038, 140138, np.nan, 131882],
            [36878, 165844, 199998, 182002],
            [21184, 39694, 21786, 83366],
            [31908, 6218, 70590, 123034],
        ])
        self.dataqdyqdx.scale_data(value)
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("scale", value, None))

    def test_scale_data_array(self):
        value = 2
        value = np.ones_like(self.dataqdyqdx.image, dtype=float)*value
        new_image = np.array([
            [88772, 40430, 97352, 40676],
            [129622, 99404, 91422, 154420],
            [9402, 18104, 122200, 92650],
            [120038, 140138, np.nan, 131882],
            [36878, 165844, 199998, 182002],
            [21184, 39694, 21786, 83366],
            [31908, 6218, 70590, 123034],
        ])
        self.dataqdyqdx.scale_data(value)
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("scale", value, None))

    def test_normalize_data_value(self):
        value = 2
        new_image = np.array([
            [22193, 10107.5, 24338, 10169],
            [32405.5, 24851, 22855.5, 38605],
            [2350.5, 4526, 30550, 23162.5],
            [30009.5, 35034.5, np.nan, 32970.5],
            [9219.5, 41461, 49999.5, 45500.5],
            [5296, 9923.5, 5446.5, 20841.5],
            [7977, 1554.5, 17647.5, 30758.5],
        ])
        self.dataqdyqdx.normalize_data(value)
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("normalize", value, None))

    def test_normalize_data_array(self):
        value = 2
        value = np.ones_like(self.dataqdyqdx.image, dtype=float)*value
        new_image = np.array([
            [22193, 10107.5, 24338, 10169],
            [32405.5, 24851, 22855.5, 38605],
            [2350.5, 4526, 30550, 23162.5],
            [30009.5, 35034.5, np.nan, 32970.5],
            [9219.5, 41461, 49999.5, 45500.5],
            [5296, 9923.5, 5446.5, 20841.5],
            [7977, 1554.5, 17647.5, 30758.5],
        ])
        self.dataqdyqdx.normalize_data(value)
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("normalize", value, None))

    def test_reset_intensity(self):
        self.dataqdyqdx.normalize_data(2)
        self.dataqdyqdx.subtract_from_data(2)
        self.dataqdyqdx.add_to_data(5)
        self.dataqdyqdx.scale_data(1.3)
        self.dataqdyqdx.reset_intensity()
        np.testing.assert_array_almost_equal(self.dataqdyqdx.image, self.image)

    def test_normalize_metadata(self):
        value = 2
        key = 'sample_phi_deg'
        self.dataqdyqdx.update_metadata({key: value})
        self.dataqdyqdx.normalize_by_metadata(key)
        new_image = np.array([
            [22193, 10107.5, 24338, 10169],
            [32405.5, 24851, 22855.5, 38605],
            [2350.5, 4526, 30550, 23162.5],
            [30009.5, 35034.5, np.nan, 32970.5],
            [9219.5, 41461, 49999.5, 45500.5],
            [5296, 9923.5, 5446.5, 20841.5],
            [7977, 1554.5, 17647.5, 30758.5],
        ])
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("normalize", value, 'sample_phi_deg'))

    def test_normalize_metadata_multiple(self):
        value = 2
        key = 'sample_phi_deg'
        self.dataqdyqdx.update_metadata({key: value})

        value2 = 1
        key2 = 'sample_phi_offset_deg'
        self.dataqdyqdx.update_metadata({key2: value2})

        self.dataqdyqdx.normalize_by_metadata([key, key2])
        new_image = np.array([
            [22193, 10107.5, 24338, 10169],
            [32405.5, 24851, 22855.5, 38605],
            [2350.5, 4526, 30550, 23162.5],
            [30009.5, 35034.5, np.nan, 32970.5],
            [9219.5, 41461, 49999.5, 45500.5],
            [5296, 9923.5, 5446.5, 20841.5],
            [7977, 1554.5, 17647.5, 30758.5],
        ])
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-2],
                              ("normalize", value, 'sample_phi_deg'))
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("normalize", value2, 'sample_phi_offset_deg'))

    def test_normalize_metadata_warnings(self):
        with self.assertWarns(UserWarning):
            self.dataqdyqdx.normalize_by_metadata('not_a_keyword')

        value2 = 1
        key2 = 'sample_phi_offset_deg'
        self.dataqdyqdx.update_metadata({key2: value2})
        self.dataqdyqdx.normalize_by_metadata('sample_phi_offset_deg')
        with self.assertWarns(UserWarning):
            self.dataqdyqdx.normalize_by_metadata('sample_phi_offset_deg')

    def test_scale_metadata(self):
        value = 2
        key = 'sample_phi_deg'
        self.dataqdyqdx.update_metadata({key: value})
        self.dataqdyqdx.scale_by_metadata(key)
        new_image = np.array([
            [88772, 40430, 97352, 40676],
            [129622, 99404, 91422, 154420],
            [9402, 18104, 122200, 92650],
            [120038, 140138, np.nan, 131882],
            [36878, 165844, 199998, 182002],
            [21184, 39694, 21786, 83366],
            [31908, 6218, 70590, 123034],
        ])
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("scale", value, 'sample_phi_deg'))

    def test_scale_metadata_multiple(self):
        value = 2
        key = 'sample_phi_deg'
        self.dataqdyqdx.update_metadata({key: value})

        value2 = 1
        key2 = 'sample_phi_offset_deg'
        self.dataqdyqdx.update_metadata({key2: value2})

        self.dataqdyqdx.scale_by_metadata([key, key2])
        new_image = np.array([
            [88772, 40430, 97352, 40676],
            [129622, 99404, 91422, 154420],
            [9402, 18104, 122200, 92650],
            [120038, 140138, np.nan, 131882],
            [36878, 165844, 199998, 182002],
            [21184, 39694, 21786, 83366],
            [31908, 6218, 70590, 123034],
        ])
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-2],
                              ("scale", value, 'sample_phi_deg'))
        self.assertTupleEqual(self.dataqdyqdx.data_transformations[-1],
                              ("scale", value2, 'sample_phi_offset_deg'))

    def test_scale_metadata_warnings(self):
        with self.assertWarns(UserWarning):
            self.dataqdyqdx.scale_by_metadata('not_a_keyword')

        value2 = 1
        key2 = 'sample_phi_offset_deg'
        self.dataqdyqdx.update_metadata({key2: value2})
        self.dataqdyqdx.scale_by_metadata('sample_phi_offset_deg')
        with self.assertWarns(UserWarning):
            self.dataqdyqdx.scale_by_metadata('sample_phi_offset_deg')

    def test_footprint_correction(self):
        self.dataqdyqdx.update_metadata(dict(sample_phi_deg=20))
        self.dataqdyqdx.apply_footprint_correction()
        new_image = self.image * self.dataqdyqdx.metadata["footprint_factor"]
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)

    def test_sample_size_correction(self):
        self.dataqdyqdx.update_metadata(dict(
            sample_size_mm=0.5,
            beam_center_mm=0.001,
            beam_fwhm_mm=0.25,
            sample_phi_deg=20,
        ))
        self.dataqdyqdx.apply_sample_size_correction()
        new_image = self.image * self.dataqdyqdx.metadata["sample_size_factor"]
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)

    def test_substrate_absorption_correction(self):
        self.dataqdyqdx.update_metadata({
            'substrate_thickness_um':100,
            'substrate_attenuation_coeff_um-1':1/546,
            'sample_phi_deg':-40
        })

        self.dataqdyqdx.apply_substrate_absorption_correction()
        new_image = self.image * self.dataqdyqdx.metadata["substrate_absorption_factor"]
        np.testing.assert_array_equal(self.dataqdyqdx.image, new_image)

    def test_get_box_dims_size(self):
        size_qdy_px = 3
        size_qdx_px = 2
        limits0, limits1 = self.dataqdyqdx.get_box_dims_size(
            size_qdy_px, size_qdx_px
        )
        self.assertTupleEqual(limits0, (3, 6))
        self.assertTupleEqual(limits1, (1, 3))
    
    def test_get_box_dims_size_edges(self):
        size_qdy_px = 7
        size_qdx_px = 2
        limits0, limits1 = self.dataqdyqdx.get_box_dims_size(
            size_qdy_px, size_qdx_px
        )
        self.assertTupleEqual(limits0, (1, 7))
        self.assertTupleEqual(limits1, (1, 3))

    def test_get_box_dims_size_shift_qdy(self):
        size_qdy_px = 3
        size_qdx_px = 2
        limits0, limits1 = self.dataqdyqdx.get_box_dims_size(
            size_qdy_px, size_qdx_px, shift_box_qdy_px=-1
        )
        self.assertTupleEqual(limits0, (4, 7))
        self.assertTupleEqual(limits1, (1, 3))

    def test_get_box_dims_size_shift_qdx(self):
        size_qdy_px = 3
        size_qdx_px = 2
        limits0, limits1 = self.dataqdyqdx.get_box_dims_size(
            size_qdy_px, size_qdx_px, shift_box_qdx_px=-2
        )
        self.assertTupleEqual(limits0, (3, 6))
        self.assertTupleEqual(limits1, (3, 4))

        # qdy = np.array([0.001139386259, 0.000854539696, 0.000569693132,
        #                 0.000284846566, 0., -0.000284846566,
        #                 -0.000569693132])
        # np.testing.assert_array_almost_equal(
        #     self.dataqdyqdx.qdy, qdy, decimal=12)

        # qdx = np.array([0.000569693132, 0.000284846566, 0., -0.000284846566])
        # np.testing.assert_array_almost_equal(
        #     self.dataqdyqdx.qdx, qdx, decimal=12)

    def test_get_box_dims_qrange(self):
        range_qdy = (0.1, 0)
        range_qdx = (-0.0003, 0.0003)

        limits = self.dataqdyqdx.get_box_dims_qrange(
            qby=range_qdy, qbx=range_qdx
        )

        self.assertTupleEqual(limits[0], (0, 5))
        self.assertTupleEqual(limits[1], (1, 4))

    def test_integrate_sum_axis0(self):
        qslice, _ = self.dataqdyqdx.integrate_box(
            range_qdy_px=(2, 4),
            range_qdx_px=(0, 4),
            mode='sum',
            axis=0,
            subtract_background_offset=[-2, 2]
        )
        int_i_expected = np.array([35689, -7222, np.nan, -2850])
        np.testing.assert_array_almost_equal(qslice.Iq, int_i_expected)

        np.testing.assert_array_almost_equal(
            qslice.q,
            np.array([0.000569693132, 0.000284846566, 0., -0.000284846566]))

        np.testing.assert_array_almost_equal(
            qslice.qby,
            np.array([0.0004272698490, 0.0004272698490, 0.0004272698490, 0.0004272698490])
        )

        self.assertTupleEqual(qslice.limits_axis0, (2, 4))
        self.assertTupleEqual(qslice.limits_axis1, (0, 4))

        self.assertEqual(qslice.q_axis, 'qbx')
        self.assertEqual(qslice.integration_axis, 0)
        self.assertEqual(qslice.integration_mode, 'sum')

        image_roi = np.array([
            [4701, 9052, 61100, 46325],
            [60019, 70069, np.nan, 65941]
        ])
        np.testing.assert_array_almost_equal(qslice.image_roi, image_roi)

        image_mask = np.array([
            [False, False, False, False],
            [False, False, True, False]
        ])
        np.testing.assert_array_equal(qslice.image_mask, image_mask)

        # keep in mind the first point in background_i_avg will just be
        # from one background slice because point (0,0) of image
        # was masked out in setup as part of the custom mask!
        background_i_avg = np.array([
            29031, 86343, 102639.5, 115116
        ])
        np.testing.assert_array_almost_equal(qslice.background_Iq, background_i_avg)

    def test_find_peaks2d_forwards_roi_filters_q_and_plots(self):
        helper_peaks = np.array([[0.5, 1.0], [2.0, 2.5]], dtype=float)

        with patch("cdsaxs.data.data2d.find_peaks_2D",
                   return_value=helper_peaks.copy()) as mock_find, \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_peaks2d",
                      return_value="figure") as mock_plot:
            peaks, peaks_q, fig = self.dataqdyqdx.find_peaks2D(
                range_qdy_px=(1, 5),
                range_qdx_px=(1, 4),
                exclude_qdy=(0.0006, 0.0011),
                exclude_qdx=[(0.0002, 0.0004)],
                log_scale=False,
                refinement_size=9,
                show_plot=True,
                zoom_plot=False,
                plotting_kwargs={"cmap": "magma"},
                min_distance=3,
            )

        expected_roi = self.image[1:5, 1:4]
        expected_mask = self.dataqdyqdx.mask[1:5, 1:4]
        np.testing.assert_array_equal(mock_find.call_args.args[0], expected_roi)
        np.testing.assert_array_equal(mock_find.call_args.kwargs["mask"], expected_mask)
        self.assertFalse(mock_find.call_args.kwargs["log_scale"])
        self.assertEqual(mock_find.call_args.kwargs["refinement_size"], 9)
        self.assertEqual(mock_find.call_args.kwargs["min_distance"], 3)

        np.testing.assert_allclose(peaks, np.array([[3.0, 3.5]]))
        np.testing.assert_allclose(peaks_q, np.array([[0.000284846566, -0.000284846566]]))
        self.assertEqual(fig, "figure")
        mock_plot.assert_called_once_with(
            self.dataqdyqdx,
            peaks=peaks,
            limits_axis0=(1, 5),
            limits_axis1=(1, 4),
            zoom_plot=False,
            cmap="magma",
        )

    def test_find_peaks2d_returns_none_figure_without_plot(self):
        helper_peaks = np.array([[1.0, 0.0]], dtype=float)

        with patch("cdsaxs.data.data2d.find_peaks_2D",
                   return_value=helper_peaks.copy()) as mock_find, \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_peaks2d") as mock_plot:
            peaks, peaks_q, fig = self.dataqdyqdx.find_peaks2D(
                range_qdy_px=(0, 3),
                range_qdx_px=(0, 2),
                show_plot=False,
            )

        np.testing.assert_allclose(peaks, np.array([[1.0, 0.0]]))
        np.testing.assert_allclose(peaks_q, np.array([[0.000854539698, 0.000569693132]]))
        self.assertIsNone(fig)
        mock_find.assert_called_once()
        mock_plot.assert_not_called()

    def test_find_peaks2d_one_axis_defaults_peak_axis_and_filters_q(self):
        helper_peaks = np.array([[0.5, 1.0], [2.0, 2.5]], dtype=float)

        with patch("cdsaxs.data.data2d.find_peaks_2D_one_axis",
                   return_value=helper_peaks.copy()) as mock_find, \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_peaks2d",
                      return_value="axis-figure") as mock_plot:
            peaks, peaks_q, fig = self.dataqdyqdx.find_peaks2D_one_axis(
                range_qdy_px=(1, 6),
                range_qdx_px=(1, 3),
                exclude_q=(0.0006, 0.0011),
                integration_mode="mean",
                log_scale=False,
                refinement_size=5,
                algorithm="scipy",
                zoom_plot=False,
                show_plot=True,
                distance=4,
            )

        expected_roi = self.image[1:6, 1:3]
        expected_mask = self.dataqdyqdx.mask[1:6, 1:3]
        np.testing.assert_array_equal(mock_find.call_args.args[0], expected_roi)
        np.testing.assert_array_equal(mock_find.call_args.kwargs["mask"], expected_mask)
        self.assertEqual(mock_find.call_args.kwargs["peak_axis"], 0)
        self.assertEqual(mock_find.call_args.kwargs["integration_mode"], "mean")
        self.assertFalse(mock_find.call_args.kwargs["log_scale"])
        self.assertEqual(mock_find.call_args.kwargs["refinement_size"], 5)
        self.assertEqual(mock_find.call_args.kwargs["algorithm"], "scipy")
        self.assertEqual(mock_find.call_args.kwargs["distance"], 4)

        np.testing.assert_allclose(peaks, np.array([[3.0, 3.5]]))
        np.testing.assert_allclose(peaks_q, np.array([[0.000284846566, -0.000284846566]]))
        self.assertEqual(fig, "axis-figure")
        mock_plot.assert_called_once_with(
            self.dataqdyqdx,
            peaks=peaks,
            limits_axis0=(1, 6),
            limits_axis1=(1, 3),
            zoom_plot=False,
            distance=4,
        )

    def test_find_peaks2d_one_axis_remaps_peak_axis_and_handles_empty_peaks(self):
        with patch("cdsaxs.data.data2d.find_peaks_2D_one_axis",
                   return_value=np.array([], dtype=float)) as mock_find, \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_peaks2d") as mock_plot:
            peaks, peaks_q, fig = self.dataqdyqdx.find_peaks2D_one_axis(
                range_qdy_px=(0, 4),
                range_qdx_px=(0, 4),
                peak_axis="qdx",
                show_plot=False,
            )

        self.assertEqual(mock_find.call_args.kwargs["peak_axis"], 1)
        self.assertEqual(peaks.shape, (0, 2))
        self.assertEqual(peaks_q.shape, (0, 2))
        self.assertIsNone(fig)
        mock_plot.assert_not_called()

    def test_find_beam_center_from_peaks_returns_center_without_update(self):
        original_center = self.dataqdyqdx.metadata["center_px"]
        peaks = np.array([[4.0, 3.0], [4.0, 1.0]], dtype=float)

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(peaks, None, None)) as mock_find, \
                patch("cdsaxs.data.data2d.line_fit",
                      return_value=(0.0, 0.0, 4.0)) as mock_line_fit, \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_beam_center",
                      return_value="beam-fig") as mock_plot:
            center, fig = self.dataqdyqdx.find_beam_center_from_peaks(
                range_qdy_px=(2, 6),
                range_qdx_px=(0, 4),
                show_plot=True,
                zoom_plot=False,
            )

        self.assertEqual(center, (4.0, 2.0))
        self.assertEqual(fig, "beam-fig")
        self.assertEqual(self.dataqdyqdx.metadata["center_px"], original_center)
        self.assertEqual(mock_find.call_args.kwargs["peak_axis"], 1)
        mock_line_fit.assert_called_once()
        mock_plot.assert_called_once()
        self.assertIs(mock_plot.call_args.args[0], self.dataqdyqdx)
        np.testing.assert_array_equal(
            mock_plot.call_args.kwargs["peaks"],
            np.array([[4.0, 1.0], [4.0, 3.0]]),
        )
        self.assertEqual(mock_plot.call_args.kwargs["limits_axis0"], (2, 6))
        self.assertEqual(mock_plot.call_args.kwargs["limits_axis1"], (0, 4))
        self.assertFalse(mock_plot.call_args.kwargs["zoom_plot"])
        self.assertEqual(mock_plot.call_args.kwargs["show_beam_center"], (4.0, 2.0))

    def test_find_beam_center_from_peaks_updates_metadata_and_honors_guess(self):
        peaks = np.array([[4.0, 3.0], [4.0, 1.0]], dtype=float)

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(peaks, None, None)) as mock_find, \
                patch("cdsaxs.data.data2d.line_fit",
                      return_value=(0.0, 0.0, 4.0)), \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_beam_center",
                      return_value="beam-fig") as mock_plot:
            center, fig = self.dataqdyqdx.find_beam_center_from_peaks(
                range_qdy_px=(2, 6),
                range_qdx_px=(0, 4),
                beam_center_guess=(4.0, 2.2),
                update=True,
                show_plot=True,
            )

        self.assertEqual(center, (4.0, 2.0))
        self.assertEqual(fig, "beam-fig")
        self.assertEqual(self.dataqdyqdx.metadata["center_px"], (4.0, 2.0))
        self.assertEqual(mock_find.call_args.kwargs["peak_axis"], 1)
        self.assertTrue(mock_plot.call_args.kwargs["show_beam_center"])

    def test_find_beam_center_from_peaks_raises_for_asymmetric_or_missing_peaks(self):
        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(np.array([[4.0, 3.0]], dtype=float), None, None)), \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_beam_center",
                      return_value="beam-fig") as mock_plot:
            with self.assertRaisesRegex(ValueError, "not symmetric"):
                self.dataqdyqdx.find_beam_center_from_peaks(
                    range_qdy_px=(2, 6),
                    range_qdx_px=(0, 4),
                    show_plot=True,
                )

        self.assertFalse(mock_plot.call_args.kwargs["show_beam_center"])

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(np.empty((0, 2)), None, None)), \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_beam_center",
                      return_value="beam-fig"):
            with self.assertRaisesRegex(ValueError, "No peaks detected"):
                self.dataqdyqdx.find_beam_center_from_peaks(
                    range_qdy_px=(2, 6),
                    range_qdx_px=(0, 4),
                    show_plot=True,
                )

    def test_find_beam_center_from_peaks_ignores_selected_peaks(self):
        peaks = np.array([[4.0, 0.5], [4.0, 1.0], [4.0, 3.0], [4.0, 3.5]], dtype=float)

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(peaks, None, None)), \
                patch("cdsaxs.data.data2d.line_fit",
                      return_value=(0.0, 0.0, 4.0)):
            center, fig = self.dataqdyqdx.find_beam_center_from_peaks(
                range_qdy_px=(2, 6),
                range_qdx_px=(0, 4),
                ignore_peaks=[0, 3],
                show_plot=False,
            )

        self.assertEqual(center, (4.0, 2.0))
        self.assertIsNone(fig)

    def test_find_sdd_from_reference_peaks_returns_and_updates(self):
        peaks = np.array([[4.0, 1.0], [4.0, 3.0]], dtype=float)
        peaks_q = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(peaks, peaks_q, None)) as mock_find, \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_sdd",
                      return_value="sdd-fig") as mock_plot:
            average_sdd, std_sdd, fig = self.dataqdyqdx.find_sdd_from_reference_peaks(
                pitch_nm=100,
                range_qdy_px=(2, 6),
                range_qdx_px=(0, 4),
                update=True,
                show_plot=True,
                zoom_plot=False,
            )

        self.assertEqual((average_sdd, std_sdd), (24.57, 0.0))
        self.assertEqual(self.dataqdyqdx.metadata["sdd_cm"], 24.57)
        self.assertEqual(fig, "sdd-fig")
        self.assertEqual(mock_find.call_args.kwargs["peak_axis"], 1)
        mock_plot.assert_called_once()
        self.assertIs(mock_plot.call_args.args[0], self.dataqdyqdx)
        np.testing.assert_array_equal(
            mock_plot.call_args.kwargs["peaks"],
            np.array([[4.0, 1.0], [4.0, 3.0]]),
        )
        self.assertEqual(mock_plot.call_args.kwargs["limits_axis0"], (2, 6))
        self.assertEqual(mock_plot.call_args.kwargs["limits_axis1"], (0, 4))
        self.assertFalse(mock_plot.call_args.kwargs["zoom_plot"])
        self.assertEqual(mock_plot.call_args.kwargs["sdd_cm"], (24.57, 0.0))

    def test_find_sdd_from_reference_peaks_respects_peak_orders_and_ignore_orders(self):
        peaks = np.array([[4.0, 0.0], [4.0, 1.0], [4.0, 3.0]], dtype=float)
        peaks_q = np.zeros_like(peaks)

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(peaks, peaks_q, None)):
            average_sdd, std_sdd, fig = self.dataqdyqdx.find_sdd_from_reference_peaks(
                pitch_nm=100,
                range_qdy_px=(2, 6),
                range_qdx_px=(0, 4),
                peak_orders=np.array([-2, -1, 1]),
                ignore_orders=[-2],
                show_plot=False,
            )

        self.assertEqual((average_sdd, std_sdd), (0.0, 24.57))
        self.assertIsNone(fig)

    def test_calculate_omega_returns_expected_angle(self):
        self.dataqdyqdx.update_metadata(
            {
                "sample_phi_deg": 30,
                "sample_phi_offset_deg": 0,
                "sample_chi_deg": 10,
                "sample_chi_offset_deg": 0,
            },
            overwrite=True,
            hide_q_warnings=True,
        )

        omega = self.dataqdyqdx.calculate_omega(5)

        self.assertAlmostEqual(omega, 7.3194743104)

    def test_find_chi_from_peaks_returns_real_angle_and_warns_on_insufficient_peaks(self):
        peaks = np.array([[2.0, 1.0], [5.0, 2.0]], dtype=float)

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(peaks, None, None)) as mock_find, \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_detector_rotation_correction",
                      return_value="chi-fig") as mock_plot:
            angle, fig = self.dataqdyqdx.find_chi_from_peaks(
                range_qdy_px=(0, 7),
                range_qdx_px=(0, 4),
                show_plot=True,
                zoom_plot=False,
            )

        self.assertAlmostEqual(angle, 63.4349488229)
        self.assertEqual(fig, "chi-fig")
        self.assertEqual(mock_find.call_args.kwargs["peak_axis"], 1)
        plotted_line = mock_plot.call_args.kwargs["line"]
        self.assertAlmostEqual(plotted_line[0], -63.4349488229)
        self.assertAlmostEqual(plotted_line[1], -2.0)
        self.assertAlmostEqual(plotted_line[2], 0.0)
        mock_plot.assert_called_once_with(
            self.dataqdyqdx,
            peaks=peaks,
            line=mock_plot.call_args.kwargs["line"],
            limits_axis0=(0, 7),
            limits_axis1=(0, 4),
            zoom_plot=False,
        )

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(np.array([[2.0, 1.0]], dtype=float), None, None)):
            with self.assertWarnsRegex(UserWarning, "Insuffient peaks"):
                angle, fig = self.dataqdyqdx.find_chi_from_peaks(
                    range_qdy_px=(0, 7),
                    range_qdx_px=(0, 4),
                    show_plot=False,
                )

        self.assertTrue(np.isnan(angle))
        self.assertIsNone(fig)

    def test_find_omega_from_peaks_uses_real_angle_math_and_warns_on_insufficient_peaks(self):
        peaks = np.array([[2.0, 1.0], [5.0, 2.0]], dtype=float)
        self.dataqdyqdx.update_metadata(
            {
                "sample_phi_deg": 30,
                "sample_phi_offset_deg": 0,
                "sample_chi_deg": 10,
                "sample_chi_offset_deg": 0,
            },
            overwrite=True,
            hide_q_warnings=True,
        )

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(peaks, None, None)) as mock_find, \
                patch("cdsaxs.data.data2d.plotting.plot_data2d_find_detector_rotation_correction",
                      return_value="omega-fig") as mock_plot:
            omega, fig = self.dataqdyqdx.find_omega_from_peaks(
                range_qdy_px=(0, 7),
                range_qdx_px=(0, 4),
                show_plot=True,
                zoom_plot=False,
            )

        self.assertAlmostEqual(omega, -74.6322033805)
        self.assertEqual(fig, "omega-fig")
        self.assertEqual(mock_find.call_args.kwargs["peak_axis"], 1)
        plotted_line = mock_plot.call_args.kwargs["line"]
        self.assertAlmostEqual(plotted_line[0], -63.4349488229)
        self.assertAlmostEqual(plotted_line[1], -2.0)
        self.assertAlmostEqual(plotted_line[2], 0.0)
        mock_plot.assert_called_once_with(
            self.dataqdyqdx,
            peaks=peaks,
            line=mock_plot.call_args.kwargs["line"],
            limits_axis0=(0, 7),
            limits_axis1=(0, 4),
            zoom_plot=False,
        )

        with patch.object(self.dataqdyqdx, "find_peaks2D_one_axis",
                          return_value=(np.array([[2.0, 1.0]], dtype=float), None, None)):
            with self.assertWarnsRegex(UserWarning, "Insuffient peaks"):
                omega, fig = self.dataqdyqdx.find_omega_from_peaks(
                    range_qdy_px=(0, 7),
                    range_qdx_px=(0, 4),
                    show_plot=False,
                )

        self.assertTrue(np.isnan(omega))
        self.assertIsNone(fig)
