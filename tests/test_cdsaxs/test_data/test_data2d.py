import numpy as np
import unittest

from cdsaxs.calculators import wavelength_to_energy
from cdsaxs.data.data2d import Data2D


class TestCombineData2D(unittest.TestCase):
    # TODO: implement a test for combining two instances of Data2D
    pass


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
        qdy = np.array([0.001139386259, 0.000854539696, 0.000569693132,
                        0.000284846566, 0., -0.000284846566,
                        -0.000569693132])
        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, qdy, decimal=12)

        qdx = np.array([0.000569693132, 0.000284846566, 0., -0.000284846566])
        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdx, qdx, decimal=12)

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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The clockwise 180 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The clockwise 270 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The clockwise 630 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The horizontal flip resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The horizontal flip resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
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

        limits0, limits1 = self.dataqdyqdx.get_box_dims_qrange(
            range_qdy, range_qdx
        )

        self.assertTupleEqual(limits0, (0, 5))
        self.assertTupleEqual(limits1, (1, 4))

    def test_integrate_sum_axis0(self):
        qslice = self.dataqdyqdx.integrate_box(
            limits_qdy_px=(2, 4),
            limits_qdx_px=(0, 4),
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
            qslice.qdy,
            np.array([0.0004272698490])
        )

        self.assertTupleEqual(qslice.limits_axis0, (2, 4))
        self.assertTupleEqual(qslice.limits_axis1, (0, 4))

        self.assertEqual(qslice.q_axis, 'qdx')
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

        background_i_avg = np.array([
            29031, 86343, 102639.5, 115116
        ])
        np.testing.assert_array_almost_equal(qslice.background, background_i_avg)

    def test_find_peaks2D(self):
        # TODO: implement peaks2D test
        pass

    def test_find_peaks2D_one_axis(self):
        # TODO: implement peaks2D one axis test
        pass

    def test_find_beam_center(self):
        # TODO: implement find center test
        pass

    def test_find_sdd(self):
        # TODO: implement find sdd test
        pass

    def test_find_detector_rotation_correction(self):
        # TODO: implement find detector rotation correction test
        pass
