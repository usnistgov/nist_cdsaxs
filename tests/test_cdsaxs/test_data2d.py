import numpy as np
import unittest

from cdsaxs.calculators import wavelength_to_energy
from cdsaxs.data2d import Data2D, DataQdyQdx


class TestData2D(unittest.TestCase):

    def setUp(self):
        self.image = np.array(
            [[44386., 20215., 48676., 20338.],
             [64811., 49702., 45711., 77210.],
             [ 4701.,  9052., 61100., 46325.],
             [60019., 70069.,   857., 65941.],
             [18439., 82922., 99999., 91001.],
             [10592., 19847., 10893., 41683.],
             [15954.,  3109., 35295., 61517.]]).astype(np.float64)

        self.data2d = Data2D(self.image)

    def test_data2d_rotate_ccw90(self):
        """test 90 degree rotation counter-clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   857., 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.rotate_image(90)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "wrong image.")
        self.assertEqual(self.data2d._image_transformations[-1],
                         "R1")

        self.data2d.reset_image_orientation()

    def test_data2d_rotate_ccw180(self):
        """test 180 degree rotation counter-clockwise"""
        expected_image = np.array(
            [[61517., 35295.,  3109., 15954.],
             [41683., 10893., 19847., 10592.],
             [91001., 99999., 82922., 18439.],
             [65941.,   857., 70069., 60019.],
             [46325., 61100.,  9052.,  4701.],
             [77210., 45711., 49702., 64811.],
             [20338., 48676., 20215., 44386.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.rotate_image(180)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " wrong image.")
        self.assertEqual(self.data2d._image_transformations[-1],
                         "R2")

        self.data2d.reset_image_orientation()

    def test_data2d_rotate_ccw270(self):
        """test 270 degree rotation counter-clockwise"""
        expected_image = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.rotate_image(270)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " wrong image.")
        self.assertEqual(self.data2d._image_transformations[-1],
                         "R3")

        self.data2d.reset_image_orientation()

    def test_data2d_rotate_ccw_m90(self):
        """test -90 degree rotation counter-clockwise (90 degree clockwise)"""
        expected_image = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.rotate_image(-90)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " wrong image.")
        self.assertEqual(self.data2d._image_transformations[-1],
                         "R3")

        self.data2d.reset_image_orientation()

    def test_data2d_rotate_cw90(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.rotate_image(90, direction='cw')

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The clockwise 90 degree rotation resulted in the"
            " wrong image.")
        self.assertEqual(self.data2d._image_transformations[-1],
                         "R3")

        self.data2d.reset_image_orientation()

    def test_data2d_rotate_cw180(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[61517., 35295.,  3109., 15954.],
             [41683., 10893., 19847., 10592.],
             [91001., 99999., 82922., 18439.],
             [65941.,   857., 70069., 60019.],
             [46325., 61100.,  9052.,  4701.],
             [77210., 45711., 49702., 64811.],
             [20338., 48676., 20215., 44386.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.rotate_image(180, direction='cw')

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The clockwise 180 degree rotation resulted in the"
            " wrong image.")
        self.assertEqual(self.data2d._image_transformations[-1],
                         "R2")

        self.data2d.reset_image_orientation()

    def test_data2d_rotate_cw270(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   857., 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.rotate_image(270, direction='cw')

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The clockwise 270 degree rotation resulted in the"
            " wrong image.")
        self.assertEqual(self.data2d._image_transformations[-1],
                         "R1")

        self.data2d.reset_image_orientation()

    def test_data2d_rotate_cw630(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   857., 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.rotate_image(630, direction='cw')

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The clockwise 630 degree rotation resulted in the"
            " wrong image.")
        self.assertEqual(self.data2d._image_transformations[-1],
                         "R1")

        self.data2d.reset_image_orientation()

    def test_flip_horizontally(self):
        expected_image = np.array(
            [[20338., 48676., 20215., 44386.],
             [77210., 45711., 49702., 64811.],
             [46325., 61100.,  9052.,  4701.],
             [65941.,   857., 70069., 60019.],
             [91001., 99999., 82922., 18439.],
             [41683., 10893., 19847., 10592.],
             [61517., 35295.,  3109., 15954.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.flip_horizontally()

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The horizontal flip resulted in the wrong image."
        )
        self.assertEqual(self.data2d._image_transformations[-1],
                         "HF")

        self.data2d.reset_image_orientation()

    def test_flip_vertically(self):
        expected_image = np.array(
            [[15954.,  3109., 35295., 61517.],
             [10592., 19847., 10893., 41683.],
             [18439., 82922., 99999., 91001.],
             [60019., 70069.,   857., 65941.],
             [ 4701.,  9052., 61100., 46325.],
             [64811., 49702., 45711., 77210.],
             [44386., 20215., 48676., 20338.]]
        ).astype(np.float64)

        self.data2d.reset_image_orientation()
        self.data2d.flip_vertically()

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The vertical flip resulted in the wrong image."
        )
        self.assertEqual(self.data2d._image_transformations[-1],
                         "VF")

        self.data2d.reset_image_orientation()

    def test_data2d_reset_image_orientation(self):
        """test the transformation reset function"""

        self.data2d.reset_image_orientation()

        self.data2d.rotate_image(90)
        self.data2d.rotate_image(90)
        self.data2d.flip_horizontally()
        self.data2d.rotate_image(-180)
        self.data2d.flip_vertically()
        self.data2d.rotate_image(270)
        self.data2d.rotate_image(270)
        self.data2d.flip_horizontally()
        self.data2d.flip_vertically()
        self.data2d.rotate_image(90, direction='cw')

        self.data2d.reset_image_orientation()
        np.testing.assert_array_equal(
            self.data2d.image, self.image)

    def test_data2d_integrate_box_sum_axis0(self):
        int_i_actual, int_params_actual = self.data2d.integrate_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            mode='sum',
            axis=0
        )
        int_i_expected = np.array([111749., 198625.,])
        np.testing.assert_array_almost_equal(int_i_actual, int_i_expected)
        self.assertListEqual(int_params_actual["limits_axis0"], [3, 6])
        self.assertListEqual(int_params_actual["limits_axis1"], [2, 4])
        self.assertEqual(int_params_actual["axis"], 0)
        self.assertEqual(int_params_actual["mode"], "sum")

    def test_data2d_integrate_box_sum_axis1(self):
        int_i_actual, int_params_actual = self.data2d.integrate_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            mode='sum',
            axis=1
        )
        int_i_expected = np.array([66798., 191000., 52576.])
        np.testing.assert_array_almost_equal(int_i_actual, int_i_expected)
        self.assertListEqual(int_params_actual["limits_axis0"], [3, 6])
        self.assertListEqual(int_params_actual["limits_axis1"], [2, 4])
        self.assertEqual(int_params_actual["axis"], 1)
        self.assertEqual(int_params_actual["mode"], "sum")

    def test_data2d_integrate_box_mean_axis0(self):
        int_i_actual, int_params_actual = self.data2d.integrate_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            mode='mean',
            axis=0
        )

        int_i_expected = np.array([37249.6666666667, 66208.3333333333])
        np.testing.assert_array_almost_equal(int_i_actual, int_i_expected)
        self.assertListEqual(int_params_actual["limits_axis0"], [3, 6])
        self.assertListEqual(int_params_actual["limits_axis1"], [2, 4])
        self.assertEqual(int_params_actual["axis"], 0)
        self.assertEqual(int_params_actual["mode"], "mean")

    def test_data2d_integrate_box_mean_axis1(self):
        int_i_actual, int_params_actual = self.data2d.integrate_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            mode='mean',
            axis=1
        )
        int_i_expected = np.array([33399., 95500., 26288.])
        np.testing.assert_array_almost_equal(int_i_actual, int_i_expected)
        self.assertListEqual(int_params_actual["limits_axis0"], [3, 6])
        self.assertListEqual(int_params_actual["limits_axis1"], [2, 4])
        self.assertEqual(int_params_actual["axis"], 1)
        self.assertEqual(int_params_actual["mode"], "mean")


class TestDataQdyQdx(unittest.TestCase):

    def setUp(self):
        self.image = np.array(
            [[44386., 20215., 48676., 20338.],
             [64811., 49702., 45711., 77210.],
             [ 4701.,  9052., 61100., 46325.],
             [60019., 70069.,   857., 65941.],
             [18439., 82922., 99999., 91001.],
             [10592., 19847., 10893., 41683.],
             [15954.,  3109., 35295., 61517.]]).astype(np.float64)

        self.metadata = {
            'center_px': [4, 2],
            'wavelength_nm': 0.07,
            'pixel_size_um': 172,
            'sdd_cm': 542,
        }
        self.user_params = {
            'custom_param': 20
        }

        self.dataqdyqdx = DataQdyQdx(
            self.image, metadata=self.metadata, user_params=self.user_params)

        self.metadata["energy_ev"] = wavelength_to_energy(
            self.metadata["wavelength_nm"])

    def test_dataqdyqdx_metadata_keywords(self):
        with self.assertRaises(ValueError):
            DataQdyQdx(self.image, {'not_a_keyword': 2})

    def test_dataqdyqdx_metadata_pass(self):
        self.assertDictEqual(self.dataqdyqdx.metadata, self.metadata)

    def test_dataqdyqdx_params_pass(self):
        self.assertDictEqual(self.dataqdyqdx.user_params, self.user_params)

    def test_dataqdyqdx_metadata_update(self):
        self.dataqdyqdx.update_metadata({"name": "test name"})
        self.assertEqual(self.dataqdyqdx.metadata["name"], "test name")

    def test_dataqdyqdx_calculate_q(self):
        qdy = np.array([0.001139386259, 0.000854539696, 0.000569693132,
                        0.000284846566, 0., -0.000284846566,
                        -0.000569693132])
        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, qdy, decimal=12)

        qdx = np.array([0.000569693132, 0.000284846566, 0., -0.000284846566])
        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdx, qdx, decimal=12)

    def test_dataqdyqdx_update_wavelength(self):
        self.dataqdyqdx.update_metadata({'wavelength_nm': 0.1})
        self.metadata['wavelength_nm'] = 0.1
        self.metadata['energy_ev'] = wavelength_to_energy(0.1)

        self.assertEqual(self.dataqdyqdx.metadata['wavelength_nm'],
                         self.metadata['wavelength_nm'])
        self.assertEqual(self.dataqdyqdx.metadata['energy_ev'],
                         self.metadata['energy_ev'])

        qdy = np.array([0.000797570381, 0.000598177787, 0.000398785192,
                        0.000199392596, 0., -0.000199392596,
                        -0.000398785192])
        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, qdy, decimal=12)

        qdx = np.array([0.000398785192, 0.000199392596, 0., -0.000199392596])
        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdx, qdx, decimal=12)

        # undo the change
        self.dataqdyqdx.update_metadata({'wavelength_nm': 0.07})
        self.metadata['wavelength_nm'] = 0.07
        self.metadata['energy_ev'] = wavelength_to_energy(0.07)

    def test_dataqdyqdx_rotate_ccw90(self):
        expected_img = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   857., 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.001139386259, 0.000854539696,
                                  0.000569693132, 0.000284846566, 0.,
                                  -0.000284846566, -0.000569693132])
        expected_qys = np.array([0.000284846566, 0., -0.000284846566,
                                 -0.000569693132])
        expected_center_px = [1, 4]

        self.dataqdyqdx.reset_image_orientation()
        self.dataqdyqdx.rotate_image(90)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_img,
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

        self.assertListEqual(self.dataqdyqdx.metadata['center_px'],
                             expected_center_px,
                             msg="The counter-clockwise 90 degree rotation "
                             "resulted in the wrong center pixel location.")

        self.dataqdyqdx.reset_image_orientation()

    def test_dataqdyqdx_rotate_ccw180(self):
        expected_img = np.array(
            [[61517., 35295.,  3109., 15954.],
             [41683., 10893., 19847., 10592.],
             [91001., 99999., 82922., 18439.],
             [65941.,   857., 70069., 60019.],
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

        expected_center_px = [2, 1]

        self.dataqdyqdx.reset_image_orientation()
        self.dataqdyqdx.rotate_image(180)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_img,
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

        self.assertListEqual(self.dataqdyqdx.metadata['center_px'],
                             expected_center_px,
                             msg="The counter-clockwise 180 degree rotation "
                             "resulted in the wrong center pixel location.")

        self.dataqdyqdx.reset_image_orientation()

    def test_dataqdyqdx_rotate_ccw270(self):
        expected_img = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.000569693132, 0.000284846566, 0.,
                                  -0.000284846566, -0.000569693132,
                                  -0.000854539696, -0.001139386259])
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566]
                                )
        expected_center_px = [2, 2]

        self.dataqdyqdx.reset_image_orientation()
        self.dataqdyqdx.rotate_image(270)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_img,
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

        self.assertListEqual(self.dataqdyqdx.metadata['center_px'],
                             expected_center_px,
                             msg="The counter-clockwise 182700 degree rotation"
                             " resulted in the wrong center pixel location.")

        self.dataqdyqdx.reset_image_orientation()

    def test_dataqdyqdx_rotate_ccw_m90(self):
        expected_img = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132,
                                 -0.000854539696, -0.001139386259])
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566]
                                )
        expected_center_px = [2, 2]

        self.dataqdyqdx.reset_image_orientation()
        self.dataqdyqdx.rotate_image(-90)

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_img,
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

        self.assertListEqual(self.dataqdyqdx.metadata['center_px'],
                             expected_center_px,
                             msg="The counter-clockwise -90 degree rotation "
                             "resulted in the wrong center pixel location.")

        self.dataqdyqdx.reset_image_orientation()

    def test_dataqdyqdx_rotate_cw90(self):
        expected_img = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)
        expected_qxzs = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132,
                                 -0.000854539696, -0.001139386259])
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566])
        expected_center_px = [2, 2]

        self.dataqdyqdx.reset_image_orientation()
        self.dataqdyqdx.rotate_image(90, direction='cw')

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_img,
            err_msg="The clockwise -90 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdx, expected_qxzs, decimal=12,
            err_msg="The clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, expected_qys, decimal=12,
            err_msg="The clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertListEqual(self.dataqdyqdx.metadata['center_px'],
                             expected_center_px,
                             msg="The clockwise -90 degree rotation "
                             "resulted in the wrong center pixel location.")

        self.dataqdyqdx.reset_image_orientation()

    def test_flip_horizontally(self):
        expected_img = np.array(
            [[20338., 48676., 20215., 44386.],
             [77210., 45711., 49702., 64811.],
             [46325., 61100.,  9052.,  4701.],
             [65941.,   857., 70069., 60019.],
             [91001., 99999., 82922., 18439.],
             [41683., 10893., 19847., 10592.],
             [61517., 35295.,  3109., 15954.]]
        )
        expected_qys = np.array([0.001139386259, 0.000854539696,
                                 0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132])
        expected_qxzs = np.array([0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132])

        expected_center_px = [4, 1]

        self.dataqdyqdx.reset_image_orientation()
        self.dataqdyqdx.flip_horizontally()

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_img,
            err_msg="The horizontal flip resulted in the"
            " wrong image.")

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

        self.assertListEqual(self.dataqdyqdx.metadata['center_px'],
                             expected_center_px,
                             msg="The horizontal flip "
                             "resulted in the wrong center pixel location.")

        self.dataqdyqdx.reset_image_orientation()

    def test_flip_vertically(self):
        expected_img = np.array(
            [[15954.,  3109., 35295., 61517.],
             [10592., 19847., 10893., 41683.],
             [18439., 82922., 99999., 91001.],
             [60019., 70069.,   857., 65941.],
             [ 4701.,  9052., 61100., 46325.],
             [64811., 49702., 45711., 77210.],
             [44386., 20215., 48676., 20338.]]
        )
        expected_qys = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566, -0.000569693132,
                                 -0.000854539696, -0.001139386259])
        expected_qxzs = np.array([0.000569693132, 0.000284846566, 0.,
                                 -0.000284846566])

        expected_center_px = [2, 2]

        self.dataqdyqdx.reset_image_orientation()
        self.dataqdyqdx.flip_vertically()

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, expected_img,
            err_msg="The horizontal flip resulted in the"
            " wrong image.")

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

        self.assertListEqual(self.dataqdyqdx.metadata['center_px'],
                             expected_center_px,
                             msg="The horizontal flip "
                             "resulted in the wrong center pixel location.")

        self.dataqdyqdx.reset_image_orientation()

    def test_dataqdyqdx_reset_image_orientation(self):
        self.dataqdyqdx.rotate_image(90)
        self.dataqdyqdx.flip_vertically()
        self.dataqdyqdx.rotate_image(90)
        self.dataqdyqdx.flip_vertically()
        self.dataqdyqdx.flip_horizontally()
        self.dataqdyqdx.rotate_image(-180)
        self.dataqdyqdx.rotate_image(270)
        self.dataqdyqdx.flip_horizontally()
        self.dataqdyqdx.rotate_image(270)
        self.dataqdyqdx.rotate_image(90, direction='cw')
        self.dataqdyqdx.reset_image_orientation()

        np.testing.assert_array_equal(
            self.dataqdyqdx.image, self.image)

        qdy = np.array([0.001139386259, 0.000854539696, 0.000569693132,
                        0.000284846566, 0., -0.000284846566,
                        -0.000569693132])
        qdx = np.array([0.000569693132, 0.000284846566, 0., -0.000284846566])

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdx, qdx, decimal=12)

        np.testing.assert_array_almost_equal(
            self.dataqdyqdx.qdy, qdy, decimal=12)

        self.assertListEqual(self.dataqdyqdx.metadata['center_px'],
                             self.metadata['center_px'])

    def test_dataqdyqdx_integrate_box_sum_qdy(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box(
            limits_qdy_px=[3, 6],
            limits_qdx_px=[2, 4],
            mode='sum',
            axis='qdy',
        )

        q = np.array([0., -0.000284846566])
        Iq = np.array([111749., 198625.])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_sum_qdx(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box(
            limits_qdy_px=[3, 6],
            limits_qdx_px=[2, 4],
            mode='sum',
            axis='qdx',
        )

        q = np.array([0.000284846566, 0., -0.000284846566])
        Iq = np.array([66798., 191000., 52576.])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_mean_qdy(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box(
            limits_qdy_px=[3, 6],
            limits_qdx_px=[2, 4],
            mode='mean',
            axis='qdy',
        )

        q = np.array([0., -0.000284846566])
        Iq = np.array([37249.6666666667, 66208.3333333333])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_mean_qdx(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box(
            limits_qdy_px=[3, 6],
            limits_qdx_px=[2, 4],
            mode='mean',
            axis='qdx',
        )

        q = np.array([0.000284846566, 0., -0.000284846566])
        Iq = np.array([33399., 95500., 26288.])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_of_size_sum_qdy(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box_of_size(
            size_qdy_px=3,
            size_qdx_px=2,
            mode='sum',
            axis='qdy',
            offset_qdx_px=-1,
        )

        q = np.array([0., -0.000284846566])
        Iq = np.array([111749., 198625.])

        print(integrated_q_slice.limits_axis0, integrated_q_slice.limits_axis1)

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_of_size_sum_qdx(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box_of_size(
            size_qdy_px=3,
            size_qdx_px=2,
            mode='sum',
            axis='qdx',
            offset_qdx_px=-1,
        )

        q = np.array([0.000284846566, 0., -0.000284846566])
        Iq = np.array([66798., 191000., 52576.])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_of_size_mean_qdy(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box_of_size(
            size_qdy_px=3,
            size_qdx_px=2,
            mode='mean',
            axis='qdy',
            offset_qdx_px=-1,
        )

        q = np.array([0., -0.000284846566])
        Iq = np.array([37249.6666666667, 66208.3333333333])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_of_size_mean_qdx(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box_of_size(
            size_qdy_px=3,
            size_qdx_px=2,
            mode='mean',
            axis='qdx',
            offset_qdx_px=-1,
        )

        q = np.array([0.000284846566, 0., -0.000284846566])
        Iq = np.array([33399., 95500., 26288.])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_of_q_range_sum_qdy(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box_of_q_range(
            range_qdy=[-0.0003, 0.0003],
            range_qdx=[0.0001, -0.0003],
            mode='sum',
            axis='qdy',
        )

        q = np.array([0., -0.000284846566])
        Iq = np.array([111749., 198625.])

        print(integrated_q_slice.limits_axis0, integrated_q_slice.limits_axis1)

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_of_q_range_sum_qdx(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box_of_q_range(
            range_qdy=[-0.0003, 0.0003],
            range_qdx=[0.0001, -0.0003],
            mode='sum',
            axis='qdx',
        )

        q = np.array([0.000284846566, 0., -0.000284846566])
        Iq = np.array([66798., 191000., 52576.])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_of_q_range_mean_qdy(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box_of_q_range(
            range_qdy=[-0.0003, 0.0003],
            range_qdx=[0.0001, -0.0003],
            mode='mean',
            axis='qdy',
        )

        q = np.array([0., -0.000284846566])
        Iq = np.array([37249.6666666667, 66208.3333333333])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)

    def test_dataqdyqdx_integrate_box_of_q_range_mean_qdx(self):
        integrated_q_slice = self.dataqdyqdx.integrate_box_of_q_range(
            range_qdy=[-0.0003, 0.0003],
            range_qdx=[0.0001, -0.0003],
            mode='mean',
            axis='qdx',
        )

        q = np.array([0.000284846566, 0., -0.000284846566])
        Iq = np.array([33399., 95500., 26288.])

        np.testing.assert_array_almost_equal(
            integrated_q_slice.q, q, decimal=12)
        np.testing.assert_array_almost_equal(
            integrated_q_slice.Iq, Iq)
