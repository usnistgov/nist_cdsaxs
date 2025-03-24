import numpy as np
import unittest

from src.cdsaxs.data import DataQyQxz, Dataset


class TestDataQyQxz(unittest.TestCase):

    def setUp(self):
        self.img = np.array(
            [[44386., 20215., 48676., 20338.],
             [64811., 49702., 45711., 77210.],
             [ 4701.,  9052., 61100., 46325.],
             [60019., 70069.,   857., 65941.],
             [18439., 82922., 99999., 91001.],
             [10592., 19847., 10893., 41683.],
             [15954.,  3109., 35295., 61517.]])

        self.qxzs = np.array([0.00063199, 0.000316, 0., -0.000316])
        self.qys = np.array([0.00126399, 0.00094799, 0.00063199, 0.000316, 0.,
                        -0.000316, -0.00063199])
        self.metadata = {
            'center_px': [4, 2]
        }
        self.params = {
            'custom_param': 20
        }

        self.test_dataqyqxz = DataQyQxz(
            self.img, self.qys, self.qxzs, self.metadata, params=self.params)

    def test_dataqyqxz_dimensions(self):
        with self.assertRaises(ValueError):
            DataQyQxz(self.img, self.qxzs, self.qys, self.metadata)
    
    def test_dataqyqxz_metadata_keywords(self):
        with self.assertRaises(ValueError):
            DataQyQxz(self.img, self.qys, self.qxzs, {'not_a_keyword': 2})
        
    def test_dataqyqxz_metadata_pass(self):
        self.assertDictEqual(self.test_dataqyqxz.metadata, self.metadata)

    def test_dataqyqxz_params_pass(self):
        self.assertDictEqual(self.test_dataqyqxz.params, self.params)

    def test_dataqyqxz_rotate_ccw90(self):
        expected_img = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   857., 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        )
        expected_qxzs = np.array([0.00126399, 0.00094799, 0.00063199,
                                   0.000316,  0., -0.000316, -0.00063199])
        expected_qys = np.array([0.000316,  0., -0.000316, -0.00063199])
        expected_center_px = [1, 4]

        self.test_dataqyqxz.reset_rotations()
        self.test_dataqyqxz.rotate_image(90)

        np.testing.assert_array_equal(
            self.test_dataqyqxz.imgdata, expected_img,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "wrong image.")
        
        np.testing.assert_array_equal(
            self.test_dataqyqxz.qxzs, expected_qxzs,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qys, expected_qys,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "incorrect qxzs array."
        )

        self.assertListEqual(self.test_dataqyqxz.metadata['center_px'],
                             expected_center_px,
                             msg="The counter-clockwise 90 degree rotation "
                             "resulted in the wrong center pixel location.")
        
        self.test_dataqyqxz.reset_rotations()

    def test_dataqyqxz_rotate_ccw180(self):
        expected_img = np.array(
            [[61517., 35295.,  3109., 15954.],
             [41683., 10893., 19847., 10592.],
             [91001., 99999., 82922., 18439.],
             [65941.,   857., 70069., 60019.],
             [46325., 61100.,  9052.,  4701.],
             [77210., 45711., 49702., 64811.],
             [20338., 48676., 20215., 44386.]]
        )
        expected_qxzs = np.array([0.000316, -0., -0.000316, -0.00063199])
        expected_qys = np.array([0.00063199, 0.000316, -0., -0.000316,
                                 -0.00063199, -0.00094799, -0.00126399])
        expected_center_px = [2, 1]

        self.test_dataqyqxz.reset_rotations()
        self.test_dataqyqxz.rotate_image(180)

        np.testing.assert_array_equal(
            self.test_dataqyqxz.imgdata, expected_img,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qxzs, expected_qxzs,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qys, expected_qys,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertListEqual(self.test_dataqyqxz.metadata['center_px'],
                             expected_center_px,
                             msg="The counter-clockwise 180 degree rotation "
                             "resulted in the wrong center pixel location.")

        self.test_dataqyqxz.reset_rotations()

    def test_dataqyqxz_rotate_ccw270(self):
        expected_img = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        )
        expected_qxzs = np.array([0.00063199, 0.000316, -0., -0.000316,
                                  -0.00063199, -0.00094799, -0.00126399])
        expected_qys = np.array([0.00063199, 0.000316, 0., -0.000316])
        expected_center_px = [2, 2]

        self.test_dataqyqxz.reset_rotations()
        self.test_dataqyqxz.rotate_image(270)

        np.testing.assert_array_equal(
            self.test_dataqyqxz.imgdata, expected_img,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qxzs, expected_qxzs,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qys, expected_qys,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertListEqual(self.test_dataqyqxz.metadata['center_px'],
                             expected_center_px,
                             msg="The counter-clockwise 182700 degree rotation"
                             " resulted in the wrong center pixel location.")

        self.test_dataqyqxz.reset_rotations()

    def test_dataqyqxz_rotate_ccw_m90(self):
        expected_img = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        )
        expected_qxzs = np.array([0.00063199, 0.000316, -0., -0.000316,
                                  -0.00063199, -0.00094799, -0.00126399])
        expected_qys = np.array([0.00063199, 0.000316, 0., -0.000316])
        expected_center_px = [2, 2]

        self.test_dataqyqxz.reset_rotations()
        self.test_dataqyqxz.rotate_image(-90)

        np.testing.assert_array_equal(
            self.test_dataqyqxz.imgdata, expected_img,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qxzs, expected_qxzs,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qys, expected_qys,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertListEqual(self.test_dataqyqxz.metadata['center_px'],
                             expected_center_px,
                             msg="The counter-clockwise -90 degree rotation "
                             "resulted in the wrong center pixel location.")

        self.test_dataqyqxz.reset_rotations()


    def test_dataqyqxz_rotate_cw90(self):
        expected_img = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   857., 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        )
        expected_qxzs = np.array([0.00063199, 0.000316, -0., -0.000316,
                                  -0.00063199, -0.00094799, -0.00126399])
        expected_qys = np.array([0.00063199, 0.000316, 0., -0.000316])
        expected_center_px = [2, 2]

        self.test_dataqyqxz.reset_rotations()
        self.test_dataqyqxz.rotate_image(90, direction='cw')

        np.testing.assert_array_equal(
            self.test_dataqyqxz.imgdata, expected_img,
            err_msg="The clockwise -90 degree rotation resulted in the"
            " wrong image.")

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qxzs, expected_qxzs,
            err_msg="The clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        np.testing.assert_array_equal(
            self.test_dataqyqxz.qys, expected_qys,
            err_msg="The clockwise -90 degree rotation resulted in the"
            " incorrect qxzs array."
        )

        self.assertListEqual(self.test_dataqyqxz.metadata['center_px'],
                             expected_center_px,
                             msg="The clockwise -90 degree rotation "
                             "resulted in the wrong center pixel location.")

        self.test_dataqyqxz.reset_rotations()



if __name__ == 'main':
    unittest.main()
