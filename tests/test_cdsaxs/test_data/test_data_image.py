import numpy as np
import unittest

from cdsaxs.data.data_image import DataImage


class TestDataImage(unittest.TestCase):

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

        self.data2d = DataImage(self.image, mask=self.custom_mask)

    def test_init(self):
        np.testing.assert_array_equal(self.data2d.image, self.image)
        np.testing.assert_array_equal(self.data2d.mask,
                                      self.custom_mask + np.isnan(self.image))

    def test_mask_points(self):
        mask = self.image == 99999
        self.data2d.mask_points(mask)

        mask[0, 0] = True
        mask[3, 2] = True
        np.testing.assert_array_equal(self.data2d.mask, mask)

    def test_overwrite_mask(self):
        mask = self.image == 99999
        self.data2d.overwrite_mask(mask)

        np.testing.assert_array_equal(self.data2d.mask, mask)

    def test_reset_mask(self):
        mask = np.isnan(self.image)
        self.data2d.reset_mask()

        np.testing.assert_array_equal(self.data2d.mask, mask)

    def test_data2d_rotate_ccw90(self):
        """test 90 degree rotation counter-clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   np.nan, 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)

        self.data2d.rotate_image_ccw(1)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The counter-clockwise 90 degree rotation resulted in the "
            "wrong image.")

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

        self.data2d.rotate_image_ccw(2)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The counter-clockwise 180 degree rotation resulted in the"
            " wrong image.")

    def test_data2d_rotate_ccw270(self):
        """test 270 degree rotation counter-clockwise"""
        expected_image = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   np.nan, 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)

        self.data2d.rotate_image_ccw(3)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The counter-clockwise 270 degree rotation resulted in the"
            " wrong image.")

    def test_data2d_rotate_ccw_m90(self):
        """test -90 degree rotation counter-clockwise (90 degree clockwise)"""
        expected_image = np.array(
            [[15954., 10592., 18439., 60019.,  4701., 64811., 44386.],
             [ 3109., 19847., 82922., 70069.,  9052., 49702., 20215.],
             [35295., 10893., 99999.,   np.nan, 61100., 45711., 48676.],
             [61517., 41683., 91001., 65941., 46325., 77210., 20338.]]
        ).astype(np.float64)

        self.data2d.rotate_image_ccw(-1)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The counter-clockwise -90 degree rotation resulted in the"
            " wrong image.")

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

        self.data2d.rotate_image_ccw(-2)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The clockwise 180 degree rotation resulted in the"
            " wrong image.")

    def test_data2d_rotate_cw270(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   np.nan, 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)

        self.data2d.rotate_image_ccw(-3)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The clockwise 270 degree rotation resulted in the"
            " wrong image.")

    def test_data2d_rotate_cw630(self):
        """test 90 degree rotation clockwise"""
        expected_image = np.array(
            [[20338., 77210., 46325., 65941., 91001., 41683., 61517.],
             [48676., 45711., 61100.,   np.nan, 99999., 10893., 35295.],
             [20215., 49702.,  9052., 70069., 82922., 19847.,  3109.],
             [44386., 64811.,  4701., 60019., 18439., 10592., 15954.]]
        ).astype(np.float64)

        self.data2d.rotate_image_ccw(-7)

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The clockwise 630 degree rotation resulted in the"
            " wrong image.")

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

        self.data2d.flip_horizontally()

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The horizontal flip resulted in the wrong image."
        )

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

        self.data2d.flip_vertically()

        np.testing.assert_array_equal(
            self.data2d.image, expected_image,
            err_msg="The vertical flip resulted in the wrong image."
        )

    def test_data2d_reset_image(self):
        """test the reset_image"""

        self.data2d.rotate_image_ccw(1)
        self.data2d.rotate_image_ccw(1)
        self.data2d.flip_horizontally()
        self.data2d.rotate_image_ccw(-2)
        self.data2d.scale_data(20)
        self.data2d.flip_vertically()
        self.data2d.rotate_image_ccw(3)
        self.data2d.rotate_image_ccw(3)
        self.data2d.flip_horizontally()
        self.data2d.flip_vertically()
        self.data2d.rotate_image_ccw(-1)

        self.data2d.reset_image()
        np.testing.assert_array_equal(
            self.data2d.image, self.image)

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
        self.data2d.add_to_data(value)
        np.testing.assert_array_equal(self.data2d.image, new_image)
        self.assertTupleEqual(self.data2d._data_transformations[-1],
                              ("add", value))

    def test_add_to_data_array(self):
        value = 2
        value = np.ones_like(self.data2d.image, dtype=float)*value
        new_image = np.array([
            [44388, 20217, 48678, 20340],
            [64813, 49704, 45713, 77212],
            [4703, 9054, 61102, 46327],
            [60021, 70071, np.nan, 65943],
            [18441, 82924, 100001, 91003],
            [10594, 19849, 10895, 41685],
            [15956, 3111, 35297, 61519],
        ])
        self.data2d.add_to_data(value)
        np.testing.assert_array_equal(self.data2d.image, new_image)
        self.assertTupleEqual(self.data2d._data_transformations[-1],
                              ("add", value))

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
        self.data2d.subtract_from_data(value)
        np.testing.assert_array_equal(self.data2d.image, new_image)
        self.assertTupleEqual(self.data2d._data_transformations[-1],
                              ("subtract", value))

    def test_subtract_from_data_array(self):
        value = 2
        value = np.ones_like(self.data2d.image, dtype=float)*value
        new_image = np.array([
            [44384, 20213, 48674, 20336],
            [64809, 49700, 45709, 77208],
            [4699, 9050, 61098, 46323],
            [60017, 70067, np.nan, 65939],
            [18437, 82920, 99997, 90999],
            [10590, 19845, 10891, 41681],
            [15952, 3107, 35293, 61515],
        ])
        self.data2d.subtract_from_data(value)
        np.testing.assert_array_equal(self.data2d.image, new_image)
        self.assertTupleEqual(self.data2d._data_transformations[-1],
                              ("subtract", value))

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
        self.data2d.scale_data(value)
        np.testing.assert_array_equal(self.data2d.image, new_image)
        self.assertTupleEqual(self.data2d._data_transformations[-1],
                              ("scale", value))

    def test_scale_data_array(self):
        value = 2
        value = np.ones_like(self.data2d.image, dtype=float)*value
        new_image = np.array([
            [88772, 40430, 97352, 40676],
            [129622, 99404, 91422, 154420],
            [9402, 18104, 122200, 92650],
            [120038, 140138, np.nan, 131882],
            [36878, 165844, 199998, 182002],
            [21184, 39694, 21786, 83366],
            [31908, 6218, 70590, 123034],
        ])
        self.data2d.scale_data(value)
        np.testing.assert_array_equal(self.data2d.image, new_image)
        self.assertTupleEqual(self.data2d._data_transformations[-1],
                              ("scale", value))

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
        self.data2d.normalize_data(value)
        np.testing.assert_array_equal(self.data2d.image, new_image)
        self.assertTupleEqual(self.data2d._data_transformations[-1],
                              ("normalize", value))
 
    def test_normalize_data_array(self):
        value = 2
        value = np.ones_like(self.data2d.image, dtype=float)*value
        new_image = np.array([
            [22193, 10107.5, 24338, 10169],
            [32405.5, 24851, 22855.5, 38605],
            [2350.5, 4526, 30550, 23162.5],
            [30009.5, 35034.5, np.nan, 32970.5],
            [9219.5, 41461, 49999.5, 45500.5],
            [5296, 9923.5, 5446.5, 20841.5],
            [7977, 1554.5, 17647.5, 30758.5],
        ])
        self.data2d.normalize_data(value)
        np.testing.assert_array_equal(self.data2d.image, new_image)
        self.assertTupleEqual(self.data2d._data_transformations[-1],
                              ("normalize", value))

    def test_reset_intensity(self):
        self.data2d.normalize_data(2)
        self.data2d.subtract_from_data(2)
        self.data2d.add_to_data(5)
        self.data2d.scale_data(1.3)
        self.data2d.reset_intensity()
        np.testing.assert_array_almost_equal(self.data2d.image, self.image)

    def test_data2d_sum_box_axis0(self):
        intensity, image_box, mask = self.data2d.sum_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            axis=0
        )
        int_i_expected = np.array([np.nan, 198625.,])
        np.testing.assert_array_almost_equal(intensity, int_i_expected)

        image_box_expected = np.array(
            [[np.nan, 65941.],
             [99999., 91001.],
             [10893., 41683.]])
        np.testing.assert_array_equal(image_box, image_box_expected)

        mask_expected = np.array(
            [[True, False],
             [False, False],
             [False, False]])
        np.testing.assert_array_equal(mask, mask_expected)

    def test_data2d_sum_box_axis1(self):
        intensity, image_box, mask = self.data2d.sum_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            axis=1
        )

        int_i_expected = np.array([np.nan, 191000., 52576.])
        np.testing.assert_array_almost_equal(intensity, int_i_expected)

        image_box_expected = np.array(
            [[np.nan, 65941.],
             [99999., 91001.],
             [10893., 41683.]])
        np.testing.assert_array_equal(image_box, image_box_expected)

        mask_expected = np.array(
            [[True, False],
             [False, False],
             [False, False]])
        np.testing.assert_array_equal(mask, mask_expected)

    def test_data2d_mean_box_axis0(self):
        intensity, image_box, mask = self.data2d.mean_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            axis=0
        )
        int_i_expected = np.array([np.nan, 66208.3333333333])
        np.testing.assert_array_almost_equal(intensity, int_i_expected)

        image_box_expected = np.array(
            [[np.nan, 65941.],
             [99999., 91001.],
             [10893., 41683.]])
        np.testing.assert_array_equal(image_box, image_box_expected)

        mask_expected = np.array(
            [[True, False],
             [False, False],
             [False, False]])
        np.testing.assert_array_equal(mask, mask_expected)

    def test_data2d_mean_box_axis1(self):
        intensity, image_box, mask = self.data2d.mean_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            axis=1
        )

        int_i_expected = np.array([np.nan, 95500., 26288.])
        np.testing.assert_array_almost_equal(intensity, int_i_expected)

        image_box_expected = np.array(
            [[np.nan, 65941.],
             [99999., 91001.],
             [10893., 41683.]])
        np.testing.assert_array_equal(image_box, image_box_expected)

        mask_expected = np.array(
            [[True, False],
             [False, False],
             [False, False]])
        np.testing.assert_array_equal(mask, mask_expected)

    def test_slice_box_mean(self):
        intensity, image_box, mask = self.data2d.slice_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            axis=1,
            mode='mean'
        )

        int_i_expected = np.array([np.nan, 95500., 26288.])
        np.testing.assert_array_almost_equal(intensity, int_i_expected)

        image_box_expected = np.array(
            [[np.nan, 65941.],
             [99999., 91001.],
             [10893., 41683.]])
        np.testing.assert_array_equal(image_box, image_box_expected)

        mask_expected = np.array(
            [[True, False],
             [False, False],
             [False, False]])
        np.testing.assert_array_equal(mask, mask_expected)

    def test_slice_box_sum(self):
        intensity, image_box, mask = self.data2d.slice_box(
            limits_axis0=[3, 6],
            limits_axis1=[2, 4],
            axis=0,
            mode='sum'
        )
        int_i_expected = np.array([np.nan, 198625.,])
        np.testing.assert_array_almost_equal(intensity, int_i_expected)

        image_box_expected = np.array(
            [[np.nan, 65941.],
             [99999., 91001.],
             [10893., 41683.]])
        np.testing.assert_array_equal(image_box, image_box_expected)

        mask_expected = np.array(
            [[True, False],
             [False, False],
             [False, False]])
        np.testing.assert_array_equal(mask, mask_expected)

    def test_rotate_image(self):
        """
        TODO: implement a test for the image rotation
        This method uses the Pillow package's rotate_image function.
        """
        pass
