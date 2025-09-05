import unittest

import numpy as np

from cdsaxs.tools import find_gaussian_peakloc, line_fit
from cdsaxs.tools import gaussian_refine_peak_2D
from cdsaxs.tools import find_peaks_2D, find_peaks_1D
from cdsaxs.tools import find_peaks_2D_one_axis


class TestTools(unittest.TestCase):

    def test_find_gaussian_peakloc(self):

        x = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3]
        y = [
            2.0474927095, 2.1699622264, 2.4737009498, 3.0282115523,
            3.7381493166, 4.2883268928, 4.3462561639, 3.8735236002,
            3.1651163299, 2.5642944643, 2.2128475571, 2.0625256089,
            2.0143045292
        ]
        y = np.array(y)
        mean = 1.4
        std_dev = 0.5
        scale = 3
        offset = 2

        peak_x, peak_index, popt = find_gaussian_peakloc(x, y)

        self.assertAlmostEqual(peak_x, mean)
        self.assertEqual(peak_index, 6)

        self.assertAlmostEqual(popt[0], mean)
        self.assertAlmostEqual(popt[1], std_dev)
        self.assertAlmostEqual(popt[2], scale)
        self.assertAlmostEqual(popt[3], offset)

    def test_line_fit(self):

        x = [-1, 0, 1, 2, 3, 4]
        y = [10.2, 12.6, 15, 17.4, 19.8, 22.2]

        angle, slope, intercept = line_fit(x, y)
        self.assertAlmostEqual(angle, 67.3801350520)
        self.assertAlmostEqual(slope, 2.4)
        self.assertAlmostEqual(intercept, 12.6)

    def test_gaussian_refine_peak_2D(self):

        image = np.array([
            [0.5436, 0.0083, 0.6645, 0.3204, 0.9405, 0.3734, 0.6948, 0.8988,
             0.6844, 0.2968, 0.2876, 0.1235, 0.2931, 0.1992, 0.4979],
            [0.1173, 0.5148, 0.6833, 0.9839, 0.3896, 0.6532, 0.0318, 0.8793,
             0.1335, 0.0493, 0.2478, 0.6803, 0.7272, 0.9512, 0.1966],
            [0.2024, 0.1881, 0.4839, 0.7944, 0.0651, 0.3514, 0.4952, 0.0073,
             0.6832, 0.5360, 0.1913, 0.7291, 0.8701, 0.7656, 0.3790],
            [0.8093, 0.2981, 0.9414, 0.8002, 0.2963, 0.5475, 0.8056, 0.2262,
             0.5344, 0.7771, 0.5599, 0.0472, 0.7013, 0.0970, 0.5859],
            [0.2698, 0.5891, 0.9991, 0.3844, 0.9458, 0.8127, 4.0000, 0.4554,
             0.3991, 0.0572, 0.1760, 0.3514, 0.0607, 0.8751, 0.5205],
            [0.1968, 0.6365, 0.4319, 0.5466, 0.1821, 0.2887, 9.0000, 0.2034,
             0.0224, 0.8485, 0.7354, 0.4876, 0.0137, 0.5752, 0.2551],
            [0.4350, 0.9826, 0.7620, 0.4841, 0.4719, 4.5000, 10.000, 3.5000,
             0.6295, 0.6860, 0.5790, 0.0243, 0.4646, 0.3012, 0.3719],
            [0.2234, 0.3080, 0.2304, 4.0000, 10.000, 12.000, 15.000, 11.000,
             9.0000, 0.6077, 0.3871, 0.1316, 0.4788, 0.3031, 0.8412],
            [0.2733, 0.0197, 0.3131, 0.7445, 0.0995, 3.5000, 9.5000, 2.5000,
             0.0681, 0.2736, 0.1964, 0.9449, 0.0104, 0.8013, 0.0800],
            [0.0848, 0.7899, 0.1518, 0.1052, 0.4164, 0.6523, 8.5000, 0.2190,
             0.4408, 0.0898, 0.7812, 0.8715, 0.2338, 0.4162, 0.3991],
        ])

        image0 = np.array([6.8269, 7.2390, 6.7417, 8.0276, 10.8963, 14.4239,
                           24.1923, 64.5114, 19.3250, 14.1517])
        a_opt, _, _ = find_gaussian_peakloc(
            np.arange(0, len(image0)), image0
        )

        image1 = np.array([3.1556, 4.3351, 5.6614, 9.1637, 13.8071, 23.6793,
                           58.0274, 19.8895, 12.5953, 4.2221, 4.1417, 4.3914,
                           3.8536, 5.2852, 4.1272])
        b_opt, _, _ = find_gaussian_peakloc(
            np.arange(0, len(image1)), image1
        )

        a_opt_test, b_opt_test = gaussian_refine_peak_2D(image)

        self.assertAlmostEqual(a_opt_test, a_opt, 5)
        self.assertAlmostEqual(b_opt_test, b_opt, 5)

    def test_rotate_image(self):
        """
        TODO: implement
        """
        pass

    def test_find_peaks_2D(self):
        image = np.array(
            [[0.91913391, 3.        , 0.91913391, 0.        , 0.        ,
                0.57987207],
             [3.        , 5.        , 3.        , 0.        , 0.        ,
                0.57987207],
             [0.91913391, 3.        , 0.91913391, 0.        , 0.        ,
                0.57987207],
             [0.        , 0.        , 0.        , 0.        , 0.        ,
                0.57987207],
             [0.        , 0.        , 0.        , 0.        , 0.        ,
                0.57987207],
             [0.76574962, 0.66395775, 0.90228066, 2.        , 3.        ,
                2.        ],
             [0.51796119, 0.59285603, 0.5356811 , 3.        , 4.        ,
                3.        ],
             [0.46730955, 0.62937721, 0.3429157 , 2.        , 3.        ,
                2.        ],
             [0.10071061, 0.10071061, 0.10071061, 0.10071061, 0.10071061,
                0.57987207],
             [0.10071061, 0.30071061, 0.30071061, 0.30071061, 0.30071061,
                0.57987207],
             [0.10071061, 0.30071061, 2.1       , 2.        , 0.30071061,
                0.57987207],
             [0.10071061, 0.30071061, 0.30071061, 0.30071061, 0.30071061,
                0.57987207],
             [0.10071061, 0.10071061, 0.10071061, 0.10071061, 0.10071061,
                0.57987207]])

        peak_coordinates = [
            [1.0, 1.0],
            [5.9955356927226766, 4.1173757322214435],
            [10.0, 2.480795490081457],
        ]

        test_coordinates = find_peaks_2D(image, log_scale=False,
                                         refinement_size=5, threshold_abs=1)

        for actual, test in zip(peak_coordinates, test_coordinates):
            self.assertListEqual(test, actual)

    def test_find_peaks_2D_refinement_width_check(self):
        image = np.array(
            [[ 3.        ,  0.91913391,  0.        ,  0.        ],
             [10.        ,  3.        ,  0.        ,  0.        ],
             [ 3.        ,  0.91913391,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.66395775,  0.90228066,  2.        ,  3.        ],
             [ 0.59285603,  0.5356811 ,  3.        ,  4.        ],
             [ 0.62937721,  0.3429157 ,  2.        ,  3.        ],
             [ 0.10071061,  0.10071061,  0.10071061,  0.10071061],
             [ 0.30071061,  0.30071061,  0.30071061,  0.30071061],
             [ 0.30071061,  2.1       ,  2.        ,  0.30071061],
             [ 0.30071061,  0.30071061,  0.30071061,  0.30071061],
             [ 0.10071061,  0.10071061,  0.10071061,  0.10071061]])

        peak_coordinates = [
            [1, 0],
            [6, 3],
            [10, 1],
        ]

        test_coordinates = find_peaks_2D(image, log_scale=False,
                                         refinement_size=3, threshold_abs=1,
                                         exclude_border=False)

        for actual, test in zip(peak_coordinates, test_coordinates):
            self.assertListEqual(test, actual)

    def test_find_peaks_1D(self):
        image = np.array(
            [[ 3.        ,  0.91913391,  0.        ,  0.        ],
             [10.        ,  3.        ,  0.        ,  0.        ],
             [ 3.        ,  0.91913391,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.66395775,  0.90228066,  2.        ,  3.        ],
             [ 0.59285603,  0.5356811 ,  3.        ,  4.        ],
             [ 0.62937721,  0.3429157 ,  2.        ,  3.        ],
             [ 0.10071061,  0.10071061,  0.10071061,  0.10071061],
             [ 0.30071061,  0.30071061,  0.30071061,  0.30071061],
             [ 0.30071061,  2.1       ,  2.        ,  0.30071061],
             [ 0.30071061,  0.30071061,  0.30071061,  0.30071061],
             [ 0.10071061,  0.10071061,  0.10071061,  0.10071061]])

        data = np.nansum(image, axis=1)[:10]
        peak_coordinates = [
            0.999827841102188,
            5.961187929025904,
        ]

        test_coordinates = find_peaks_1D(data, log_scale=False,
                                         refinement_size=8, threshold_abs=1,
                                         exclude_border=False, min_distance=2)

        self.assertListEqual(test_coordinates, peak_coordinates)

    def test_find_peaks_1D_scipy(self):
        image = np.array(
            [[ 3.        ,  0.91913391,  0.        ,  0.        ],
             [10.        ,  3.        ,  0.        ,  0.        ],
             [ 3.        ,  0.91913391,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.66395775,  0.90228066,  2.        ,  3.        ],
             [ 0.59285603,  0.5356811 ,  3.        ,  4.        ],
             [ 0.62937721,  0.3429157 ,  2.        ,  3.        ],
             [ 0.10071061,  0.10071061,  0.10071061,  0.10071061],
             [ 0.30071061,  0.30071061,  0.30071061,  0.30071061],
             [ 0.30071061,  2.1       ,  2.        ,  0.30071061],
             [ 0.30071061,  0.30071061,  0.30071061,  0.30071061],
             [ 0.10071061,  0.10071061,  0.10071061,  0.10071061]])

        data = np.nansum(image, axis=1)[:10]
        peak_coordinates = [
            0.999827841102188,
            5.961187929025904,
        ]

        test_coordinates = find_peaks_1D(data, log_scale=False,
                                         refinement_size=8, algorithm='scipy')

        self.assertListEqual(test_coordinates, peak_coordinates)

    def test_find_peaks_2D_one_axis(self):
        image = np.array(
            [[ 3.        ,  0.91913391,  0.        ,  0.        ],
             [10.        ,  3.        ,  0.        ,  0.        ],
             [ 3.        ,  0.91913391,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.66395775,  0.90228066,  2.        ,  3.        ],
             [ 0.59285603,  0.5356811 ,  3.        ,  4.        ],
             [ 0.62937721,  0.3429157 ,  2.        ,  3.        ],
             [ 0.10071061,  0.10071061,  0.10071061,  0.10071061],
             [ 0.30071061,  0.30071061,  0.30071061,  0.30071061]])

        peak_coordinates = [
            [0.999827841102188, 0.0],
            [5.961187929025904, 2.5404629363356612],
        ]

        test_coordinates = find_peaks_2D_one_axis(
            image, peak_axis=0, log_scale=False,
            refinement_size=8, threshold_abs=1,
            exclude_border=False)

        for actual, test in zip(peak_coordinates, test_coordinates):
            self.assertListEqual(test, actual)

    def test_find_peaks_2D_one_axis_scipy(self):
        image = np.array(
            [[ 3.        ,  0.91913391,  0.        ,  0.        ],
             [10.        ,  3.        ,  0.        ,  0.        ],
             [ 3.        ,  0.91913391,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.66395775,  0.90228066,  2.        ,  3.        ],
             [ 0.59285603,  0.5356811 ,  3.        ,  4.        ],
             [ 0.62937721,  0.3429157 ,  2.        ,  3.        ],
             [ 0.10071061,  0.10071061,  0.10071061,  0.10071061],
             [ 0.30071061,  0.30071061,  0.30071061,  0.30071061]])

        peak_coordinates = [
            [0.999827841102188, 0.0],
            [5.961187929025904, 2.5404629363356612],
        ]

        test_coordinates = find_peaks_2D_one_axis(
            image, peak_axis=0, log_scale=False,
            refinement_size=8, algorithm='scipy')

        for actual, test in zip(peak_coordinates, test_coordinates):
            self.assertListEqual(test, actual)
