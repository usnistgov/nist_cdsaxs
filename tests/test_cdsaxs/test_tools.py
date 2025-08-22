import unittest

import numpy as np

from cdsaxs.tools import find_gaussian_peakloc, line_fit


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

    def test_gaussian_find_peaks_2D(self):
        """
        TODO: implement
        The referenced function in tools may actually need to be
        revised as it is very convoluted right now and could be
        broken up into more useful modular components shareable with
        other functions.
        """
        pass

    def test_rotate_image(self):
        """
        TODO: implement
        """
        pass
