import unittest

import numpy as np

from cdsaxs.tools import find_gaussian_peakloc, line_fit
from cdsaxs.tools import gaussian_refine_peak_2D
from cdsaxs.tools import find_peaks_2D, find_peaks_1D
from cdsaxs.tools import find_peaks_2D_one_axis
from cdsaxs.tools import estimate_sample_detector_distance
from cdsaxs.tools import estimate_sample_detector_distance_from_rings
from cdsaxs.tools import estimate_sample_detector_distance_from_ring_sectors
from cdsaxs.tools import _score_sdd_ring_candidate
from cdsaxs.tools import rotate_image


class TestTools(unittest.TestCase):

    @staticmethod
    def _generate_rotated_peak_positions(
            beam_center_px,
            sdd_cm,
            pitch_nm,
            wavelength_nm,
            pixel_size_um,
            orders,
            angle_deg):
        q_spacing = 2 * np.pi / pitch_nm
        angle_rad = np.deg2rad(angle_deg)
        direction = np.array([np.sin(angle_rad), np.cos(angle_rad)])
        peak_positions = []
        for order in orders:
            q = q_spacing * order
            theta = 2 * np.arcsin(q * wavelength_nm / (4 * np.pi))
            radial_distance_cm = sdd_cm * np.tan(theta)
            radial_distance_px = radial_distance_cm * 1e4 / pixel_size_um
            peak_positions.append(
                np.asarray(beam_center_px, dtype=float)
                + radial_distance_px * direction
            )
        return np.asarray(peak_positions)

    @staticmethod
    def _generate_peak_image(shape, peak_positions, sigma_px=1.2,
                             amplitude=100.0, background=1.0):
        row_grid, col_grid = np.indices(shape, dtype=float)
        image = np.full(shape, background, dtype=float)
        for row_peak, col_peak in peak_positions:
            exponent = (
                ((row_grid - row_peak) ** 2) / (2 * sigma_px ** 2)
                + ((col_grid - col_peak) ** 2) / (2 * sigma_px ** 2)
            )
            image += amplitude * np.exp(-exponent)
        return image

    @staticmethod
    def _generate_ring_sample_positions(
            beam_center_px,
            sdd_cm,
            pitch_nm,
            wavelength_nm,
            pixel_size_um,
            ring_indices,
            angles_deg):
        q_spacing = 2 * np.pi / pitch_nm
        peak_positions = []
        for ring_index in ring_indices:
            q = q_spacing * ring_index
            theta = 2 * np.arcsin(q * wavelength_nm / (4 * np.pi))
            radial_distance_cm = sdd_cm * np.tan(theta)
            radial_distance_px = radial_distance_cm * 1e4 / pixel_size_um
            for angle_deg in angles_deg:
                angle_rad = np.deg2rad(angle_deg)
                direction = np.array([
                    np.sin(angle_rad),
                    np.cos(angle_rad),
                ])
                peak_positions.append(
                    np.asarray(beam_center_px, dtype=float)
                    + radial_distance_px * direction
                )
        return np.asarray(peak_positions)

    @staticmethod
    def _generate_ring_arc_image(
            shape,
            beam_center_px,
            peak_positions,
            radial_sigma_px=1.2,
            tangential_sigma_px=6.0,
            amplitude=100.0,
            background=1.0):
        row_grid, col_grid = np.indices(shape, dtype=float)
        image = np.full(shape, background, dtype=float)
        beam_center_px = np.asarray(beam_center_px, dtype=float)

        for row_peak, col_peak in peak_positions:
            peak_position = np.asarray([row_peak, col_peak], dtype=float)
            radial_vector = peak_position - beam_center_px
            radial_norm = np.linalg.norm(radial_vector)
            radial_unit = radial_vector / radial_norm
            tangential_unit = np.array([-radial_unit[1], radial_unit[0]])

            delta_row = row_grid - row_peak
            delta_col = col_grid - col_peak
            radial_offset = (
                delta_row * radial_unit[0] + delta_col * radial_unit[1]
            )
            tangential_offset = (
                delta_row * tangential_unit[0]
                + delta_col * tangential_unit[1]
            )
            exponent = (
                (radial_offset ** 2) / (2 * radial_sigma_px ** 2)
                + (tangential_offset ** 2) / (2 * tangential_sigma_px ** 2)
            )
            image += amplitude * np.exp(-exponent)

        return image

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

        peak_x, popt = find_gaussian_peakloc(x, y)

        self.assertAlmostEqual(peak_x, mean)

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
        a_opt, _ = find_gaussian_peakloc(
            np.arange(0, len(image0)), image0
        )

        image1 = np.array([3.1556, 4.3351, 5.6614, 9.1637, 13.8071, 23.6793,
                           58.0274, 19.8895, 12.5953, 4.2221, 4.1417, 4.3914,
                           3.8536, 5.2852, 4.1272])
        b_opt, _ = find_gaussian_peakloc(
            np.arange(0, len(image1)), image1
        )

        a_opt_test, b_opt_test = gaussian_refine_peak_2D(image)

        self.assertAlmostEqual(a_opt_test, a_opt, 5)
        self.assertAlmostEqual(b_opt_test, b_opt, 5)

    # def test_rotate_image(self):
        # TODO: implement, although this is really a wrapper for
        # PILLOW.Image.rotate()
        # pass

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
            for x, y in zip(actual, test):
                self.assertAlmostEqual(x, y, places=4)

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
            for x, y in zip(actual, test):
                self.assertAlmostEqual(x, y, places=4)

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

        for x, y in zip(test_coordinates, peak_coordinates):
            self.assertAlmostEqual(x, y, places=3)

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

        for x, y in zip(test_coordinates, peak_coordinates):
            self.assertAlmostEqual(x, y, places=3)

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
            for x, y in zip(actual, test):
                self.assertAlmostEqual(x, y, places=3)

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
            for x, y in zip(actual, test):
                self.assertAlmostEqual(x, y, places=3)

    def test_estimate_sample_detector_distance_rotated_axis(self):
        beam_center_px = np.array([103.4, 87.2])
        sdd_cm = 215.0
        pitch_nm = 80.0
        wavelength_nm = 0.1
        pixel_size_um = 75.0
        peak_positions = self._generate_rotated_peak_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            orders=[-3, -2, -1, 1, 2, 3],
            angle_deg=33.0,
        )
        image = self._generate_peak_image((220, 220), peak_positions)

        sdd_fit_cm, uncertainty_cm, details = estimate_sample_detector_distance(
            image=image,
            peak_positions=peak_positions,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            beam_center_guess_px=(101.0, 90.0),
            beam_center_search_radius_px=20.0,
            sdd_search_range_cm=(150.0, 260.0),
            coarse_grid_points=21,
            fine_grid_points=21,
            return_details=True,
        )

        self.assertAlmostEqual(sdd_fit_cm, sdd_cm, delta=3.0)
        self.assertGreater(uncertainty_cm, 0)
        self.assertEqual(details["orders"].tolist(), [-3, -2, -1, 1, 2, 3])
        q_spacing = 2 * np.pi / pitch_nm
        np.testing.assert_allclose(
            details["q_values_nm_inverse"],
            q_spacing * np.array([-3, -2, -1, 1, 2, 3]),
            atol=5e-4,
        )
        self.assertAlmostEqual(
            details["beam_center_px"][0],
            beam_center_px[0],
            delta=2.0,
        )
        self.assertAlmostEqual(
            details["beam_center_px"][1],
            beam_center_px[1],
            delta=2.0,
        )
        self.assertAlmostEqual(
            np.linalg.norm(details["diffraction_direction"]),
            1.0,
            places=6,
        )
        self.assertGreaterEqual(details["score"], 0)
        self.assertEqual(details["peak_position_uncertainty_px"].shape, (6,))
        self.assertTrue(np.all(details["peak_position_uncertainty_px"] > 0))

    def test_estimate_sample_detector_distance_refine_with_orders(self):
        beam_center_px = np.array([103.4, 87.2])
        sdd_cm = 215.0
        pitch_nm = 80.0
        wavelength_nm = 0.1
        pixel_size_um = 75.0
        peak_positions = self._generate_rotated_peak_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            orders=[-3, -2, -1, 1, 2, 3],
            angle_deg=33.0,
        )
        image = self._generate_peak_image((220, 220), peak_positions)

        sdd_fit_cm, uncertainty_cm, details = estimate_sample_detector_distance(
            image=image,
            peak_positions=peak_positions,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            beam_center_guess_px=(101.0, 90.0),
            beam_center_search_radius_px=20.0,
            sdd_search_range_cm=(150.0, 260.0),
            coarse_grid_points=21,
            fine_grid_points=21,
            return_details=True,
        )

        self.assertAlmostEqual(sdd_fit_cm, sdd_cm, delta=1.0)
        self.assertTrue(np.isfinite(uncertainty_cm))
        self.assertAlmostEqual(
            uncertainty_cm,
            details["standard_uncertainty_cm"],
            places=12,
        )
        self.assertAlmostEqual(details["grid_sdd_cm"], sdd_cm, delta=3.0)
        self.assertGreater(details["grid_uncertainty_cm"], 0)

    def test_estimate_sample_detector_distance_resolution_weighting(self):
        beam_center_px = np.array([103.4, 87.2])
        sdd_cm = 215.0
        pitch_nm = 80.0
        wavelength_nm = 0.1
        pixel_size_um = 75.0
        peak_positions = self._generate_rotated_peak_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            orders=[-3, -2, -1, 1, 2, 3],
            angle_deg=33.0,
        )
        image = self._generate_peak_image((220, 220), peak_positions)

        _, uncertainty_cm, details = estimate_sample_detector_distance(
            image=image,
            peak_positions=peak_positions,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            beam_center_guess_px=(101.0, 90.0),
            beam_center_search_radius_px=20.0,
            sdd_search_range_cm=(150.0, 260.0),
            coarse_grid_points=21,
            fine_grid_points=21,
            return_details=True,
        )

        self.assertTrue(np.isfinite(uncertainty_cm))
        self.assertAlmostEqual(
            uncertainty_cm,
            details["standard_uncertainty_cm"],
            places=12,
        )
        self.assertEqual(details["fitted_peak_positions_px"].shape, (6, 2))
        self.assertTrue(np.all(details["peak_position_uncertainty_px"] > 0))

    def test_estimate_sample_detector_distance_input_validation(self):
        with self.assertRaises(ValueError):
            estimate_sample_detector_distance(
                image=np.ones((10, 10)),
                peak_positions=np.array([1.0, 2.0]),
                pitch_nm=80.0,
                wavelength_nm=0.1,
                pixel_size_um=75.0,
            )

    def test_estimate_sample_detector_distance_from_rings(self):
        beam_center_px = np.array([103.4, 87.2])
        sdd_cm = 215.0
        pitch_nm = 80.0
        wavelength_nm = 0.1
        pixel_size_um = 75.0
        peak_positions = self._generate_ring_sample_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            ring_indices=[1, 2, 3],
            angles_deg=[15.0, 75.0, 135.0, 255.0],
        )
        image = self._generate_peak_image((220, 220), peak_positions)

        sdd_fit_cm, uncertainty_cm, details = (
            estimate_sample_detector_distance_from_rings(
                image=image,
                peak_positions=peak_positions,
                pitch_nm=pitch_nm,
                wavelength_nm=wavelength_nm,
                pixel_size_um=pixel_size_um,
                beam_center_guess_px=(100.0, 90.0),
                beam_center_search_radius_px=20.0,
                sdd_search_range_cm=(150.0, 260.0),
                coarse_grid_points=21,
                fine_grid_points=21,
                return_details=True,
            )
        )

        self.assertAlmostEqual(sdd_fit_cm, sdd_cm, delta=3.0)
        self.assertGreater(uncertainty_cm, 0)
        self.assertTrue(np.isfinite(uncertainty_cm))
        self.assertAlmostEqual(
            details["beam_center_px"][0],
            beam_center_px[0],
            delta=2.0,
        )
        self.assertAlmostEqual(
            details["beam_center_px"][1],
            beam_center_px[1],
            delta=2.0,
        )
        self.assertEqual(details["ring_indices"].shape, (12,))
        self.assertTrue(np.all(details["ring_indices"] >= 1))
        self.assertNotIn("orders", details)
        self.assertEqual(details["radial_distances_px"].shape, (12,))
        self.assertEqual(details["peak_position_uncertainty_px"].shape, (12,))
        self.assertTrue(np.all(details["peak_position_uncertainty_px"] > 0))
        self.assertAlmostEqual(
            uncertainty_cm,
            details["standard_uncertainty_cm"],
            places=12,
        )

    def test_estimate_sample_detector_distance_from_rings_repeated_ring(self):
        beam_center_px = np.array([103.4, 87.2])
        sdd_cm = 215.0
        pitch_nm = 80.0
        wavelength_nm = 0.1
        pixel_size_um = 75.0
        peak_positions = self._generate_ring_sample_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            ring_indices=[1, 1, 2, 2, 3, 3],
            angles_deg=[0.0, 90.0],
        )
        image = self._generate_peak_image((220, 220), peak_positions)

        _, uncertainty_cm, details = estimate_sample_detector_distance_from_rings(
            image=image,
            peak_positions=peak_positions,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            beam_center_guess_px=(100.0, 90.0),
            beam_center_search_radius_px=20.0,
            sdd_search_range_cm=(150.0, 260.0),
            coarse_grid_points=21,
            fine_grid_points=21,
            minimum_ring_uncertainty_px=0.25,
            return_details=True,
        )

        self.assertTrue(np.isfinite(uncertainty_cm))
        self.assertEqual(details["aggregated_ring_indices"].shape, (3,))
        self.assertTrue(
            np.all(details["aggregated_peak_position_uncertainty_px"] >= 0.25)
        )

    def test_score_sdd_ring_candidate_excludes_sparse_outliers(self):
        beam_center_px = np.array([103.4, 87.2])
        sdd_cm = 215.0
        pitch_nm = 80.0
        wavelength_nm = 0.1
        pixel_size_um = 75.0
        peak_positions = self._generate_ring_sample_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            ring_indices=[1, 2, 3],
            angles_deg=[15.0, 75.0, 135.0, 255.0],
        )
        q_spacing = 2 * np.pi / pitch_nm
        outer_q_1 = 4.0 * q_spacing
        outer_q_2 = 5.0 * q_spacing

        def q_to_position(q_value, angle_deg):
            theta = 2 * np.arcsin(q_value * wavelength_nm / (4 * np.pi))
            radial_distance_cm = sdd_cm * np.tan(theta)
            radial_distance_px = radial_distance_cm * 1e4 / pixel_size_um
            angle_rad = np.deg2rad(angle_deg)
            direction = np.array([np.sin(angle_rad), np.cos(angle_rad)])
            return beam_center_px + radial_distance_px * direction

        outlier_positions = np.asarray([
            q_to_position(outer_q_1, 40.0),
            q_to_position(outer_q_2, 210.0),
        ])
        peak_positions = np.vstack([peak_positions, outlier_positions])
        score, ring_indices, _, supported_mask = _score_sdd_ring_candidate(
            peak_positions=peak_positions,
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pixel_size_um=pixel_size_um,
            wavelength_nm=wavelength_nm,
            pitch_nm=pitch_nm,
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertEqual(ring_indices.shape, (14,))
        self.assertEqual(supported_mask.shape, (14,))
        self.assertEqual(np.count_nonzero(~supported_mask), 2)
        self.assertEqual(np.count_nonzero(ring_indices == 1), 4)
        self.assertEqual(np.count_nonzero(ring_indices == 2), 4)
        self.assertEqual(np.count_nonzero(ring_indices == 3), 4)
        self.assertEqual(np.count_nonzero(ring_indices == 4), 1)
        self.assertEqual(np.count_nonzero(ring_indices == 5), 1)

    def test_estimate_sample_detector_distance_from_rings_arc_segments(self):
        beam_center_px = np.array([110.0, 110.0])
        sdd_cm = 12.5
        pitch_nm = 80.0
        wavelength_nm = 0.154
        pixel_size_um = 75.0
        ring_indices = np.array([1, 2, 3])
        angles_deg = [-35.0, -10.0, 20.0]

        peak_positions = self._generate_ring_sample_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            ring_indices=ring_indices,
            angles_deg=angles_deg,
        )
        image = self._generate_ring_arc_image(
            shape=(240, 240),
            beam_center_px=beam_center_px,
            peak_positions=peak_positions,
        )

        estimated_sdd_cm, _, details = estimate_sample_detector_distance_from_rings(
            image=image,
            peak_positions=peak_positions,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            beam_center_guess_px=beam_center_px,
            beam_center_search_radius_px=0.0,
            return_details=True,
        )

        self.assertAlmostEqual(estimated_sdd_cm, sdd_cm, delta=0.3)
        self.assertTrue(np.all(np.isfinite(details["peak_position_uncertainty_px"])))
        self.assertTrue(np.all(details["peak_position_uncertainty_px"] > 0))
        self.assertTrue(np.all(details["radial_fit_success"]))

    def test_estimate_sample_detector_distance_from_ring_sectors(self):
        beam_center_px = np.array([110.0, 110.0])
        sdd_cm = 12.5
        pitch_nm = 80.0
        wavelength_nm = 0.154
        pixel_size_um = 75.0
        ring_indices = [1, 2, 3]
        angles_deg = np.arange(0.0, 360.0, 10.0)

        peak_positions = self._generate_ring_sample_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            ring_indices=ring_indices,
            angles_deg=angles_deg,
        )
        image = self._generate_ring_arc_image(
            shape=(240, 240),
            beam_center_px=beam_center_px,
            peak_positions=peak_positions,
            radial_sigma_px=1.0,
            tangential_sigma_px=8.0,
        )

        estimated_sdd_cm, uncertainty_cm, details = (
            estimate_sample_detector_distance_from_ring_sectors(
                image=image,
                beam_center_guess_px=(108.0, 112.0),
                sdd_guess_cm=13.0,
                pitch_nm=pitch_nm,
                wavelength_nm=wavelength_nm,
                pixel_size_um=pixel_size_um,
                ring_indices=ring_indices,
                sector_step_deg=10.0,
                sector_width_deg=10.0,
                radial_window_px=6.0,
                beam_center_search_radius_px=4.0,
                sdd_search_half_width_cm=2.0,
                coarse_grid_points=9,
                fine_grid_points=9,
                return_details=True,
            )
        )

        self.assertAlmostEqual(estimated_sdd_cm, sdd_cm, delta=0.6)
        self.assertTrue(np.isfinite(uncertainty_cm))
        self.assertAlmostEqual(
            details["beam_center_px"][0],
            beam_center_px[0],
            delta=2.0,
        )
        self.assertAlmostEqual(
            details["beam_center_px"][1],
            beam_center_px[1],
            delta=2.0,
        )
        self.assertGreater(details["fitted_peak_positions_px"].shape[0], 10)
        self.assertEqual(
            details["fitted_peak_positions_px"].shape[0],
            details["ring_indices"].shape[0],
        )
        self.assertEqual(
            details["fitted_peak_positions_px"].shape[0],
            details["sector_angles_deg"].shape[0],
        )
        self.assertTrue(np.all(details["peak_position_uncertainty_px"] > 0))
        self.assertTrue(np.all(np.isin(details["ring_indices"], ring_indices)))

    def test_estimate_sample_detector_distance_from_ring_sectors_validation(self):
        with self.assertRaises(ValueError):
            estimate_sample_detector_distance_from_ring_sectors(
                image=np.ones((10, 10)),
                beam_center_guess_px=(5.0,),
                sdd_guess_cm=10.0,
                pitch_nm=80.0,
                wavelength_nm=0.1,
                pixel_size_um=75.0,
            )

        with self.assertRaises(ValueError):
            estimate_sample_detector_distance_from_ring_sectors(
                image=np.ones((10, 10)),
                beam_center_guess_px=(5.0, 5.0),
                sdd_guess_cm=10.0,
                pitch_nm=80.0,
                wavelength_nm=0.1,
                pixel_size_um=75.0,
                sector_step_deg=0.0,
            )

    def test_estimate_sample_detector_distance_from_rings_excludes_beamstop_points(self):
        beam_center_px = np.array([103.4, 87.2])
        sdd_cm = 215.0
        pitch_nm = 80.0
        wavelength_nm = 0.1
        pixel_size_um = 75.0
        ring_positions = self._generate_ring_sample_positions(
            beam_center_px=beam_center_px,
            sdd_cm=sdd_cm,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            ring_indices=[1, 2, 3],
            angles_deg=[15.0, 75.0, 135.0, 255.0],
        )
        beamstop_points = np.array([
            beam_center_px + np.array([2.0, 1.0]),
            beam_center_px + np.array([-1.5, -2.0]),
        ])
        peak_positions = np.vstack([beamstop_points, ring_positions])
        image = self._generate_peak_image((220, 220), peak_positions)

        estimated_sdd_cm, _, details = estimate_sample_detector_distance_from_rings(
            image=image,
            peak_positions=peak_positions,
            pitch_nm=pitch_nm,
            wavelength_nm=wavelength_nm,
            pixel_size_um=pixel_size_um,
            beam_center_guess_px=beam_center_px,
            exclude_within_beamstop_radius_px=5.0,
            beam_center_search_radius_px=20.0,
            sdd_search_range_cm=(150.0, 260.0),
            coarse_grid_points=21,
            fine_grid_points=21,
            return_details=True,
        )

        self.assertAlmostEqual(estimated_sdd_cm, sdd_cm, delta=3.0)
        self.assertEqual(details["ring_indices"].shape, (12,))
        self.assertEqual(details["peak_position_uncertainty_px"].shape, (12,))

    def test_estimate_sample_detector_distance_from_rings_input_validation(self):
        with self.assertRaises(ValueError):
            estimate_sample_detector_distance_from_rings(
                image=np.ones((10, 10)),
                peak_positions=np.array([1.0, 2.0]),
                pitch_nm=80.0,
                wavelength_nm=0.1,
                pixel_size_um=75.0,
            )

        with self.assertRaises(ValueError):
            estimate_sample_detector_distance_from_rings(
                image=np.ones((10, 10)),
                peak_positions=np.array([[1.0, 2.0], [3.0, 4.0]]),
                pitch_nm=80.0,
                wavelength_nm=0.1,
                pixel_size_um=75.0,
                minimum_ring_uncertainty_px=0.0,
            )

        with self.assertRaises(ValueError):
            estimate_sample_detector_distance_from_rings(
                image=np.ones((10, 10)),
                peak_positions=np.array([[1.0, 2.0], [3.0, 4.0]]),
                pitch_nm=80.0,
                wavelength_nm=0.1,
                pixel_size_um=75.0,
                exclude_within_beamstop_radius_px=-1.0,
            )
