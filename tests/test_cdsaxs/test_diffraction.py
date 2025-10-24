import unittest

import cdsaxs.diffraction as diffraction


class TestDetectorPhiCorr(unittest.TestCase):

    def test_phi_corr(self):

        phi = 16.27323
        phi0 = 2.502
        scale = 1.05

        expected_phi_corr_rad = 0.2523709708269
        actual_phi_corr_rad = diffraction._detector_phi_corr(
            detector_phi_deg=phi,
            detector_phi0_deg=phi0,
            detector_phiscale=scale
        )

        self.assertAlmostEqual(expected_phi_corr_rad, actual_phi_corr_rad)

    def test_defaults(self):

        phi = 16.27323
        expected_phi_corr_rad = 0.2840214434343
        actual_phi_corr_rad = diffraction._detector_phi_corr(
            detector_phi_deg=phi,
        )

        self.assertAlmostEqual(expected_phi_corr_rad, actual_phi_corr_rad)


class TestCenterPxBeamToDetector(unittest.TestCase):

    def test_center(self):
        center_px_beam = (6.5, 2)
        pixel_size_um = 100
        sdd_cm = 500
        detector_phi_deg = 0.95238095238
        detector_y_mm = 0.3
        detector_phi0_deg = 0.76466980564
        detector_y0_mm = 0.1
        detector_phiscale = 1.05

        expected_center = (8.5, 174.0)
        actual_center = diffraction.center_px_beam_to_detector(
            center_px_beam=center_px_beam,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_phi0_deg=detector_phi0_deg,
            detector_phiscale=detector_phiscale,
            detector_y_mm=detector_y_mm,
            detector_y0_mm=detector_y0_mm,
        )

        self.assertAlmostEqual(
            actual_center[0],
            expected_center[0]
        )

        self.assertAlmostEqual(
            actual_center[1],
            expected_center[1]
        )

    def test_center_neg_phi(self):
        center_px_beam = (6.5, 2)
        pixel_size_um = 100
        sdd_cm = 500
        detector_phi_deg = -0.95238095238
        detector_y_mm = 0.3
        detector_phi0_deg = -0.76466980564
        detector_y0_mm = 0.1
        detector_phiscale = 1.05

        expected_center = (8.5, -170.0)
        actual_center = diffraction.center_px_beam_to_detector(
            center_px_beam=center_px_beam,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_phi0_deg=detector_phi0_deg,
            detector_phiscale=detector_phiscale,
            detector_y_mm=detector_y_mm,
            detector_y0_mm=detector_y0_mm,
        )

        self.assertAlmostEqual(
            actual_center[0],
            expected_center[0]
        )

        self.assertAlmostEqual(
            actual_center[1],
            expected_center[1]
        )

    def test_center_neg_y(self):
        center_px_beam = (6.5, 2)
        pixel_size_um = 100
        sdd_cm = 500
        detector_phi_deg = 0.95238095238
        detector_y_mm = -0.3
        detector_phi0_deg = 0.76466980564
        detector_y0_mm = -0.1
        detector_phiscale = 1.05

        expected_center = (4.5, 174.0)
        actual_center = diffraction.center_px_beam_to_detector(
            center_px_beam=center_px_beam,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_phi0_deg=detector_phi0_deg,
            detector_phiscale=detector_phiscale,
            detector_y_mm=detector_y_mm,
            detector_y0_mm=detector_y0_mm,
        )

        self.assertAlmostEqual(
            actual_center[0],
            expected_center[0]
        )

        self.assertAlmostEqual(
            actual_center[1],
            expected_center[1]
        )

    def test_center_neg_phi_neg_y(self):
        center_px_beam = (6.5, 2)
        pixel_size_um = 100
        sdd_cm = 500
        detector_phi_deg = -0.95238095238
        detector_y_mm = -0.3
        detector_phi0_deg = -0.76466980564
        detector_y0_mm = -0.1
        detector_phiscale = 1.05

        expected_center = (4.5, -170.0)
        actual_center = diffraction.center_px_beam_to_detector(
            center_px_beam=center_px_beam,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_phi0_deg=detector_phi0_deg,
            detector_phiscale=detector_phiscale,
            detector_y_mm=detector_y_mm,
            detector_y0_mm=detector_y0_mm,
        )

        self.assertAlmostEqual(
            actual_center[0],
            expected_center[0]
        )

        self.assertAlmostEqual(
            actual_center[1],
            expected_center[1]
        )


class TestCenterPxDetectorToBeam(unittest.TestCase):

    def test_center(self):
        center_px_detector = (8.5, 174.0)
        pixel_size_um = 100
        sdd_cm = 500
        detector_phi_deg = 0.95238095238
        detector_y_mm = 0.3
        detector_phi0_deg = 0.76466980564
        detector_y0_mm = 0.1
        detector_phiscale = 1.05

        expected_center = (6.5, 2)
        actual_center = diffraction.center_px_detector_to_beam(
            center_px_detector=center_px_detector,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_phi0_deg=detector_phi0_deg,
            detector_phiscale=detector_phiscale,
            detector_y_mm=detector_y_mm,
            detector_y0_mm=detector_y0_mm,
        )

        self.assertAlmostEqual(
            actual_center[0],
            expected_center[0]
        )

        self.assertAlmostEqual(
            actual_center[1],
            expected_center[1]
        )

    def test_center_neg_phi(self):
        center_px_detector = (8.5, -170.0)
        pixel_size_um = 100
        sdd_cm = 500
        detector_phi_deg = -0.95238095238
        detector_y_mm = 0.3
        detector_phi0_deg = -0.76466980564
        detector_y0_mm = 0.1
        detector_phiscale = 1.05

        expected_center = (6.5, 2)
        actual_center = diffraction.center_px_detector_to_beam(
            center_px_detector=center_px_detector,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_phi0_deg=detector_phi0_deg,
            detector_phiscale=detector_phiscale,
            detector_y_mm=detector_y_mm,
            detector_y0_mm=detector_y0_mm,
        )

        self.assertAlmostEqual(
            actual_center[0],
            expected_center[0]
        )

        self.assertAlmostEqual(
            actual_center[1],
            expected_center[1]
        )

    def test_center_neg_y(self):
        center_px_detector = (4.5, 174.0)
        pixel_size_um = 100
        sdd_cm = 500
        detector_phi_deg = 0.95238095238
        detector_y_mm = -0.3
        detector_phi0_deg = 0.76466980564
        detector_y0_mm = -0.1
        detector_phiscale = 1.05

        expected_center = (6.5, 2)
        actual_center = diffraction.center_px_detector_to_beam(
            center_px_detector=center_px_detector,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_phi0_deg=detector_phi0_deg,
            detector_phiscale=detector_phiscale,
            detector_y_mm=detector_y_mm,
            detector_y0_mm=detector_y0_mm,
        )

        self.assertAlmostEqual(
            actual_center[0],
            expected_center[0]
        )

        self.assertAlmostEqual(
            actual_center[1],
            expected_center[1]
        )

    def test_center_neg_phi_neg_y(self):
        center_px_detector = (4.5, -170.0)
        pixel_size_um = 100
        sdd_cm = 500
        detector_phi_deg = -0.95238095238
        detector_y_mm = -0.3
        detector_phi0_deg = -0.76466980564
        detector_y0_mm = -0.1
        detector_phiscale = 1.05

        expected_center = (6.5, 2)
        actual_center = diffraction.center_px_detector_to_beam(
            center_px_detector=center_px_detector,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_phi0_deg=detector_phi0_deg,
            detector_phiscale=detector_phiscale,
            detector_y_mm=detector_y_mm,
            detector_y0_mm=detector_y0_mm,
        )

        self.assertAlmostEqual(
            actual_center[0],
            expected_center[0]
        )

        self.assertAlmostEqual(
            actual_center[1],
            expected_center[1]
        )
