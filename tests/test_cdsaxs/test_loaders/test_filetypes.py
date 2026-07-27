import os
import unittest

import numpy as np

import cdsaxs.loaders.filetypes as filetypes
from cdsaxs._dtypes import REAL_DTYPE


class TestReadTiff(unittest.TestCase):

    def setUp(self):

        image_file = "../../data/test_loaders/simple.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(os.path.join(current_dir, image_file))

        self.image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, -9],
            [10, 11, 12]
        ]).astype(REAL_DTYPE)

        self.return_read = filetypes.read_tiff(self.filepath)

    def test_image(self):
        image = self.return_read[0]

        np.testing.assert_array_equal(image, self.image)
        self.assertEqual(image.dtype, REAL_DTYPE)

    def test_filepath(self):
        filepath = self.return_read[1]

        self.assertEqual(filepath, self.filepath)


class TestReadNistBin(unittest.TestCase):

    def setUp(self):
        image_file = "../../data/test_loaders/nist.bin"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.return_read = filetypes.read_nist_bin(self.filepath)

    def test_image(self):
        image = self.return_read[0]
        self.assertEqual(image[105, 910], 2581.0)
        self.assertEqual(image.dtype, np.float64)

    def test_filepath(self):
        filepath = self.return_read[1]
        self.assertEqual(filepath, self.filepath)

    def test_metadta(self):
        metadata = self.return_read[2]
        self.assertEqual(
            metadata["wavelength_nm"],
            0.13404
        )
        self.assertEqual(
            metadata["exposure_time_s"],
            600
        )
        self.assertEqual(
            metadata["pixel_size_um"],
            172
        )


class TestReadALS(unittest.TestCase):

    def setUp(self):
        image_file = "../../data/test_loaders/jk_srm_i10_300ev_align87448-00038.fits"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.return_read = filetypes.read_als_11_0_1_2(self.filepath)

    def test_image(self):
        # TODO: implement this test
        pass

    def test_metadata(self):
        metadata_return = self.return_read[-1]

        self.assertAlmostEqual(
            metadata_return["energy_ev"],
            300.011826472274)
        self.assertAlmostEqual(
            metadata_return["sample_phi_deg"],
            -90 + 90  # adjust for different coordinate systems
        )
        self.assertAlmostEqual(
            metadata_return["I0"],
            0.0579485378986026
        )
        self.assertAlmostEqual(
            metadata_return["exposure_time_s"],
            0.00999999977648258
        )
        self.assertAlmostEqual(
            metadata_return["beam_current"],
            500.108294433594
        )
        self.assertAlmostEqual(
            metadata_return["detector_phi_deg"],
            17
        )
        self.assertAlmostEqual(
            metadata_return["detector_y_mm"],
            98
        )
        self.assertAlmostEqual(
            metadata_return["CCD Y"],
            20
        )
        self.assertAlmostEqual(
            metadata_return["epu_polarization"],
            100
        )
        self.assertAlmostEqual(
            metadata_return["beam_stop_position"],
            8
        )
        self.assertAlmostEqual(
            metadata_return["pixel_size_um"],
            27
        )

    def test_pixel_size(self):
        # TODO: implement a more robust test for pixel size if/else
        pass