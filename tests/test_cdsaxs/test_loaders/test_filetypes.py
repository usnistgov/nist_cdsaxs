import os
import unittest

import numpy as np

import cdsaxs.loaders.filetypes as filetypes


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
        ]).astype(np.float64)

        self.return_read = filetypes.read_tiff(self.filepath)

    def test_image(self):
        image = self.return_read[0]

        np.testing.assert_array_equal(image, self.image)
        self.assertEqual(image.dtype, np.float64)

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
