import os
import unittest

import numpy as np

import cdsaxs.loaders.detectors as detectors
import cdsaxs.loaders.filetypes as filetypes
import cdsaxs.loaders.load_data as load_data
from cdsaxs._dtypes import REAL_DTYPE


class TestPilatusHeader(unittest.TestCase):

    def setUp(self):

        image_file = "../data/test_loaders/smi.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, image_file)

        self.image, self.filepath, self.header = load_data.read_tiff(image_path)

    def test_exposure_time(self):

        exposure_time = 0.2  # seconds

        self.assertEqual(
            detectors.extract_exposure_time_pilatus(self.header),
            exposure_time)

    def test_pixel_size(self):

        pixel_size = 172  # mircrons

        self.assertEqual(
            detectors.extract_pixel_size_pilatus(self.header),
            pixel_size)

    def test_pilatus_header_to_metadata(self):

        metadata = detectors.pilatus_header_to_metadata(self.header)
        self.assertEqual(
            metadata["exposure_time_s"],
            0.2
        )
        self.assertEqual(
            metadata["pixel_size_um"],
            172
        )

        self.assertListEqual(["exposure_time_s", "pixel_size_um"],
                             list(metadata.keys()))


class TestReadPilatus(unittest.TestCase):

    def setUp(self):
        image_file = "../data/test_loaders/smi.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.image, self.filepath, self.metadata = detectors.read_pilatus(
            filepath=self.filepath
        )

        self.image_check, self.filepath_check, header = filetypes.read_tiff(
            self.filepath
        )
        self.metadata_check = detectors.pilatus_header_to_metadata(header)

    def test_image(self):
        np.testing.assert_array_almost_equal(self.image_check, self.image)
        self.assertEqual(self.image[735, 450], 12)
        self.assertEqual(self.image.dtype, REAL_DTYPE)

    def test_filepath(self):
        self.assertEqual(self.filepath_check, self.filepath)

    def test_metadata(self):
        self.assertEqual(
            self.metadata["exposure_time_s"],
            0.2
        )
        self.assertEqual(
            self.metadata["pixel_size_um"],
            172
        )
        self.assertDictEqual(self.metadata_check, self.metadata)
