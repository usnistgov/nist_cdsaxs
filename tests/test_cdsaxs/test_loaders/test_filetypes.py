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


class TestReadSmiH5(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.single_scan_filepath = os.path.abspath(
            os.path.join(current_dir, '../../data/test_loaders/mock_smi_h5_single_scan.h5')
        )
        self.multiple_scan_filepath = os.path.abspath(
            os.path.join(current_dir, '../../data/test_loaders/mock_smi_h5_multiple_scans.h5')
        )

    def test_read_smi_h5_single_scan_real_fixture(self):
        images = filetypes.read_smi_h5(self.single_scan_filepath)

        self.assertEqual(len(images), 1)
        image, filepath, metadata = images[0]

        self.assertEqual(filepath, self.single_scan_filepath)
        self.assertEqual(image.dtype, np.float64)
        self.assertEqual(image.shape, (128, 128))
        self.assertEqual(image[0, 0], 74.0)
        self.assertEqual(image[0, 1], 75.0)
        self.assertEqual(image[1, 0], 83.0)
        self.assertEqual(image[-1, -1], 86.0)
        self.assertEqual(metadata['energy_ev'], 1.7000000000000002)
        self.assertEqual(metadata['sdd_cm'], 0.16)
        self.assertEqual(metadata['exposure_time_s'], 1.5)
        self.assertEqual(metadata['pixel_size_um'], 172)
        self.assertEqual(metadata['bpm'], 1.8)
        self.assertEqual(metadata['sample_phi_deg'], -2.1)

    def test_read_smi_h5_uses_one_based_seq_num_lookup(self):
        images = filetypes.read_smi_h5(self.multiple_scan_filepath)

        self.assertEqual(images[0][2]['bpm'], 3.0)
        self.assertEqual(images[1][2]['bpm'], 1.8)
        self.assertEqual(images[0][2]['sample_phi_deg'], -3.3)
        self.assertEqual(images[1][2]['sample_phi_deg'], -2.1)

    def test_read_smi_h5_multiple_scans_real_fixture(self):
        images = filetypes.read_smi_h5(self.multiple_scan_filepath)

        self.assertEqual(len(images), 61)

        first_image, first_filepath, first_metadata = images[0]
        middle_image, middle_filepath, middle_metadata = images[30]
        last_image, last_filepath, last_metadata = images[-1]

        self.assertEqual(first_filepath, self.multiple_scan_filepath)
        self.assertEqual(middle_filepath, self.multiple_scan_filepath)
        self.assertEqual(last_filepath, self.multiple_scan_filepath)

        self.assertEqual(first_image.dtype, np.float64)
        self.assertEqual(first_image[0, 0], 74.0)
        self.assertEqual(first_image[-1, -1], 86.0)
        self.assertEqual(middle_image[0, 0], 90.0)
        self.assertEqual(middle_image[-1, -1], 85.0)
        self.assertEqual(last_image[0, 0], 89.0)
        self.assertEqual(last_image[-1, -1], 84.0)

        self.assertEqual(first_metadata['energy_ev'], 1.7000000000000002)
        self.assertEqual(first_metadata['sdd_cm'], 0.16)
        self.assertEqual(first_metadata['exposure_time_s'], 1.5)
        self.assertEqual(first_metadata['pixel_size_um'], 172)
        self.assertEqual(first_metadata['bpm'], 3.0)
        self.assertEqual(first_metadata['sample_phi_deg'], -3.3)

        self.assertEqual(middle_metadata['bpm'], 2.38)
        self.assertEqual(middle_metadata['sample_phi_deg'], -2.68)
        self.assertEqual(last_metadata['bpm'], 2.9800000000000004)
        self.assertEqual(last_metadata['sample_phi_deg'], -3.28)
