import os
import unittest

import numpy as np
import pytest

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
        self.real_scan_filepath = os.path.abspath(
            os.path.join(
                current_dir,
                '../../data/test_loaders/1129050_CW_H11_measure1_sdd_cm_0.00_energy_ev_16100_exposure_time_s_0.20_num_1_result.h5'
            )
        )
        self.multiple_scan_filepath = os.path.abspath(
            os.path.join(current_dir, '../../data/test_loaders/mock_smi_h5_multiple_scans.h5')
        )

    @pytest.mark.slow
    def test_read_smi_h5_real_fixture_extracts_scan_images_and_metadata(self):
        images = filetypes.read_smi_h5(self.real_scan_filepath)

        self.assertEqual(len(images), 121)
        image, filepath, metadata = images[0]

        self.assertEqual(filepath, self.real_scan_filepath)
        self.assertEqual(image.dtype, np.float64)
        self.assertEqual(image.shape, (1679, 1475))
        self.assertEqual(image[0, 0], 0.0)
        self.assertEqual(image[0, 1], 0.0)
        self.assertEqual(image[1, 0], 0.0)
        self.assertEqual(image[-1, -1], 0.0)
        self.assertAlmostEqual(metadata['energy_ev'], 16099.99500544827)
        self.assertAlmostEqual(metadata['sdd_cm'], 500.000026)
        self.assertEqual(metadata['exposure_time_s'], 0.2)
        self.assertEqual(metadata['pixel_size_um'], 172)
        self.assertAlmostEqual(metadata['bpm'], 1.8655923583401413)
        self.assertEqual(metadata['sample_phi_deg'], 58.7)

    def test_read_smi_h5_uses_one_based_seq_num_lookup(self):
        images = filetypes.read_smi_h5(self.multiple_scan_filepath)

        self.assertEqual(images[0][2]['bpm'], 3.0)
        self.assertEqual(images[1][2]['bpm'], 1.8)
        self.assertEqual(images[0][2]['sample_phi_deg'], -3.3)
        self.assertEqual(images[1][2]['sample_phi_deg'], -2.1)

    @pytest.mark.slow
    def test_read_smi_h5_real_fixture_maps_metadata_by_seq_num(self):
        images = filetypes.read_smi_h5(self.real_scan_filepath)

        self.assertEqual(len(images), 121)

        first_image, first_filepath, first_metadata = images[0]
        middle_image, middle_filepath, middle_metadata = images[60]
        last_image, last_filepath, last_metadata = images[-1]

        self.assertEqual(first_filepath, self.real_scan_filepath)
        self.assertEqual(middle_filepath, self.real_scan_filepath)
        self.assertEqual(last_filepath, self.real_scan_filepath)

        self.assertEqual(first_image.dtype, np.float64)
        self.assertEqual(first_image[0, 0], 0.0)
        self.assertEqual(first_image[-1, -1], 0.0)
        self.assertEqual(middle_image[0, 0], 0.0)
        self.assertEqual(middle_image[-1, -1], 0.0)
        self.assertEqual(last_image[0, 0], 0.0)
        self.assertEqual(last_image[-1, -1], 0.0)

        self.assertAlmostEqual(first_metadata['energy_ev'], 16099.99500544827)
        self.assertAlmostEqual(first_metadata['sdd_cm'], 500.000026)
        self.assertEqual(first_metadata['exposure_time_s'], 0.2)
        self.assertEqual(first_metadata['pixel_size_um'], 172)
        self.assertAlmostEqual(first_metadata['bpm'], 1.8655923583401413)
        self.assertEqual(first_metadata['sample_phi_deg'], 58.7)

        self.assertAlmostEqual(middle_metadata['bpm'], 1.8661785444530121)
        self.assertEqual(middle_metadata['sample_phi_deg'], -1.3)
        self.assertAlmostEqual(last_metadata['bpm'], 1.8599068901587745)
        self.assertEqual(last_metadata['sample_phi_deg'], -61.3)
