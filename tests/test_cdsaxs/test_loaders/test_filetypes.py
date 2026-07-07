import os
import unittest
from unittest.mock import patch

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

    def test_read_smi_h5_returns_per_image_metadata(self):
        fake_scan = {
            'raw_images': {'pil2M_image': np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ], dtype=float)},
            'baseline': {
                'energy_energy': np.array([16100.0]),
                'pil2M_motor_z': np.array([2980.0]),
            },
            'config': {'pil2M_cam_acquire_time': np.array([2.5])},
            'primary': {
                'xbpm3_sumX': np.array([10.0, 20.0]),
                'stage_phi': np.array([-1.5, 2.5]),
                'seq_num': np.array([1, 2]),
            },
        }

        with patch('cdsaxs.loaders.filetypes.h5py.File', return_value=fake_scan):
            images = filetypes.read_smi_h5('C:/tmp/scan.h5')

        self.assertEqual(len(images), 2)
        np.testing.assert_array_equal(images[0][0], np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(images[0][1], os.path.abspath('C:/tmp/scan.h5'))
        self.assertEqual(images[0][2]['energy_ev'], 16100.0)
        self.assertEqual(images[0][2]['sdd_cm'], 298.0)
        self.assertEqual(images[0][2]['exposure_time_s'], 2.5)
        self.assertEqual(images[0][2]['pixel_size_um'], 172)
        self.assertEqual(images[0][2]['bpm'], 10.0)
        self.assertEqual(images[0][2]['sample_phi_deg'], 1.5)
        self.assertEqual(images[1][2]['bpm'], 20.0)
        self.assertEqual(images[1][2]['sample_phi_deg'], -2.5)
