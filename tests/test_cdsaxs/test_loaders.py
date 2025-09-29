import os
import unittest

import numpy as np

import cdsaxs.loaders as loaders
import cdsaxs._loader_tools as lt
import cdsaxs.calculators as calculators


class TestReadTiff(unittest.TestCase):

    def setUp(self):

        image_file = "../data/test_loaders/simple.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(os.path.join(current_dir, image_file))

        self.image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, -9],
            [10, 11, 12]
        ]).astype(np.float64)

        self.return_read = loaders.read_tiff(self.filepath)

    def test_image(self):
        image = self.return_read[0]

        np.testing.assert_array_equal(image, self.image)
        self.assertEqual(image.dtype, np.float64)

    def test_filepath(self):
        filepath = self.return_read[1]

        self.assertEqual(filepath, self.filepath)


class TestReadNistBin(unittest.TestCase):

    def setUp(self):
        image_file = "../data/test_loaders/nist.bin"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.return_read = loaders.read_nist_bin(self.filepath)

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


class TestReadPilatus(unittest.TestCase):

    def setUp(self):
        image_file = "../data/test_loaders/smi.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.image, self.filepath, self.metadata = loaders.read_pilatus(
            filepath=self.filepath
        )

        self.image_check, self.filepath_check, header = loaders.read_tiff(
            self.filepath
        )
        self.metadata_check = lt.pilatus_header_to_metadata(header)

    def test_image(self):
        np.testing.assert_array_almost_equal(self.image_check, self.image)
        self.assertEqual(self.image[735, 450], 12)
        self.assertEqual(self.image.dtype, np.float64)

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


class TestFilterFilenames(unittest.TestCase):

    def setUp(self):
        directory_path = "../data/test_filter_filenames"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.directory_path = os.path.abspath(
            os.path.join(current_dir, directory_path))

        self.filenames = [
            "green_apple_bad.txt",
            "green_apple_good.txt",
            "red_apple_bad_strawberry.txt",
            "red_apple_bad.txt",
            "red_apple_good.txt",
            "red_apple_strawberry.txt"
        ]

    def test_or(self):
        filenames = loaders.filter_filenames(
            self.directory_path,
            filter_substrings=["strawberry", "good"]
            )

        self.assertCountEqual(
            filenames,
            [
                "red_apple_bad_strawberry.txt",
                "red_apple_strawberry.txt",
                "green_apple_good.txt",
                "red_apple_good.txt",
            ]
        )

    def test_and(self):
        filenames = loaders.filter_filenames(
            self.directory_path,
            filter_substrings=[["strawberry", "bad"]]
            )

        self.assertCountEqual(
            filenames,
            [
                "red_apple_bad_strawberry.txt",
            ]
        )

    def test_not(self):
        filenames = loaders.filter_filenames(
            self.directory_path,
            filter_substrings=[["strawberry", ("NOT", "bad")]]
            )

        self.assertCountEqual(
            filenames,
            [
                "red_apple_strawberry.txt"
            ]
        )

    def test_complex(self):
        filenames = loaders.filter_filenames(
            self.directory_path,
            filter_substrings=[["strawberry", ("NOT", "bad")],
                               ["red", "apple"]]
            )

        self.assertCountEqual(
            filenames,
            [
                "red_apple_bad_strawberry.txt",
                "red_apple_bad.txt",
                "red_apple_good.txt",
                "red_apple_strawberry.txt"
            ]
        )

    def test_file_extension(self):
        filenames = loaders.filter_filenames(
            self.directory_path,
            file_extension="txt"
            )

        self.assertCountEqual(
            filenames,
            self.filenames
        )


class TestLoadData(unittest.TestCase):

    def setUp(self):
        image_file = "../data/test_loaders/smi.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.data = loaders.LoadData(filepath=self.filepath,
                                     detector_type='Pilatus',
                                     metadata={'center_px': (100, 100)},
                                     user_params={'test': 'testvalue'},
                                     name='Test Load Data')

    def test_load_pilatus(self):
        image, filepath, metadata = loaders.read_pilatus(
            self.filepath
        )
        image[image < 0] = np.nan

        np.testing.assert_array_almost_equal(self.data.image, image)
        self.assertEqual(self.data.metadata['filename'],
                         os.path.basename(filepath))
        self.assertEqual(self.data.metadata['data_directory'],
                         os.path.dirname(filepath))
        for key in metadata.keys():
            self.assertEqual(self.data.metadata[key], metadata[key])

    def test_center_px(self):
        self.assertTupleEqual(self.data.metadata['center_px'], (100, 100))

    def test_user_param(self):
        self.assertEqual(self.data.user_params['test'], 'testvalue')

    def test_name(self):
        self.assertEqual(self.data.name, 'Test Load Data')

    def test_default_name(self):
        data = loaders.LoadData(filepath=self.filepath)
        self.assertEqual(data.name, os.path.basename(self.filepath))

    def test_metadata_name(self):
        data = loaders.LoadData(filepath=self.filepath,
                                metadata={'name': 'Test Load Data Name'})
        self.assertEqual(data.name, "Test Load Data Name")

    def test_metadata_name_and_name(self):
        data = loaders.LoadData(filepath=self.filepath, name="new name",
                                metadata={'name': 'Test Load Data Name'})
        self.assertEqual(data.name, "new name")


class TestLoadDataset(unittest.TestCase):

    def setUp(self):
        data_directory = "../data/test_load_dataset"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = os.path.abspath(os.path.join(current_dir, data_directory))
        self.dataset = loaders.LoadDataset(
            "test dataset", self.data_directory,
            metadata_pattern="{sample}measure1_{sdd_cm}m_{energy_ev}keV"
            + "_num{num}_{sample_phi_deg}deg_bpm{bpm}_id{id}_combined.tif",
            metadata_scales={'sdd_cm': 100, 'energy_ev': 1000},
            metadata={'center_px': (200, 200)},
            user_params={'test user': 'test param'},
            data_name_pattern="{sample} {sample_phi_deg}",
            filetype='tif')

    def test_dataset_name(self):
        self.assertEqual(self.dataset.name, "test dataset")

    def test_datas_keys(self):
        keys = [
            "W204_F2 -5.0",
            "W204_F2 0.0",
            "W204_F2 5.0",
        ]
        self.assertListEqual(
            keys,
            list(self.dataset.datas.keys())
        )

    def test_metadata_dicts(self):
        metadata = {
            'energy_ev': 16100,
            'sample_phi_deg': -5.0,
            'sdd_cm': 520,
            'data_directory': self.data_directory,
            'filename': "W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_id857176_combined.tif",
            'center_px': (200, 200),
            'wavelength_nm': calculators.energy_to_wavelength(16100),
            'sample_phi_offset_deg': 0,
        }
        for key, value in self.dataset.datas["W204_F2 -5.0"].metadata.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    metadata[key],
                    value)
            else:
                self.assertEqual(
                    metadata[key],
                    value)

        metadata = {
            'energy_ev': 16100,
            'sample_phi_deg': 0.0,
            'sdd_cm': 520,
            'data_directory': self.data_directory,
            'filename': "W204_F2measure1_5.2m_16.1keV_num60_00deg_bpm0.417_id857181_combined.tif",
            'center_px': (200, 200),
            'wavelength_nm': calculators.energy_to_wavelength(16100),
            'sample_phi_offset_deg': 0,
        }
        for key, value in self.dataset.datas["W204_F2 0.0"].metadata.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    metadata[key],
                    value)
            else:
                self.assertEqual(
                    metadata[key],
                    value)

        metadata = {
            'energy_ev': 16100,
            'sample_phi_deg': 5.0,
            'sdd_cm': 520,
            'data_directory': self.data_directory,
            'filename': "W204_F2measure1_5.2m_16.1keV_num65_05deg_bpm0.413_id857186_combined.tif",
            'center_px': (200, 200),
            'wavelength_nm': calculators.energy_to_wavelength(16100),
            'sample_phi_offset_deg': 0,
        }
        for key, value in self.dataset.datas["W204_F2 5.0"].metadata.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    metadata[key],
                    value)
            else:
                self.assertEqual(
                    metadata[key],
                    value)

    def test_user_params_dicts(self):
        user_params = {
            'sample': 'W204_F2',
            'num': '55',
            'bpm': '0.415',
            'id': '857176',
            'test user': 'test param'
        }
        for key, value in self.dataset.datas["W204_F2 -5.0"].user_params.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    user_params[key],
                    value)
            else:
                self.assertEqual(
                    user_params[key],
                    value)

        user_params = {
            'sample': 'W204_F2',
            'num': '60',
            'bpm': '0.417',
            'id': '857181',
            'test user': 'test param'
        }
        for key, value in self.dataset.datas["W204_F2 0.0"].user_params.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    user_params[key],
                    value)
            else:
                self.assertEqual(
                    user_params[key],
                    value)

        user_params = {
            'sample': 'W204_F2',
            'num': '65',
            'bpm': '0.413',
            'id': '857186',
            'test user': 'test param'
        }
        for key, value in self.dataset.datas["W204_F2 5.0"].user_params.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    user_params[key],
                    value)
            else:
                self.assertEqual(
                    user_params[key],
                    value)


class TestLoadDataset_MetadataCSV(unittest.TestCase):

    def setUp(self):
        csv_path = "../data/test_load_dataset/metadata_load.csv"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.abspath(os.path.join(current_dir, csv_path))
        self.dataset = loaders.LoadDataset_MetadataCSV(
            "test dataset", self.csv_path)

    def test_dataset_name(self):
        self.assertEqual(self.dataset.name, "test dataset")

    def test_datas_keys(self):
        keys = [
            "W204_F2 -5.0",
            "W204_F2 0.0",
            "W204_F2 5.0",
        ]
        self.assertListEqual(
            keys,
            list(self.dataset.datas.keys())
        )

    def test_metadata_dicts(self):
        metadata = {
            'energy_ev': 16100,
            'sample_phi_deg': -5.0,
            'sdd_cm': 520,
            'data_directory': os.path.dirname(self.csv_path),
            'filename': "W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_id857176_combined.tif",
            'center_px': (0, 0),
            'wavelength_nm': calculators.energy_to_wavelength(16100),
            'sample_phi_offset_deg': 0,
        }
        for key, value in self.dataset.datas["W204_F2 -5.0"].metadata.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    metadata[key],
                    value)
            else:
                self.assertEqual(
                    metadata[key],
                    value)

        metadata = {
            'energy_ev': 16100,
            'sample_phi_deg': 0.0,
            'sdd_cm': 520,
            'data_directory': os.path.dirname(self.csv_path),
            'filename': "W204_F2measure1_5.2m_16.1keV_num60_00deg_bpm0.417_id857181_combined.tif",
            'center_px': (0, 0),
            'wavelength_nm': calculators.energy_to_wavelength(16100),
            'sample_phi_offset_deg': 0,
        }
        for key, value in self.dataset.datas["W204_F2 0.0"].metadata.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    metadata[key],
                    value)
            else:
                self.assertEqual(
                    metadata[key],
                    value)

        metadata = {
            'energy_ev': 16100,
            'sample_phi_deg': 5.0,
            'sdd_cm': 520,
            'data_directory': os.path.dirname(self.csv_path),
            'filename': "W204_F2measure1_5.2m_16.1keV_num65_05deg_bpm0.413_id857186_combined.tif",
            'center_px': (0, 0),
            'wavelength_nm': calculators.energy_to_wavelength(16100),
            'sample_phi_offset_deg': 0,
        }
        for key, value in self.dataset.datas["W204_F2 5.0"].metadata.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    metadata[key],
                    value)
            else:
                self.assertEqual(
                    metadata[key],
                    value)

    def test_user_params_dicts(self):
        user_params = {
            'bpm': '0.415',
        }
        for key, value in self.dataset.datas["W204_F2 -5.0"].user_params.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    user_params[key],
                    value)
            else:
                self.assertEqual(
                    user_params[key],
                    value)

        user_params = {
            'bpm': '0.417',
        }
        for key, value in self.dataset.datas["W204_F2 0.0"].user_params.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    user_params[key],
                    value)
            else:
                self.assertEqual(
                    user_params[key],
                    value)

        user_params = {
            'bpm': '0.413',
        }
        for key, value in self.dataset.datas["W204_F2 5.0"].user_params.items():
            if type(value) is not str:
                self.assertAlmostEqual(
                    user_params[key],
                    value)
            else:
                self.assertEqual(
                    user_params[key],
                    value)
