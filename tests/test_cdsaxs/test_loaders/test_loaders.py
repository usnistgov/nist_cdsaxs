import os
import unittest

import numpy as np

import cdsaxs.loaders.load_data as load_data
import cdsaxs.calculators as calculators
from cdsaxs.tools import bin_image


class TestFilterFilenames(unittest.TestCase):

    def setUp(self):
        directory_path = "../../data/test_filter_filenames"
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
        filenames = load_data.filter_filenames(
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
        filenames = load_data.filter_filenames(
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
        filenames = load_data.filter_filenames(
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
        filenames = load_data.filter_filenames(
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
        filenames = load_data.filter_filenames(
            self.directory_path,
            file_extension="txt"
            )

        self.assertCountEqual(
            filenames,
            self.filenames
        )


class TestLoadData(unittest.TestCase):

    def setUp(self):
        image_file = "../../data/test_loaders/smi.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.data = load_data.LoadData(filepath=self.filepath,
                                     detector_type='Pilatus',
                                     metadata={'center_px': (100, 100)},
                                     user_params={'test': 'testvalue'},
                                     name='Test Load Data')

    def test_load_pilatus(self):
        image, filepath, metadata = load_data.read_pilatus(
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
        data = load_data.LoadData(filepath=self.filepath)
        self.assertEqual(data.name, os.path.basename(self.filepath))

    def test_metadata_name(self):
        data = load_data.LoadData(filepath=self.filepath,
                                metadata={'name': 'Test Load Data Name'})
        self.assertEqual(data.name, "Test Load Data Name")

    def test_metadata_name_and_name(self):
        data = load_data.LoadData(filepath=self.filepath, name="new name",
                                metadata={'name': 'Test Load Data Name'})
        self.assertEqual(data.name, "new name")

    def test_crop_region(self):
        crop_region = ((100, None), (None, None))
        data = load_data.LoadData(
            filepath=self.filepath,
            detector_type='Pilatus',
            metadata={'center_px': (150, 25)},
            crop_region=crop_region,
        )
        image, _, _ = load_data.read_pilatus(self.filepath)
        image[image < 0] = np.nan

        np.testing.assert_array_almost_equal(
            data.image,
            image[100:, :],
        )
        self.assertTupleEqual(data.metadata['center_px'], (50, 25))

    def test_crop_region_invalid(self):
        with self.assertRaises(ValueError):
            load_data.LoadData(
                filepath=self.filepath,
                crop_region=(100, None),
            )

    def test_bin_size(self):
        data = load_data.LoadData(
            filepath=self.filepath,
            detector_type='Pilatus',
            bin_size=2,
        )
        image, _, _ = load_data.read_pilatus(self.filepath)
        image[image < 0] = np.nan
        expected_image, _ = bin_image(image, bin_size=2)

        np.testing.assert_array_almost_equal(data.image, expected_image)

    def test_bin_size_after_crop(self):
        crop_region = ((100, 300), (50, 250))
        data = load_data.LoadData(
            filepath=self.filepath,
            detector_type='Pilatus',
            metadata={'center_px': (150, 125), 'pixel_size_um': 172},
            crop_region=crop_region,
            bin_size=2,
        )
        image, _, _ = load_data.read_pilatus(self.filepath)
        image[image < 0] = np.nan
        expected_image, _ = bin_image(image[100:300, 50:250], bin_size=2)

        np.testing.assert_array_almost_equal(data.image, expected_image)
        self.assertTupleEqual(data.metadata['center_px'], (25.0, 37.5))
        self.assertEqual(data.metadata['pixel_size_um'], 344)

    def test_bin_size_invalid(self):
        with self.assertRaises(ValueError):
            load_data.LoadData(
                filepath=self.filepath,
                bin_size=0,
            )

        with self.assertRaises(ValueError):
            load_data.LoadData(
                filepath=self.filepath,
                bin_size=True,
            )

    def test_keep_raw_image_false(self):
        data = load_data.LoadData(
            filepath=self.filepath,
            keep_raw_image=False,
        )

        self.assertIsNone(data._raw_image)


class TestLoadDataset(unittest.TestCase):

    def setUp(self):
        data_directory = "../../data/test_load_dataset"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = os.path.abspath(os.path.join(current_dir, data_directory))
        self.dataset = load_data.LoadDataset(
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

    def test_crop_region(self):
        crop_region = ((100, None), (None, None))
        dataset = load_data.LoadDataset(
            "cropped dataset",
            self.data_directory,
            filenames=[
                "W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_"
                "id857176_combined.tif",
            ],
            metadata={'center_px': (200, 200)},
            crop_region=crop_region,
            filetype='tif',
            verbose=False,
        )
        data = next(iter(dataset.datas.values()))
        image, _, _ = load_data.read_tiff(
            os.path.join(
                self.data_directory,
                "W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_"
                "id857176_combined.tif",
            )
        )

        np.testing.assert_array_almost_equal(
            data.image,
            image[100:, :],
        )
        self.assertTupleEqual(data.metadata['center_px'], (100, 200))

    def test_bin_size_after_crop(self):
        dataset = load_data.LoadDataset(
            "binned dataset",
            self.data_directory,
            filenames=[
                "W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_"
                "id857176_combined.tif",
            ],
            metadata={'center_px': (200, 200), 'pixel_size_um': 100},
            crop_region=((100, 300), (50, 250)),
            bin_size=2,
            filetype='tif',
            verbose=False,
        )
        data = next(iter(dataset.datas.values()))
        image, _, _ = load_data.read_tiff(
            os.path.join(
                self.data_directory,
                "W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_"
                "id857176_combined.tif",
            )
        )
        expected_image, _ = bin_image(image[100:300, 50:250], bin_size=2)

        np.testing.assert_array_almost_equal(data.image, expected_image)
        self.assertTupleEqual(data.metadata['center_px'], (50.0, 75.0))
        self.assertEqual(data.metadata['pixel_size_um'], 200)

    def test_keep_raw_image_false(self):
        dataset = load_data.LoadDataset(
            "no raw dataset",
            self.data_directory,
            filenames=[
                "W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_"
                "id857176_combined.tif",
            ],
            filetype='tif',
            verbose=False,
            keep_raw_image=False,
        )

        data = next(iter(dataset.datas.values()))
        self.assertIsNone(data._raw_image)

    def test_reset_all_data_raises_without_raw_image(self):
        dataset = load_data.LoadDataset(
            "no raw dataset",
            self.data_directory,
            filenames=[
                "W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_"
                "id857176_combined.tif",
            ],
            filetype='tif',
            verbose=False,
            keep_raw_image=False,
        )

        with self.assertRaisesRegex(ValueError, "raw image was not retained"):
            dataset.reset_all_data()

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
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
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
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
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
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
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
        csv_path = "../../data/test_load_dataset/metadata_load.csv"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.abspath(os.path.join(current_dir, csv_path))
        self.dataset = load_data.LoadDataset_MetadataCSV(
            "test dataset", self.csv_path)

    def test_dataset_name(self):
        self.assertEqual(self.dataset.name, "test dataset")

    def test_keep_raw_image_false(self):
        dataset = load_data.LoadDataset_MetadataCSV(
            "test dataset",
            self.csv_path,
            keep_raw_image=False,
            verbose=False,
        )

        data = next(iter(dataset.datas.values()))
        self.assertIsNone(data._raw_image)

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
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
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
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
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
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
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
