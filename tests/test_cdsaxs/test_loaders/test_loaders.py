import os
import io
import unittest
import warnings
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest
import numpy as np

import cdsaxs.loaders.load_data as load_data
import cdsaxs.calculators as calculators


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
        nist_file = "../../data/test_loaders/nist.bin"
        simple_file = "../../data/test_loaders/simple.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.nist_filepath = os.path.abspath(
            os.path.join(current_dir, nist_file))
        self.simple_filepath = os.path.abspath(
            os.path.join(current_dir, simple_file))
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

    def test_load_tiff_without_detector_type_uses_tiff_reader(self):
        data = load_data.LoadData(filepath=self.simple_filepath)

        expected_image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, -9],
            [10, 11, 12],
        ], dtype=np.float64)

        np.testing.assert_array_equal(data.image, expected_image)
        self.assertEqual(data.metadata['filename'], os.path.basename(self.simple_filepath))
        self.assertEqual(data.metadata['data_directory'], os.path.dirname(self.simple_filepath))

    def test_load_nist_bin_real_fixture_extracts_metadata_and_cleans_negative_pixels(self):
        data = load_data.LoadData(filepath=self.nist_filepath)

        self.assertEqual(data.metadata['filename'], os.path.basename(self.nist_filepath))
        self.assertEqual(data.metadata['data_directory'], os.path.dirname(self.nist_filepath))
        self.assertEqual(data.metadata['wavelength_nm'], 0.13404)
        self.assertEqual(data.metadata['exposure_time_s'], 600)
        self.assertEqual(data.metadata['pixel_size_um'], 172)
        self.assertFalse(np.any(data.image < 0))

    def test_load_nist_bin_alias_with_underscore_extracts_metadata(self):
        data = load_data.LoadData(filepath=self.nist_filepath, filetype='nist_bin')

        self.assertEqual(data.metadata['filename'], os.path.basename(self.nist_filepath))
        self.assertEqual(data.metadata['data_directory'], os.path.dirname(self.nist_filepath))
        self.assertEqual(data.metadata['wavelength_nm'], 0.13404)
        self.assertEqual(data.metadata['exposure_time_s'], 600)
        self.assertEqual(data.metadata['pixel_size_um'], 172)
        self.assertFalse(np.any(data.image < 0))

    def test_unknown_extension_without_filetype_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'Did not recognize the filtype extension'):
            load_data.LoadData(filepath='C:/tmp/example.unknown')

    def test_unsupported_explicit_filetype_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'Did not recognize the filetype bogus-type'):
            load_data.LoadData(filepath=self.filepath, filetype='bogus-type')

    def test_metadata_validation_failure_propagates_before_loading(self):
        with patch('cdsaxs.loaders.load_data.check_metadata', side_effect=ValueError('bad metadata')) as check_mock:
            with patch('cdsaxs.loaders.load_data.read_tiff') as read_tiff_mock:
                with self.assertRaisesRegex(ValueError, 'bad metadata'):
                    load_data.LoadData(
                        filepath=self.filepath,
                        filetype='tiff',
                        metadata={'invalid_key': 'value'},
                    )

        check_mock.assert_called_once()
        read_tiff_mock.assert_not_called()

    def test_pilatus_user_metadata_wins_with_warning_and_negative_pixels_become_nan(self):
        image = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=float)
        metadata_add = {'center_px': (9, 9), 'pixel_size_um': 172}

        with patch('cdsaxs.loaders.load_data.read_pilatus', return_value=(image, self.filepath, metadata_add)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                data = load_data.LoadData(
                    filepath=self.filepath,
                    filetype='tiff',
                    detector_type='Pilatus',
                    metadata={'center_px': (100, 100)},
                )

        self.assertTupleEqual(data.metadata['center_px'], (100, 100))
        self.assertEqual(data.metadata['pixel_size_um'], 172)
        self.assertTrue(np.isnan(data.image[0, 1]))
        self.assertTrue(any('Metadata for center_px was provided by the user' in str(w.message) for w in caught))

    def test_nist_bin_user_metadata_wins_with_warning_and_negative_pixels_become_nan(self):
        image = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=float)
        metadata_add = {'wavelength_nm': 0.1, 'pixel_size_um': 172}

        with patch('cdsaxs.loaders.load_data.read_nist_bin', return_value=(image, self.filepath, metadata_add)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                data = load_data.LoadData(
                    filepath=self.filepath,
                    filetype='nist-bin',
                    metadata={'wavelength_nm': 0.2},
                )

        self.assertEqual(data.metadata['wavelength_nm'], 0.2)
        self.assertEqual(data.metadata['pixel_size_um'], 172)
        self.assertTrue(np.isnan(data.image[0, 1]))
        self.assertTrue(any('Metadata for wavelength_nm was provided by the user' in str(w.message) for w in caught))

    def test_smi_h5_user_metadata_wins_and_negative_pixels_become_nan(self):
        image_stack = [
            (
                np.array([[1.0, -2.0], [3.0, 4.0]], dtype=float),
                self.filepath,
                {'sample_phi_deg': 5.0, 'pixel_size_um': 172},
            )
        ]

        with patch('cdsaxs.loaders.load_data.read_smi_h5', return_value=image_stack):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                datas = load_data.LoadData(
                    filepath='C:/tmp/scan.h5',
                    filetype='smi-h5',
                    metadata={'sample_phi_deg': 1.0},
                )

        self.assertEqual(len(datas), 1)
        self.assertEqual(datas[0].metadata['sample_phi_deg'], 1.0)
        self.assertEqual(datas[0].metadata['pixel_size_um'], 172)
        self.assertTrue(np.isnan(datas[0].image[0, 1]))
        self.assertTrue(any('Metadata for sample_phi_deg was provided by the user' in str(w.message) for w in caught))

    def test_smi_h5_returns_list_of_data2d_with_generated_names(self):
        image_stack = [
            (
                np.array([[1.0, -2.0], [3.0, 4.0]], dtype=float),
                'C:/tmp/scan.h5',
                {'sample_phi_deg': 1.0, 'bpm': 10.0, 'pixel_size_um': 172},
            ),
            (
                np.array([[5.0, 6.0], [7.0, -8.0]], dtype=float),
                'C:/tmp/scan.h5',
                {'sample_phi_deg': 2.0, 'bpm': 20.0, 'pixel_size_um': 172},
            ),
        ]

        with patch('cdsaxs.loaders.load_data.read_smi_h5', return_value=image_stack):
            datas = load_data.LoadData(
                filepath='C:/tmp/scan.h5',
                filetype='smi-h5',
                name='phi-{sample_phi_deg}-bpm-{bpm}',
            )

        self.assertEqual(len(datas), 2)
        self.assertEqual(datas[0].name, 'phi-1.0-bpm-10.0')
        self.assertEqual(datas[1].name, 'phi-2.0-bpm-20.0')
        self.assertEqual(datas[0].metadata['filename'], 'scan.h5')
        self.assertEqual(os.path.normpath(datas[0].metadata['data_directory']), os.path.normpath('C:/tmp'))
        self.assertEqual(datas[0].user_params['bpm'], 10.0)
        self.assertEqual(datas[1].user_params['bpm'], 20.0)
        self.assertTrue(np.isnan(datas[0].image[0, 1]))
        self.assertTrue(np.isnan(datas[1].image[1, 1]))

    def test_smi_h5_metadata_merge_preserves_user_values(self):
        image_stack = [
            (
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                'C:/tmp/scan.h5',
                {'sample_phi_deg': 5.0, 'bpm': 10.0, 'pixel_size_um': 172},
            )
        ]

        with patch('cdsaxs.loaders.load_data.read_smi_h5', return_value=image_stack):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                datas = load_data.LoadData(
                    filepath='C:/tmp/scan.h5',
                    filetype='smi-h5',
                    metadata={'sample_phi_deg': 9.0},
                    user_params={'scan_id': 'abc'},
                )

        self.assertEqual(len(datas), 1)
        self.assertEqual(datas[0].metadata['sample_phi_deg'], 9.0)
        self.assertEqual(datas[0].user_params['scan_id'], 'abc')
        self.assertEqual(datas[0].user_params['bpm'], 10.0)
        self.assertTrue(any('Metadata for sample_phi_deg was provided by the user' in str(w.message) for w in caught))

    @pytest.mark.slow
    def test_load_smi_h5_real_fixture_extracts_metadata_and_user_params(self):
        h5_filepath = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '../../data/test_loaders/1129050_CW_H11_measure1_sdd_cm_0.00_energy_ev_16100_exposure_time_s_0.20_num_1_result.h5',
            )
        )

        datas = load_data.LoadData(filepath=h5_filepath, filetype='smi-h5')

        self.assertEqual(len(datas), 121)
        data = datas[0]
        self.assertEqual(data.image.dtype, np.float64)
        self.assertEqual(data.image.shape, (1679, 1475))
        self.assertEqual(data.image[0, 0], 0.0)
        self.assertEqual(data.image[0, 1], 0.0)
        self.assertEqual(data.image[1, 0], 0.0)
        self.assertEqual(data.image[-1, -1], 0.0)
        self.assertEqual(data.metadata['filename'], os.path.basename(h5_filepath))
        self.assertEqual(data.metadata['data_directory'], os.path.dirname(h5_filepath))
        self.assertAlmostEqual(data.metadata['energy_ev'], 16099.99500544827)
        self.assertAlmostEqual(data.metadata['sdd_cm'], 500.000026)
        self.assertEqual(data.metadata['exposure_time_s'], 0.2)
        self.assertEqual(data.metadata['pixel_size_um'], 172)
        self.assertEqual(data.metadata['sample_phi_deg'], 58.7)
        self.assertAlmostEqual(data.user_params['bpm'], 1.8655923583401413)

    @pytest.mark.slow
    def test_load_smi_h5_real_fixture_generates_names_from_metadata_and_user_params(self):
        h5_filepath = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '../../data/test_loaders/1129050_CW_H11_measure1_sdd_cm_0.00_energy_ev_16100_exposure_time_s_0.20_num_1_result.h5',
            )
        )

        datas = load_data.LoadData(
            filepath=h5_filepath,
            filetype='smi-h5',
            name='phi-{sample_phi_deg}',
        )

        self.assertEqual(len(datas), 121)

        first_data = datas[0]
        middle_data = datas[60]
        last_data = datas[-1]

        self.assertEqual(first_data.name, 'phi-58.7')
        self.assertEqual(middle_data.name, 'phi--1.3')
        self.assertEqual(last_data.name, 'phi--61.3')

        self.assertEqual(first_data.metadata['filename'], os.path.basename(h5_filepath))
        self.assertEqual(first_data.metadata['data_directory'], os.path.dirname(h5_filepath))
        self.assertEqual(first_data.image[0, 0], 0.0)
        self.assertEqual(first_data.image[-1, -1], 0.0)
        self.assertEqual(middle_data.image[0, 0], 0.0)
        self.assertEqual(middle_data.image[-1, -1], 0.0)
        self.assertEqual(last_data.image[0, 0], 0.0)
        self.assertEqual(last_data.image[-1, -1], 0.0)

        self.assertAlmostEqual(first_data.user_params['bpm'], 1.8655923583401413)
        self.assertEqual(first_data.metadata['sample_phi_deg'], 58.7)
        self.assertAlmostEqual(middle_data.user_params['bpm'], 1.8661785444530121)
        self.assertEqual(middle_data.metadata['sample_phi_deg'], -1.3)
        self.assertAlmostEqual(last_data.user_params['bpm'], 1.8599068901587745)
        self.assertEqual(last_data.metadata['sample_phi_deg'], -61.3)

    def test_load_smi_h5_without_filetype_exposes_h5_autodetect_bug(self):
        h5_filepath = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '../../data/test_loaders/1129050_CW_H11_measure1_sdd_cm_0.00_energy_ev_16100_exposure_time_s_0.20_num_1_result.h5',
            )
        )

        with self.assertRaises(AttributeError):
            load_data.LoadData(filepath=h5_filepath)


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

    def test_single_filename_string_is_loaded(self):
        filename = 'W204_F2measure1_5.2m_16.1keV_num60_00deg_bpm0.417_id857181_combined.tif'

        dataset = load_data.LoadDataset(
            'single file dataset',
            self.data_directory,
            filenames=filename,
            metadata_pattern=(
                '{sample}measure1_{sdd_cm}m_{energy_ev}keV'
                '_num{num}_{sample_phi_deg}deg_bpm{bpm}_id{id}_combined.tif'
            ),
            metadata_scales={'sdd_cm': 100, 'energy_ev': 1000},
            verbose=False,
            filetype='tif',
        )

        self.assertEqual(list(dataset.datas.keys()), [filename])
        self.assertEqual(dataset.datas[filename].metadata['sample_phi_deg'], 0.0)

    def test_filetype_filters_out_non_matching_filenames(self):
        with patch('cdsaxs.loaders.load_data.os.listdir', return_value=['keep.tif', 'skip.csv', 'also_skip.txt']):
            with patch('cdsaxs.loaders.load_data.LoadData') as load_data_mock:
                load_data_mock.return_value = load_data.Data2D(
                    image=np.ones((2, 2), dtype=float),
                    filename='keep.tif',
                    data_directory=self.data_directory,
                )

                dataset = load_data.LoadDataset(
                    'filtered dataset',
                    self.data_directory,
                    verbose=False,
                    filetype='tif',
                )

        load_data_mock.assert_called_once()
        self.assertEqual(load_data_mock.call_args.kwargs['filepath'], os.path.join(self.data_directory, 'keep.tif'))
        self.assertEqual(list(dataset.datas.keys()), ['keep.tif'])

    def test_metadata_pattern_conflicts_preserve_user_values_with_warnings(self):
        filename = 'W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_id857176_combined.tif'

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            dataset = load_data.LoadDataset(
                'conflict dataset',
                self.data_directory,
                filenames=[filename],
                metadata_pattern=(
                    '{sample}measure1_{sdd_cm}m_{energy_ev}keV'
                    '_num{num}_{sample_phi_deg}deg_bpm{bpm}_id{id}_combined.tif'
                ),
                metadata_scales={'sdd_cm': 100, 'energy_ev': 1000},
                metadata={'sample_phi_deg': 99.0},
                user_params={'sample': 'manual-sample'},
                verbose=False,
                filetype='tif',
            )

        data = dataset.datas[filename]
        self.assertEqual(data.metadata['sample_phi_deg'], 99.0)
        self.assertEqual(data.user_params['sample'], 'manual-sample')
        self.assertIn('Metadata for sample_phi_deg was provided by the user', stdout.getvalue())
        self.assertIn('User params for sample was provided by the user', stdout.getvalue())

    def test_generated_name_uses_pattern_values(self):
        filename = 'W204_F2measure1_5.2m_16.1keV_num65_05deg_bpm0.413_id857186_combined.tif'

        dataset = load_data.LoadDataset(
            'named dataset',
            self.data_directory,
            filenames=[filename],
            metadata_pattern=(
                '{sample}measure1_{sdd_cm}m_{energy_ev}keV'
                '_num{num}_{sample_phi_deg}deg_bpm{bpm}_id{id}_combined.tif'
            ),
            metadata_scales={'sdd_cm': 100, 'energy_ev': 1000},
            data_name_pattern='sample-{sample}-phi-{sample_phi_deg}-run-{num}',
            verbose=False,
            filetype='tif',
        )

        self.assertEqual(list(dataset.datas.keys()), ['sample-W204_F2-phi-5.0-run-65'])

    def test_metadata_pattern_mismatch_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            load_data.LoadDataset(
                'bad pattern dataset',
                self.data_directory,
                filenames=['W204_F2measure1_5.2m_16.1keV_num55_-05deg_bpm0.415_id857176_combined.tif'],
                metadata_pattern='does_not_match_{sample_phi_deg}.tif',
                verbose=False,
                filetype='tif',
            )

    def test_unsupported_filetype_propagates_value_error(self):
        with self.assertRaisesRegex(ValueError, 'Did not recognize the filetype bogus-type'):
            load_data.LoadDataset(
                'unsupported filetype dataset',
                self.data_directory,
                verbose=False,
                filetype='bogus-type',
            )

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
