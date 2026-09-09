import itertools
import os
import warnings

import unittest

import cdsaxs.loaders._loader_tools as loader_tools


class TestCleanFilepath(unittest.TestCase):

    def test_clean_filepath_makes_path_absolute(self):
        relative_path = os.path.join('tests', 'data', 'example.tif')

        cleaned_path = loader_tools.clean_filepath(relative_path)

        self.assertTrue(os.path.isabs(cleaned_path))
        self.assertEqual(cleaned_path, os.path.abspath(relative_path))

    def test_clean_filepath_warns_for_long_path(self):
        long_path = 'a' * 257

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            cleaned_path = loader_tools.clean_filepath(long_path)

        self.assertEqual(cleaned_path, os.path.abspath(long_path))
        self.assertEqual(len(caught_warnings), 1)
        self.assertIn(
            'Caution: your file path is quite long',
            str(caught_warnings[0].message),
        )


class TestFilterFilenamesByFiletype(unittest.TestCase):

    def setUp(self):

        self.filenames = [
            'one.tif',
            'two.tif',
            'three.tiff',
            'four.txt',
            'five.csv',
            'six.bin',
            'seven.binn',
            'eight.tifff',
            'nine.png',
            'ten.jpg',
        ]

    def test_tiff(self):

        filenames_tiff = [
            'one.tif',
            'two.tif',
            'three.tiff',
        ]

        # make sure that the function is not case sensitive in any way
        filetypes_tif = map("".join, itertools.product(*zip(
            'tif'.upper(),
            'tif'.lower()
        )))
        for filetype in filetypes_tif:
            self.assertListEqual(
                loader_tools.filter_filenames_by_filetype(
                    self.filenames, filetype),
                filenames_tiff
            )

        filetypes_tiff = map("".join, itertools.product(*zip(
            'tiff'.upper(),
            'tiff'.lower()
        )))
        for filetype in filetypes_tiff:
            self.assertListEqual(
                loader_tools.filter_filenames_by_filetype(
                    self.filenames, filetype),
                filenames_tiff
            )

    def test_nist_bin(self):

        filenames_bin = [
            'six.bin',
        ]

        # make sure that the function is not case sensitive in any way
        filetypes = map("".join, itertools.product(*zip(
            'nist_bin'.upper(),
            'nist_bin'.lower()
        )))
        for filetype in filetypes:
            self.assertListEqual(
                loader_tools.filter_filenames_by_filetype(
                    self.filenames, filetype),
                filenames_bin
            )

    def test_nist_dash_bin(self):

        filenames_bin = [
            'six.bin',
        ]

        # make sure that the function is not case sensitive in any way
        filetypes = map("".join, itertools.product(*zip(
            'nist-bin'.upper(),
            'nist-bin'.lower()
        )))
        for filetype in filetypes:
            self.assertListEqual(
                loader_tools.filter_filenames_by_filetype(
                    self.filenames, filetype),
                filenames_bin
            )

    def test_none_filetype_returns_all_filenames(self):
        self.assertListEqual(
            loader_tools.filter_filenames_by_filetype(self.filenames, None),
            self.filenames,
        )

    def test_smi_h5_filters_only_h5_files(self):
        filenames = self.filenames + ['eleven.h5', 'twelve.h55']
        self.assertListEqual(
            loader_tools.filter_filenames_by_filetype(filenames, 'smi-h5'),
            ['eleven.h5'],
        )

    def test_smi_h5_underscore_filters_only_h5_files(self):
        filenames = self.filenames + ['eleven.h5', 'twelve.h55']
        self.assertListEqual(
            loader_tools.filter_filenames_by_filetype(filenames, 'smi_h5'),
            ['eleven.h5'],
        )

    def test_unsupported_filetype_returns_original_filenames(self):
        self.assertListEqual(
            loader_tools.filter_filenames_by_filetype(self.filenames, 'unknown'),
            self.filenames,
        )


class TestExtractMetadataFromPattern(unittest.TestCase):

    def setUp(self):
        self.filename = "sample_phi_deg_-45.0_sdd_m_5.02_sample_25A.tif"
        self.pattern = (
            "sample_phi_deg_{sample_phi_deg}_sdd_m_{sdd_cm}_"
            "sample_{sample_name}.tif"
        )
        self.scales = {'sdd_cm': 100}
        self.metadata, self.user_params = (
            loader_tools.extract_metadata_from_pattern(
                {'energy_ev': 16100},
                {},
                self.filename,
                self.pattern,
                self.scales,
            )
        )

    def testMetadata(self):

        self.assertAlmostEqual(self.metadata['sample_phi_deg'], -45.0)
        self.assertAlmostEqual(self.metadata['sdd_cm'], 502)
        self.assertAlmostEqual(self.metadata['energy_ev'], 16100)

    def testUserParams(self):

        self.assertEqual(self.user_params['sample_name'], "25A")

    def test_non_matching_pattern_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            loader_tools.extract_metadata_from_pattern(
                {}, {}, 'other_file.tif', self.pattern, self.scales
            )

    def test_no_pattern_no_scales_returns_inputs(self):
        metadata = {'energy_ev': 16100}
        user_params = {'sample_name': '25A'}

        returned_metadata, returned_user_params = (
            loader_tools.extract_metadata_from_pattern(
                metadata,
                user_params,
                self.filename,
                None,
                None,
            )
        )

        self.assertEqual(returned_metadata, metadata)
        self.assertEqual(returned_user_params, user_params)

    def test_warns_when_scale_key_missing(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            metadata, user_params = loader_tools.extract_metadata_from_pattern(
                {},
                {},
                self.filename,
                None,
                {'missing_key': 10},
            )

        self.assertEqual(metadata, {})
        self.assertEqual(user_params, {})
        self.assertEqual(len(caught_warnings), 1)
        self.assertIn(
            'Did not find the parameter missing_key to scale.',
            str(caught_warnings[0].message),
        )

    def test_uses_basename_only(self):
        metadata, user_params = loader_tools.extract_metadata_from_pattern(
            {},
            {},
            os.path.join('nested', 'directory', self.filename),
            self.pattern,
            self.scales,
        )

        self.assertAlmostEqual(metadata['sample_phi_deg'], -45.0)
        self.assertAlmostEqual(metadata['sdd_cm'], 502)
        self.assertEqual(user_params['sample_name'], '25A')

    def test_invalid_metadata_dtype_raises_value_error(self):
        with self.assertRaises(ValueError):
            loader_tools.extract_metadata_from_pattern(
                {},
                {},
                'sample_energy_ev_not_a_number.tif',
                'sample_energy_ev_{energy_ev}.tif',
                None,
            )


class TestGenerateDataNameFromPattern(unittest.TestCase):

    def setUp(self):
        pattern = "Sample: {sample}, Phi: {sample_phi_deg} deg"
        metadata = {'sample_phi_deg': -45}
        user_params = {'sample': '25A'}
        self.name = loader_tools.generate_data_name_from_pattern(
            pattern, metadata, user_params
        )

    def testName(self):

        self.assertEqual(self.name,
                         "Sample: 25A, Phi: -45 deg")

    def test_missing_pattern_key_is_left_in_braces(self):
        name = loader_tools.generate_data_name_from_pattern(
            "Sample: {sample}, Run: {run}",
            {'sample_phi_deg': -45},
            {'sample': '25A'},
        )

        self.assertEqual(name, "Sample: 25A, Run: {run}")

    def test_none_pattern_falls_back_to_filename(self):
        name = loader_tools.generate_data_name_from_pattern(
            None,
            {'filename': 'example.tif'},
            {},
        )

        self.assertEqual(name, 'example.tif')

    def test_none_metadata_and_user_params_leave_missing_keys_in_braces(self):
        name = loader_tools.generate_data_name_from_pattern(
            'Sample: {sample}',
            None,
            None,
        )

        self.assertEqual(name, 'Sample: {sample}')

    def test_none_pattern_without_filename_returns_none(self):
        name = loader_tools.generate_data_name_from_pattern(
            None,
            {'sample_phi_deg': -45},
            {},
        )

        self.assertIsNone(name)

    def test_repeated_placeholders_are_all_replaced(self):
        name = loader_tools.generate_data_name_from_pattern(
            'Sample {sample} / {sample}',
            {},
            {'sample': '25A'},
        )

        self.assertEqual(name, 'Sample 25A / 25A')

    def test_missing_placeholder_substrings_are_preserved(self):
        name = loader_tools.generate_data_name_from_pattern(
            'Sample: {sample}, Phi: {sample_phi_deg} deg',
            None,
            None,
        )

        self.assertEqual(name, 'Sample: {sample}, Phi: {sample_phi_deg} deg')
