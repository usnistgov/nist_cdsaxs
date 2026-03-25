import itertools

import unittest

import cdsaxs.loaders._loader_tools as loader_tools


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


class TestExtractMetadataFromPattern(unittest.TestCase):

    def setUp(self):
        self.filename = "sample_phi_deg_-45.0_sdd_m_5.02_sample_25A.tif"
        self.pattern = "sample_phi_deg_{sample_phi_deg}_sdd_m_{sdd_cm}_sample_{sample_name}.tif"
        self.scales = {'sdd_cm': 100}
        self.metadata, self.user_params\
            = loader_tools.extract_metadata_from_pattern(
                {'energy_ev': 16100}, {}, self.filename, self.pattern, self.scales
            )

    def testMetadata(self):

        self.assertAlmostEqual(self.metadata['sample_phi_deg'], -45.0)
        self.assertAlmostEqual(self.metadata['sdd_cm'], 502)
        self.assertAlmostEqual(self.metadata['energy_ev'], 16100)

    def testUserParams(self):

        self.assertEqual(self.user_params['sample_name'], "25A")


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
