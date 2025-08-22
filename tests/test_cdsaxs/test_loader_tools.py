import os
import itertools

import numpy as np
import unittest

import cdsaxs._loader_tools as loader_tools
import cdsaxs.loaders as loaders


class TestPilatusHeader(unittest.TestCase):

    def setUp(self):

        image_file = "../data/test_loaders/smi.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, image_file)

        self.image, self.filepath, self.header = loaders.read_tiff(image_path)

    def test_exposure_time(self):

        print(os.path.abspath("."))
        exposure_time = 0.2  # seconds

        self.assertEqual(
            loader_tools.extract_exposure_time_pilatus(self.header),
            exposure_time)

    def test_pixel_size(self):

        print(os.path.abspath("."))
        pixel_size = 172  # mircrons

        self.assertEqual(
            loader_tools.extract_pixel_size_pilatus(self.header),
            pixel_size)


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
