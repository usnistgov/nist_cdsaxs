import os
import unittest

import numpy as np

import cdsaxs.loaders as loaders


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


class testReadNistBin(unittest.TestCase):

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


class testReadPilatus(unittest.TestCase):

    def setUp(self):
        image_file = "../data/test_loaders/smi.tif"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.abspath(
            os.path.join(current_dir, image_file))
        self.return_read_filename = loaders.read_pilatus(
            filepath=self.filepath)
        self.return_read_header = loaders.read_pilatus(
            header=loaders.read_tiff(self.filepath)[2]
        )

    def test_image(self):
        image = self.return_read_filename[0]

        self.assertEqual(image[735, 450], 12)
        self.assertEqual(image.dtype, np.float64)

    def test_filepath(self):
        filepath = self.return_read_filename[1]

        self.assertEqual(filepath, self.filepath)

    def test_header(self):
        metadata = self.return_read_filename[2]
        self.assertEqual(
            metadata["exposure_time_s"],
            0.2
        )
        self.assertEqual(
            metadata["pixel_size_um"],
            172
        )

    def test_read_header(self):
        metadata = self.return_read_header[2]
        self.assertEqual(
            metadata["exposure_time_s"],
            0.2
        )
        self.assertEqual(
            metadata["pixel_size_um"],
            172
        )


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
    # TODO: implement
    pass


class TestLoadDataset(unittest.TestCase):
    # TODO: implement
    pass


class TestLoadDataset_MetadataCSV(unittest.TestCase):
    # TODO: implement
    pass
