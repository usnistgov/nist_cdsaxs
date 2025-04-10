"""
This module contains dataset loaders. Each loader must create one
instance of data.Dataset that contains one or more instances of
data.DataQyQxz corresponding to each scattering file in the loaded
dataset.
"""

from __future__ import annotations
import os
import warnings

import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile

from cdsaxs.data2d import DataQdyQdx, Dataset
from cdsaxs.metadata import correct_dtype, METADATA_KEYWORDS


class TiffTools():
    """Tools for handling TIFF image files."""

    def __init__(self, filepath):
        """Read an image and header information from TIFF file."""

        filepath = os.path.abspath(filepath)
        self.filepath = filepath
        try:
            image = Image.open(filepath)
            self.image = np.array(image).astype(np.float32)

            header = {TAGS[key]: image.tag[key] for key in image.tag_v2
                      if key in TAGS.keys()}
            self.header = header
        except:
            image = tifffile.imread(filepath).astype(np.float32)
            self.image = image

            with tifffile.TiffFile(filepath) as tif:
                self.header = {tag.name: tag.value
                               for tag in tif.pages[0].tags}

    def extract_exposure_time(self):

        "Extract exposure time in seconds from the TIFF file header."

        try:
            key, value, units = [
                x for x in self.header['ImageDescription'][0].split('#')
                if 'Exposure_time' in x][0].split()
            if units != 's':
                raise ValueError("Exposure time is in wrong units.")
            else:
                return float(value)
        except:
            raise ValueError("Count not extract count time from file.")


def GeneralTIFFLoader(filepath_csv, name=None):
    """
    General TIFF loader. Any scattering metadata or user-defined
    parameters should be passed as a csv file where the first row is
    comprised of the column headers. The csv file should be in the
    same directory as the tiff files.

    The column headers should be selected from the following list when
    possible. The user can specify their own unique headers to store
    user-specified parameters associated with each file, but these
    will not be recognized by the built-in processes in CD-SAXS. They
    can, however, be useful when developing custom workflows in Jupyter
    notebooks. These additional parmaeters will get passed
    to data.DataQyQxz.params.

    List of accepted CSV column headers:
        Required
        --------
        filename
        sample_phi_deg : sample rotation angle during cd-saxs measurement in degrees
        energy_ev : source energy in eV (cannot be used with wavelength_nm)
        wavelength_nm : source wavelength in nm (cannot be used with energy_ev)
        exposure_time_s : exposture time in s

        Optional
        --------
        sample_label : user-specified sample label
        sdd_cm : sample-to-detector distance in cm
        sample_chi_deg : rotation in the sample xy plane about the z axis
    """

    # load the csv metadata file
    csv_data = np.loadtxt(filepath_csv, dtype='str', delimiter=',')
    header = csv_data[0, :]
    csv_data = csv_data[1:, :]

    # extract data directory
    folder, _ = os.path.split(filepath_csv)
    print('Made dataset from ' + folder)

    dataset = Dataset(name=name)

    for i, row in enumerate(csv_data):
        metadata = {}
        params = {}
        for ii, value in enumerate(row):
            if header[ii] in METADATA_KEYWORDS:
                metadata[str(header[ii])] = correct_dtype(header[ii], value)
            else:
                params[str(header[ii])] = value
        metadata["data_directory"] = folder
        tiff = TiffTools(os.path.join(folder, metadata["filename"]))
        if "exposure_time_s" not in metadata.keys():
            try:
                metadata["exposure_time_s"] = tiff.extract_exposure_time()
            except:
                pass

        image = tiff.image

        if 'center_px' not in metadata.keys():
            # default center pixel at bottom right of image
            metadata['center_px'] = [image.shape[0]-1, image.shape[1]-1]

        if 'pixel_size_um' not in metadata.keys():
            # default pixel size
            metadata['pixel_size_um'] = 172
            warnings.warn(
                f"Using default pixel size of {metadata['pixel_size_um']}")

        data = DataQdyQdx(image, metadata=metadata, user_params=params)

        dataset.add_data(data)

    return dataset
