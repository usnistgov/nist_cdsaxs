"""
This module contains dataset loaders. Each loader must create one
instance of data.Dataset that contains one or more instances of
data.DataQyQxz corresponding to each scattering file in the loaded
dataset.
"""

from __future__ import annotations
import os

import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile

from cdsaxs.data import METADATA_KEYWORDS, DataQyQxz, Dataset


class TiffTools():
    """Tools for handling TIFF image files."""

    def __init__(self, filepath):
        """Read an image and header information from TIFF file."""

        filepath = os.path.abspath(filepath)
        self.filepath = filepath
        try:
            image = Image.open(filepath)
            self.image = np.array(image).astype(float)

            header = {TAGS[key]: image.tag[key] for key in image.tag_v2
                      if key in TAGS.keys()}
            self.header = header
        except:
            image = tifffile.imread(filepath).astype(float)
            self.image = image

            with tifffile.TiffFile(filepath) as tif:
                self.header = {tag.name : tag.value
                               for tag in tif.pages[0].tags}

    def extract_count_time(self, filepath):

        "Extract count time in seconds from the TIFF file header."
        
        try:
            keyword, value, units = [
                x for x in self.header['ImageDescription'][0].split('#')
                if 'Exposure_time' in x][0].split()
            if units != 's':
                raise ValueError("Count time is in wrong units.")
            else:
                return float(value)
        except:
            raise ValueError("Count not extract count time from file.")


class GeneralTIFFLoader():
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

    Required Metadata Parameters
    ----------------------------
    filename : tiff filename (not a file path)
    sample_phi_deg : sample rotation angle about positive y-axis, degrees
    *One of the following:*
        energy_ev : source energy, eV
        wavelength_nm : source wavelength, nm
    exposure_time_s : exposture time of image, s

    Optional Metadata Parameters
    ----------------------------
    scaling_factor : data scaling factor, defaults to 1
    detector_theta
    detector_x
    sdd_cm : sample to detector distance in cm
    sample_chi_deg : rotation about the z-axis (beam direction)
    div_photodiode
    center_px : [xz, y] beam center pixel location in xz and y
    """

    def __init__(self, filepath_csv, name=None):

        # load the csv metadata file
        csv_data = np.loadtxt(filepath_csv, dtype='str', delimiter=',')
        header = csv_data[0, :]
        csv_data = csv_data[1:, :]

        # extract data directory
        folder, _ = os.path.split(filepath_csv)
        print('Made dataset from ' + self.folder)

        dataset = Dataset(self.folder, name=name)

        for i, row in enumerate(csv_data):
            metadata = {}
            params = {}
            for value in row:
                if header[i] in METADATA_KEYWORDS and header[i] != 'filename':
                    metadata[header[i]] = value
                elif header[i] == 'filename':
                    filename = value
                else:
                    params[header[i]] = value
            image = TiffTools(os.path.join(folder, filename)).image

            if 'center_px' not in metadata.keys():
                # default center pixel at bottom right of image
                metadata['center_px'] = [image.shape[0]-1, image.shape[1]-1]

            if 'pixel_size_um' not in metadata.keys():
                # default pixel size
                metadata['pixel_size_um'] = 172

            qxz_px = -1*np.arange(0, image.shape[1]) + metadata['center_px'][1]
            qy_px = -1*np.arange(0, image.shape[0]) + metadata['center_px'][0]

            # TODO: URGENT - fix this once we have diffraction function!
            qxzs = np.copy(qxz_px).astype(float)
            qys = np.copy(qy_px).astype(float)

            data = DataQyQxz(image, qys, qxzs, metadata, params=params)

            dataset.add_data(filename, data)
