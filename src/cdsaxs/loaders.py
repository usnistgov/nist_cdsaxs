"""
This module contains dataset loaders. Each loader must create one
instance of data.Dataset that contains one or more instances of
data.DataQyQxz corresponding to each scattering file in the loaded
dataset.
"""

from __future__ import annotations
import os
import re
import warnings

import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile
from tqdm import tqdm

from cdsaxs.data2d import DataQdyQdx
from cdsaxs.dataset import Dataset
import cdsaxs.metadata
from cdsaxs.metadata import correct_dtype, METADATA_KEYWORDS


class TiffTools():
    """Tools for handling TIFF image files."""

    def __init__(self, filepath):
        """Read an image and header information from TIFF file."""

        if len(filepath) > 256:
            warnings.warn(
                "Caution: your file path is quite long"
                f"({len(filepath)} characters) and might result in"
                "this loader failing.")

        filepath = os.path.abspath(filepath)
        self.filepath = filepath
        try:
            image = Image.open(filepath)
            self.image = np.array(image).astype(np.float64)

            header = {TAGS[key]: image.tag[key] for key in image.tag_v2
                        if key in TAGS.keys()}
            self.header = header
        except:
            image = tifffile.imread(filepath).astype(np.float64)
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


def GeneralTIFFLoader(filepath_csv, name):
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
        sample_phi_deg : sample rotation angle during cd-saxs in degrees
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
        # treat pixels with negative values as nan
        image[image < 0] = np.nan

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


def GeneralTIFFLoader_MetadataKeywords(directory_path, name,
                                       pattern=None, scales=None,
                                       filter_by_substrings=None,
                                       filter_by_names=None,
                                       verbose=True,
                                       data_name=None):
    """
    General TIFF loader that pulls metadata from keywords in the
    filename. The keywords must match the metadata keywords in this
    library exactly. This loader will assume that all tiff images in the
    directory provided should be imported. It will ignore other files
    with a different format.

    The following keywords are required in the filename:
        sample_phi_deg : sample rotation angle during cd-saxs in degrees
        energy_ev : source energy in eV (cannot be used with wavelength_nm)
        wavelength_nm : source wavelength in nm (only if energy_ev unavailable)
        exposure_time_s : exposture time in s

    All other keywords are optional. This function will search the
    filename for all the keywords that it recognizes.

    Alternatively, regular expressions can be used to assign sections
    of the filename to different metadata keywords and user parameters.
    In this case, the variable names should match the accepted metadata
    keywords, otherwise they will be stored in the user params
    dictionary of the 2D data. Please see the markdown file in
    nist_cdsaxs/extras/SMI_filename_format_20250728.md for more info.

    An example of this is:

    filename = 'test_sample_sdd_cm_520_energy_ev_16100.tif
    pattern = "{name}_sdd_cm_{sdd_cm}_energy_ev_{energy_ev}.tif"

    The following will get stored in data.metadata:
        'name' = 'test_sample'
        'sdd_cm' = 520
        'energy_ev' = 161000

    Scaling values for any of the extracted key: value pairs can also
    be provided as a dictionary of keyword: scale pairs. This enables
    the user to control unit conversions as needed.

    For example, if the energy_ev was provided as 16.1 (units keV), the
    following dictionary can be passed for the scales argument:
    {'energy_ev': 1000} so that 'energy_ev':16100 will be stored in the
    metadata dictionary with correct units.
    TODO: implement unit handling for metadata in the future

    The files can also be filtered so that this function does not load
    in all the images at once. A string or list of strings can be
    provided and this loader will read any files that include any one
    or more of these keywords. If you would like to ensure that two
    or more keywords are included in the same filename, please nest
    them in another list. For example:

    [['red', 'apple'], 'orange', 'strawberry']

    If filtering by the keyword list above, any files that contain the
    word orange or strawberry or BOTH red and apple will be loaded. Files
    that include all of these words will also be loaded.

    If you would like more control over which files are loaded, you
    can feed a list of the filenames directly as the 'filenames' keyword
    argument to this laoder.

    """

    # create a list of files in the provided directory
    directory_path = os.path.abspath(directory_path)
    filenames = [x for x in os.listdir(directory_path) if '.tif' in x]

    if filter_by_substrings is not None and filter_by_names is not None:
        raise ValueError(
            "You cannot define both filter_by_substrings"
            "and filter_by_names. Please choose one or the"
            "other to select which files to load.")
    elif filter_by_substrings is not None:
        filtered_filenames = []
        if type(filter_by_substrings) is str:
            filter_by_substrings = [filter_by_substrings]
        for string in filter_by_substrings:
            if type(string) is str:
                filtered_filenames.extend([
                    x for x in filenames if string in x])
            else:
                temp_filtered = filenames.copy()
                for string_i in string:
                    temp_filtered = [x for x in temp_filtered if string_i in x]
                filtered_filenames.extend(temp_filtered)
        filenames = list(set(filtered_filenames))
    elif filter_by_names is not None:
        filenames = filter_by_names

    print(f"Found {len(filenames)} images to load into this dataset.")

    dataset = Dataset(name=name)

    if verbose:
        pbar = tqdm(range(len(filenames)), desc="Loading files: ",
                    position=0, leave=True)

    for i, filename in enumerate(filenames):
        metadata = {}
        params = {}

        # if a pattern is provided use that to interpret filename
        if pattern is not None:
            regex = re.sub(r'{(.+?)}', r'(?P<\1>.+)', pattern)
            values = list(re.search(regex, filename).groups())
            keys = re.findall(r'{(.+?)}', pattern)
            for key, value in zip(keys, values):
                if key in METADATA_KEYWORDS:
                    metadata[key] = correct_dtype(key, value)
                else:
                    params[key] = value

        # otherwise use the standard accepted keyword filename format
        else:
            filename_clean = filename[:filename.find('.tif')]
            for keyword in METADATA_KEYWORDS:
                keyword_search = f'_{keyword}_'
                loc = filename_clean.find(keyword_search)
                if loc != -1:
                    value = filename_clean[loc+len(keyword_search):].split('_')[0]
                    metadata[keyword] = correct_dtype(keyword, value)

        # apply any scaling parameters
        if scales is not None:
            for key, value in scales.items():
                if key in metadata.keys():
                    metadata[key] = metadata[key]*value
                elif key in params.keys():
                    params[key] = float(params[key])*value
                else:
                    print(f"WARNING: the scale for {key} was not applied"
                          "as the keyword could not be found in metadata or"
                          "user params.")

        # add in data directory and filename as metadata always
        metadata["data_directory"] = directory_path
        metadata["filename"] = filename

        # load the image and try to extract info
        tiff = TiffTools(os.path.join(directory_path, filename))
        if "exposure_time_s" not in metadata.keys():
            try:
                metadata["exposure_time_s"] = tiff.extract_exposure_time()
            except:
                pass
        image = tiff.image
        # treat pixels with negative values as nan
        image[image < 0] = np.nan

        if 'center_px' not in metadata.keys():
            # default center pixel at bottom right of image
            metadata['center_px'] = [image.shape[0]-1, image.shape[1]-1]

        if 'pixel_size_um' not in metadata.keys():
            # default pixel size
            metadata['pixel_size_um'] = 172
            warnings.warn(
                f"Using default pixel size of {metadata['pixel_size_um']}")

        if data_name is not None:
            new_name = data_name
            while '{' in new_name and '}' in new_name:
                start = new_name.find('{')
                stop = new_name.find('}')
                key = new_name[start+1:stop]
                if key in metadata.keys():
                    value = metadata[key]
                elif key in params.keys():
                    value = params[key]
                else:
                    value = key
                old_str = "{"+key+"}"
                new_str = str(value)
                new_name = new_name.replace(old_str, new_str)
        else:
            new_name = data_name

        data = DataQdyQdx(image, metadata=metadata,
                          user_params=params,
                          name=new_name)

        dataset.add_data(data)

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    print('Made dataset from ' + directory_path)

    return dataset
