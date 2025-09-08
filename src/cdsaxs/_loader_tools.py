"""
Tools for the loader module.
"""
import os
import re

import warnings

from cdsaxs.metadata import METADATA_KEYWORDS, correct_metadata_dtype


def clean_filepath(filepath):
    """checks filepath length and returns an os formatted absolute path"""
    filepath = os.path.abspath(filepath)
    if len(filepath) > 256:
        warnings.warn(
            "Caution: your file path is quite long"
            f"({len(filepath)} characters) and might result in"
            "this loader failing.")
    return filepath


def extract_exposure_time_pilatus(header):
    """
    Extract exposure time in seconds from the TIFF file header"
    of a Pilaturs detectr."
    """

    try:
        if type(header["ImageDescription"]) is str:
            image_desc = header["ImageDescription"]
        else:
            image_desc = header["ImageDescription"][0]
        _, value, units = [
            x for x in image_desc.split('#')
            if 'Exposure_time' in x][0].split()
        if units != 's':
            warnings.warn(
                "Exposure time is in wrong units; returning None.")
            return None
        else:
            return float(value)
    except:
        warnings.warn(
            "Count not extract count time from file; returning None.")
        return None


def extract_pixel_size_pilatus(header):

    "Extract pixel time in um from the TIFF file header."

    try:
        if type(header["ImageDescription"]) is str:
            image_desc = header["ImageDescription"]
        else:
            image_desc = header["ImageDescription"][0]

        _, value0, units0, _, value1, units1 = [
            x for x in image_desc.split('#')
            if 'Pixel_size' in x][0].split()

        if units0 != 'm' or units1 != 'm':
            warnings.warn(
                "Pixel size is in the wrong units; returning None."
            )
            return None
        if float(value0) != float(value1):
            raise ValueError(
                "Pixel dimensions are not square. This is currently"
                "not implemented in the code and requires consideration."
            )
        return float(value0) * 1e6
    except:
        warnings.warn(
            "Could not extract pixel size from the header; returning None"
        )
        return None


def filter_filenames_by_filetype(filenames, filetype):
    """
    Accepted filtypes:
        tiff or tif
        nist-bin
    """
    if filetype is None:
        return filenames

    if filetype.lower() in ['tiff', 'tif']:
        filenames = [x for x in filenames if 'tif' == x.split('.')[-1]
                     or 'tiff' == x.split('.')[-1]]
    elif filetype.lower() in ['nist_bin', 'nist-bin']:
        filenames = [x for x in filenames if 'bin' == x.split('.')[-1]]

    return filenames


def extract_metadata_from_pattern(metadata, user_params, filename,
                                  metadata_pattern, metadata_scales):
    """
    Extract metadata from the filename and place into the appropriate
    attribute of metadata or user_params.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary to store results. Only uses accepted
        metadata keywords.
    user_params : dict
        Additional metadata parameters with user-specified keys.
    filename : filename from which to extract the metadata
    metadata_pattern : str
        Extract metadata from information stored in the filenames.
        A pattern for the filenames can be provided where the
        keywords for either metadata or user parameters should be
        enclosed in {}. For example, if two images had filenames of:
            sample1_phi0_sdd_500_run001.tif
            sample1_phi-1_sdd_500_run002.tif
        The following pattern could be provided to extract meatadata
        parameters of 'sample_phi_deg' and 'sdd_cm' as well as user
        parameter 'run' for each data:
            sample1_phi{sample_phi_deg}_sdd_{sdd_cm}_run{run}.tif
        NOTE: conflicts can arise if both this pattern and the
        metadata_csv_filepath are provided. Metadata parameters
        specified in both places can result in one overwriting the
        other.
    metadata_scales : dict
        If any of the metadata was provided in incorrect units, a
        scaling value can be provided to perform unit conversions. The
        argument should be provided as a dictionary where the key
        is the metadata keyword or user_params keyword, and the
        value is the amount by which to scale or multiply the
        parameter's current value.
        NOTE: this will only apply to values extracted from the metadata
        pattern of the filenames.

    """
    filename = os.path.basename(filename)
    if metadata_pattern is not None:
        regex = re.sub(r'{(.+?)}', r'(?P<\1>.+)', metadata_pattern)
        values = list(re.search(regex, filename).groups())
        keys = re.findall(r'{(.+?)}', metadata_pattern)
        for key, value in zip(keys, values):
            if key in METADATA_KEYWORDS:
                metadata[key] = correct_metadata_dtype(key, value)
            else:
                user_params[key] = value

    if metadata_scales is not None:
        for key, value in metadata_scales.items():
            if key in metadata.keys():
                metadata[key] = metadata[key]*value
            elif key in user_params.keys():
                user_params[key] = user_params[key]*value
            else:
                warnings.warn(
                    f"Did not find the parameter {key} to scale."
                )

    return metadata, user_params


def generate_data_name_from_pattern(data_name_pattern,
                                    metadata={},
                                    user_params={}):
    """
    Generate the name for a signle data instance from a pattern and the
    metadata parameters.
    """

    # handle any None metadata or user_params that may get passed
    if metadata is None:
        metadata = {}
    if user_params is None:
        user_params = {}

    if data_name_pattern is not None:
        new_name = data_name_pattern
        while '{' in new_name and '}' in new_name:
            start = new_name.find('{')
            stop = new_name.find('}')
            key = new_name[start+1:stop]
            if key in metadata.keys():
                value = metadata[key]
            elif key in user_params.keys():
                value = user_params[key]
            else:
                value = key
            old_str = "{"+key+"}"
            new_str = str(value)
            new_name = new_name.replace(old_str, new_str)
    else:
        new_name = data_name_pattern

    return new_name
