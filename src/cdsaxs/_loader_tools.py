"""
Tools for the loader module.
"""
import os

import warnings


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
