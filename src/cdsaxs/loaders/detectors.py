import warnings

from .filetypes import read_tiff


def read_pilatus(filepath=None):
    """
    Read a tiff file from a Pilatus detector, returning the image and
    the formatted header as metadata. If only the header is provided as
    a dictionary of tag.name: tag.value pairs, only the metadata
    dictionary will be returned with None as the image.

    Parameters
    ----------
    filepath : str
        Filepath to the tiff file from a Pilatus detector.

    Returns
    -------
    image : NDArray | None
        Image loaded from the TIFF file. If filepath is not provided,
        this is None.
    filepath : str | None
        Normalized filepath from which the image was loaded. If filepath
        is not provided, this is None.
    metadata : dict
        Metadata dictionary with accepted metadata keywords extracted
        from the TIFF header.
    """
    image, filepath, header = read_tiff(filepath=filepath)
    metadata = pilatus_header_to_metadata(header)

    return image, filepath, metadata


def pilatus_header_to_metadata(header):
    """
    Extract exposure time and pixel size from a pilatus detector file
    header.

    Parameters
    ----------
    header : dict
        Dictionary of tag.name: tag.value pairs from a tiff file from
        a Pilatus detector.

    Returns
    -------
    metadata : dict
        Metadata dictionary with accepted metadata keywords extracted
        from the TIFF header.
    """
    metadata = {}

    # exposure time
    exposure_time_s = extract_exposure_time_pilatus(header)
    if exposure_time_s is not None:
        metadata["exposure_time_s"] = exposure_time_s

    # pixel size
    pixel_size_um = extract_pixel_size_pilatus(header)
    if pixel_size_um is not None:
        metadata["pixel_size_um"] = pixel_size_um
    else:
        metadata["pixel_size_um"] = 172
        warnings.warn(
            "Assuming a pixel size of 172 micron for Pilatus."
            "Please confirm this is correct before proceeding.")

    return metadata


def extract_exposure_time_pilatus(header):
    """
    Extract the exposure time in seconds from a Pilatus TIFF header.

    Parameters
    ----------
    header : dict
        TIFF header represented as tag-name to tag-value pairs.

    Returns
    -------
    exposure_time_s : float | None
        Exposure time in seconds. Returns None when the value cannot be
        extracted or is reported in unexpected units.
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
    """
    Extract the pixel size in micrometers from a Pilatus TIFF header.

    Parameters
    ----------
    header : dict
        TIFF header represented as tag-name to tag-value pairs.

    Returns
    -------
    pixel_size_um : float | None
        Pixel size in micrometers. Returns None when the value cannot be
        extracted or is reported in unexpected units.
    """

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
