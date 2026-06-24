import os

import fabio as fabio
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile

import cdsaxs.loaders._loader_tools as loader_tools


def read_tiff(filepath):
    """
    Load an image and header from a tiff file.

    Parameters
    ----------
    filepath : str, path
        Path to the tiff file to be loaded.

    Returns
    -------
    NDArray
        Two-dimensional numpy array that contains the image data.
    str
        Formatted filepath used to load the data.
    dict
        Dictionary of the header information where the key: value paris
        correpond to the tag.name: tag.value pairs of the header tags.
    """
    filepath = loader_tools.clean_filepath(filepath=filepath)

    try:
        image = Image.open(filepath)
        image = np.array(image).astype(np.float64)
        header = {
            TAGS[key]: image.tag[key] for key in image.tag_v2
            if key in TAGS.keys()
            }
    except:
        image = tifffile.imread(filepath).astype(np.float64)
        with tifffile.TiffFile(filepath) as tif:
            header = {
                tag.name: tag.value
                for tag in tif.pages[0].tags}

    return image, filepath, header


def read_nist_bin(filepath):
    """
    Load an image and metadata from a NIST-formatted bin/info file pair
    from the CD-SAXS instrument in group 642.06.

    Parameters
    ----------
    filepath : str, path
        Path to the bin file to be loaded.
        The paired info file should be in the same directory and have
        the same filename (apart from the different extension).

    Returns
    -------
    NDArray
        Two-dimensional numpy array that contains the image data.
    str
        Formatted filepath used to load the data.
    dict
        Dictionary with metadata keyword: value pairs.
    """

    filepath = loader_tools.clean_filepath(filepath=filepath)

    # read the image from the .bin file first
    image = np.fromfile(filepath, dtype=np.float64)[1:].reshape(195, 1475)

    # read the .info file
    infopath = os.path.join(
        os.path.dirname(filepath),
        os.path.basename(filepath)[:-4] + ".info"
    )
    file = open(infopath)
    sample_meta = file.readlines()
    sample_meta = {x.split('=')[0]: x.split('=')[1] for x in sample_meta}
    file.close()

    # extact required information and insert into clean dictionary
    metadata = {}
    metadata['wavelength_nm'] = float(sample_meta['Wavelength (nm) '])
    metadata['exposure_time_s'] = float(sample_meta['LiveTime '])
    metadata['pixel_size_um'] = float(sample_meta['Pixel Size '])

    return image, filepath, metadata


def read_nist_edf(filepath):
    """
    Load an image and metadata from the Xenocs Xeuss Pro instrument
    at NIST that stores data in edf files.

    Parameters
    ----------
    filepath : str, path
        Path to the bin file to be loaded.
        The paired info file should be in the same directory and have
        the same filename (apart from the different extension).

    Returns
    -------
    NDArray
        Two-dimensional numpy array that contains the image data.
    str
        Formatted filepath used to load the data.
    dict
        Dictionary with metadata keyword: value pairs.
    """

    filepath = loader_tools.clean_filepath(filepath=filepath)

    # read image and header from the file
    with fabio.open(filepath) as file:
        image = file.data
        header = file.header

    # extact required information and insert into clean dictionary
    metadata = {}
    metadata['wavelength_nm'] = float(header['WaveLength']) * 1e9  # m to Ang
    metadata['exposure_time_s'] = float(header['ExposureTime'])
    metadata['pixel_size_um'] = float(header['PSize_1']) * 1e6  # m to um
    metadata['sample_phi_deg'] = float(header['CD_Phi'])
    metadata['sample_chi_deg'] = float(header['CD_Ry'])
    metadata['sample_omega_deg'] = float(header['CD_Rx'])
    metadata['center_px'] = (float(header["Center_2"]), float(header["Center_1"]))
    metadata['sdd_cm'] = float(header["SampleDistance"]) * 1e2  # m to cm
    try:
        metadata['sample_reference'] = header["sample_reference"]
    except:
        pass
    try:
        metadata['sample_name'] = header['sample_name']
    except:
        pass
    try:
        metadata['sample_comment'] = header['sample_comment']
    except:
        pass
    metadata['date'] = header["Date"]
    metadata['sample_stage_x'] = float(header["x"])
    metadata['sample_stage_y'] = float(header["z"])

    return image, filepath, metadata
