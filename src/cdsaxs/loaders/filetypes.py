import os

import h5py
from astropy.io import fits
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile

from . import _loader_tools as loader_tools


def read_tiff(filepath):
    """
    Load an image and header from a tiff file.

    Parameters
    ----------
    filepath : str, path
        Path to the tiff file to be loaded.

    Returns
    -------
    image : NDArray
        Two-dimensional image array loaded from the TIFF file.
    filepath : str
        Normalized filepath used to load the data.
    header : dict
        Header information as tag-name to tag-value pairs.
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
    image : NDArray
        Two-dimensional image array loaded from the BIN file.
    filepath : str
        Normalized filepath used to load the data.
    metadata : dict
        Metadata dictionary containing accepted metadata key-value pairs
        extracted from the paired INFO file.
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

def read_smi_h5(filepath):
    """
    Load an image and metadata from an H5 file from the SMI beamline
    at NSLS-II. This will load an entire cd-saxs scan, i.e., set of
    images, not just one image at a time.

    Parameters
    ----------
    filepath : str, path
        Path to the H5 file to be loaded.

    Returns
    -------
    images : list[tuple[NDArray, str, dict]]
        List of per-image tuples containing the image array, normalized
        filepath, and metadata dictionary for each image in the scan.
    """

    filepath = loader_tools.clean_filepath(filepath=filepath)

    # load the entire scan
    scan = h5py.File(filepath, 'r')
    
    # extract image stack and number of images in the scan
    image_stack = scan['raw_images']['pil2M_image'][:]
    num_images = image_stack.shape[0]
    
    metadata = {}
    metadata['energy_ev'] = scan['baseline']['energy_energy'][0]
    metadata['sdd_cm'] = scan['baseline']['pil2M_motor_z'][0]/10
    metadata['exposure_time_s'] = scan['config']['pil2M_cam_acquire_time'][0]
    metadata['pixel_size_um'] = 172  # pilatus2m
    metadata['bpm'] = scan['primary']['xbpm3_sumX'][:]
    metadata['sample_phi_deg'] = scan['primary']['stage_phi'][:]

    seq_num = scan['primary']['seq_num'][:]  

    images = []
    for i in range(num_images):
        temp_metadata = dict(metadata)
        metadata_index = np.where(seq_num == i+1)[0]
        temp_metadata['bpm'] = metadata['bpm'][metadata_index][0]
        temp_metadata['sample_phi_deg'] = np.round(-1*metadata['sample_phi_deg'][metadata_index],2)[0]
        images.append(
            (image_stack[i].astype(np.float64), filepath, temp_metadata)
        )

    return images


def read_fits(filepath):
    """
    Load an image and header from a fits file.

    Parameters
    ----------
    filepath : str, path
        Path to the fits file to be loaded.

    Returns
    -------
    NDArray
        Two-dimensional numpy array that contains the image data.
    str
        Formatted filepath used to load the data.
    dict
        Dictionary of the header information where the key: value pairs.
    """

    filepath = loader_tools.clean_filepath(filepath=filepath)

    # load image
    image = fits.getdata(filepath, ext=2).astype(np.float64)

    # header dictionary
    info = [hdu.header for hdu in fits.open(filepath)][0]
    header = dict(info)

    return image, filepath, header


def read_als_11_0_1_2(filepath):

    """
    Load an image and metadata from data collected at beamline 11.0.1.2
    at ALS.

    CAUTION: This loader assumes data collected in 2025 or earlier
    uses a detector with a 27 um pixel size. Any data collected in
    2026 or later is assumed to have a pixel size of 9 um.

    Parameters
    ----------
    filepath : str, path
        Path to the fits file to be loaded.


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

    image, filepath, header = read_fits(filepath)


    metadata = {}
    metadata['energy_ev'] = header['Beamline Energy']
    metadata['sample_phi_deg'] = header['Sample Theta'] + 90
    metadata['I0'] = header['AI 3 Izero']
    metadata['exposure_time_s'] = header['EXPOSURE']
    metadata['beam_current'] = header['Beam Current']
    metadata['detector_phi_deg'] = header['CCD Theta']
    metadata['detector_y_mm'] = header['CCD X']
    metadata['CCD Y'] = header['CCD Y']
    metadata['epu_polarization'] = header['EPU Polarization']
    metadata['beam_stop_position'] = header['Beam Stop']
    date = header['DATE']
    if float(date[:4]) <= 2025:
        metadata['pixel_size_um'] = 27
        # orient the detector image with our coordinate system
        image = np.flipud(np.rot90(image, 3))
    else:
        metadata['pixel_size_um'] = 9
        # orient the detector image with our coordinate system
        image = np.flipud(np.rot90(image, 2))

    return image, filepath, metadata
