from datetime import datetime, date
from zoneinfo import ZoneInfo
import os

import fabio as fabio
import h5py
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import tifffile

from . import _loader_tools as loader_tools


SMI_H5_PHI_FLIP_DATE = date(2026, 7, 8)  # this date is when the H5 scan file format flipped the sign of the phi rotation to align to right-handed rule

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


def read_nist_edf(filepath):
    """
    Load an image and metadata from the Xenocs Xeuss Pro instrument
    at NIST that stores data in edf files.
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

    
def read_smi_h5(filepath):
    """
    Load an image and metadata from an H5 file from the SMI beamline
    at NSLS-II. This will load an entire cd-saxs scan, i.e., set of
    images, not just one image at a time.

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

    # scale phi by +1 or -1 depending on when the h5 file was exported
    # Eliot fixed the rotation direction to follow right-hand rule on
    # July 8, 2026; before this date, scale phi by -1
    phi_scale = 1.0
    timestamp = scan['baseline']['time'][0]
    timestamp = datetime.fromtimestamp(
        timestamp, tz=ZoneInfo("America/New_York"))
    check_day = timestamp.date()
    if check_day < SMI_H5_PHI_FLIP_DATE:
        phi_scale = -1.0

    metadata = {}
    metadata['energy_ev'] = scan['baseline']['energy_energy'][0]
    metadata['sdd_cm'] = scan['baseline']['pil2M_motor_z'][0]/10
    metadata['exposure_time_s'] = scan['config']['pil2M_cam_acquire_time'][0]
    metadata['pixel_size_um'] = 172  # pilatus2m
    metadata['bpm'] = scan['primary']['xbpm3_sumX'][:]
    metadata['sample_phi_deg'] = scan['primary']['stage_phi'][:]
    metadata['epoch_time'] = scan['primary']['time']

    seq_num = scan['primary']['seq_num'][:]

    images = []
    for i in range(num_images):
        temp_metadata = dict(metadata)
        metadata_index = np.where(seq_num == i + 1)[0]
        temp_metadata['bpm'] = metadata['bpm'][metadata_index][0]
        temp_metadata['sample_phi_deg'] = np.round(phi_scale*metadata['sample_phi_deg'][metadata_index], 2)[0]
        temp_metadata['epoch_time'] = metadata['epoch_time'][metadata_index][0]
        images.append(
            (image_stack[i].astype(np.float64), filepath, temp_metadata)
        )

    return images
