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
from cdsaxs.metadata import correct_metadata_dtype, METADATA_KEYWORDS


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
    if len(filepath) > 256:
            warnings.warn(
                "Caution: your file path is quite long"
                f"({len(filepath)} characters) and might result in"
                "this loader failing.")
    filepath = os.path.abspath(filepath)

    try:
        image = Image.open(filepath)
        image = np.array(image).astype(np.float64)
        header = {TAGS[key]: image.tag[key] for key in image.tag_v2
                    if key in TAGS.keys()}
    except:
        image = tifffile.imread(filepath).astype(np.float64)
        with tifffile.TiffFile(filepath) as tif:
            header = {tag.name: tag.value
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

    if len(filepath) > 256:
            warnings.warn(
                "Caution: your file path is quite long"
                f"({len(filepath)} characters) and might result in"
                "this loader failing.")
    filepath = os.path.abspath(filepath)

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
    metadata['wavelength_nm'] = (float(sample_meta['Wavelength (nm) ']),
                                       'nm')
    metadata['exposure_time_s'] = (float(sample_meta['LiveTime ']),
                                       's')
    metadata['pixel_size_um'] = (float(sample_meta['Pixel Size ']),
                                       'um')

    return image, filepath, metadata


def LoadData(
    filepath,
    metadata=None,
    user_params=None,
    filetype=None,
    detector_type=None,
    ):
    """
    Create an instance of DataQdyQdx from a single data file.

    Parameters
    ----------
    filepath : str, path
        Path to the file to be loaded. The loader will try and determine
        the proper reader from the file extension. This can be
        overwritten by providing the keyword argument 'filetype'.
    metadata : dict, optional
        Dictionary of metadata keyword, value pairs to be added to
        the metadata attribute of the 2D data instance.
        See metadata.py for full list of accepted keywords.
    user_params : dict, optional
        Dictionary of user specified keyword, value pairs that provide
        additional parameters about the data that are outside the scope
        of the code's standard metadata.
    filetype : str
        Specify the filetype so that the proper reader is used.
        Currently, the accepted filetypes are:
            'tiff'
            'nist-bin'
    detector_type : str
        Specify the type of detector used to collect the image. This is
        helpful if you know there is metadata stored in the file's
        header (or other location in the file depending on the type).
        Currently, the accepted detector types are:
            'Pilatus'
    """

    filepath = os.path.abspath(filepath)



    DataQdyQdx(image, metadata=metadata, user_params=params)


class 

class Load():

    def __init__(self, filepaht):

        """
        Reads image files for the dataset loaders.
        """

def tiff_loader(filepath):
    """
    Load an image and header information fr
    """



class TiffData():
    """Tools for reading and handling TIFF image files."""

    def __init__(self, filepath):
        """
        Read an image and header information from TIFF file.

        Parameters
        ----------
        filepath : filepath to the image file

        Attributes
        ----------
        image : NDArray
            Image extracted from the file.
        header : dict
            Header information extracted from the tiff file.
        """

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


class PilatusData(TiffData):
    """
    Tools for handling TIFF image files from a Pilatus detector.
    """

    def __init__(self, filepath):
        """
        Read an image and header information from TIFF file.

        Parameters
        ----------
        filepath : filepath to the image file

        Attributes
        ----------
        image : NDArray
            Image extracted from the file.
        header : dict
            Header information extracted from the tiff file.
        """
        super().__init__(filepath)

    def extract_exposure_time(self):
        """
        Extract exposure time in seconds from the TIFF file header"
        of a Pilaturs detectr."
        """

        try:
            _, value, units = [
                x for x in self.header['ImageDescription'][0].split('#')
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

    def extract_pixel_size(self):

        "Extract pixel time in um from the TIFF file header."

        try:
            _, value0, units0, _, value1, units1 = [
                x for x in self.header['ImageDescription'][0].split('#')
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
            else:
                return float(value0) * 1e6
        except:
            warnings.warn(
                "Could not extract pixel size from the header; returning None"
            )
            return None


class GeneralDataLoader():

    def __init__(
        self,
        dataset_name,
        directory_path=None,
        metadata_csv_filepath=None,
        metadata_pattern=None,
        metadata_scales=None,
        data_name_pattern=None,
        filter_by_substrings=None,
        filter_by_filenames=None,
        verbose=True,
        detector_type=None
    ):
        """
        General data loader to create a dataset from a CD-SAXS angle scan
        and load relevant metadata.
        TODO : currently only implemented for tiff files

        Note that there are multiple filters that can be applied to the
        dataset but this could result in unexpected behavior. Please refer
        to this documentation to understand which keyword arguments are
        critical to reading your data.

        Parameters
        ----------
        dataset_name : str
        mode : str
            Set the mode for the loader. This determines which keyword
            arguments should be provided; others will be ignored if
            provided anyway. Accepted modes include:
            'csv' : Provide a csv file with metadata for each file to
                be loaded. Accepted keyword arguments include:
                    metadata_csv_filepath
                    metadata_scales
                    verbose
                    detector_type
            'filepaths' : Provide a list of filepaths to the files you
                would like to load into the dataset. Accepted keywrod
                arguments include:
                    metadata_pattern
                    metadata_scales
                    data_name_
            'substring_filter'
        directory_path : str, path, optional
            Set the path to the directory from which the tiff images should
            be loaded. This will be ignored if filepats or
            metadata_csv_filepath are provided as these also provide all
            required information to locate the files.
        filepaths : list[string], optional
            Provide a list of filepaths to the specific tiff images to load.
            This list will be further filtered by the following keyword
            arguments if also provided:
                filter_files_by_substrings
                metadata_csv_filepath (from filename column)
        metadata_csv_filepath : str, path, optional
            The filepath to the csv file used to define the metadata for
            each tiff image loaded. The column headers should specifiy
            the metadata parameter or unique user-specified parameter the
            value should be assigned to. An accepted metadata keyword must
            be used otherwise the program will assign the information to
            the user_params attribute of the data. The csv file should be
            located in the same directory as the data.
            Accepted metadata keywords:
            ---------------------------
                "filename" (required)
                "sample_phi_deg"
                "sample_phi_offset_deg",
                "sample_omega_deg"
                "sample_chi_deg",
                "energy_ev"
                "wavelength_nm",
                "exposure_time_s"
                "sdd_cm"
                "pixel_size_um",
                "scaling_factor"
                "I0"
                "beam_current",
                "data_directory"
                "name"
                "center_px"
                'sample_phi_offset_deg'
        metadata_pattern : str, optional
            Extract metadata from information stored in the filename.
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
        metadata_scales : dict, optional
            If the metadata was provided in incorrect units, a scaling
            value can be provided to perform unit conversions. The
            argument should be provided as a dictionary where the key
            is the metadata keyword or user_params keyword, and the
            value is the amount by which to scale or multiply the
            parameter's current value.
        data_name_pattern : str, optional
            Provide a pattern to create a unique name for each data in
            the dataset. Unless the name is provided in the csv file,
            the default behavior is that the filename is used as the
            data name. Alternatively, a pattern can be provided that
            pulls information from helpful metadata. Keep in mind that
            the name needs to be unique in the dataset.
            An example is:
                "Sample 4, Angle: {sample_phi_deg} deg"
            The data name would be for a sample at a phi rotation angle
            of 20 degrees:
                "Sample 4, Angle: 20 deg"
        filter_files_by_substrings : list[str, list[str]], optional
            Filter which files to read from the specified data directory
            by a set of substrings. If a list of strings (substrings) is
            provided, a filename that includes at least one of those
            substrings will be loaded. If it is required that the
            filenames have all the required substrings present, then
            they should be inclosed in a nested list. For example, if
            the filter_files_by_substrings is set as:
                [["red", "apple"], "orange", "strawberry"]
            Then only files that have orange or strawberry or both (red
            and apple) or any combination of those three will be loaded.
            A filename of "apple_orange.tif" will be loaded,
            "apple.tif" will not, and "red_apple.tif" will be.
            This is only meant to provide simple filtering functionality
            and for more complex filtering, we encourage the user to
            specify the 'filepaths' or 'metadata_csv_filepath' instead.
        verbose : bool, optional
            If set to True, a progress bar will be displayed during the
            loading process. Set to False to turn off this feature.
            Default value is True
        detector_type : str, optional
            Specify the detector type to use built-in loader functions
            specific to the detector. Accepted detector types are:
                'Pilatus'
        """

        dataset = Dataset(name=dataset_name)

         
        
        
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
                    metadata[str(header[ii])] = correct_metadata_dtype(header[ii], value)
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


def GeneralDataLoader_MetadataCSV(
    dataset_name,
    metadata_csv_filepath,
    verbose=True,
    detector_type=None
):
    """
    General data loader to create a dataset from a CD-SAXS angle scan
    and load relevant metadata from a CSV file.
    TODO : currently only implemented for tiff files

    Parameters
    ----------
    dataset_name : str
    metadata_csv_filepath : str, path
        The filepath to the csv file used to define the metadata for
        each tiff image loaded. The column headers should specifiy
        the metadata parameter or unique user-specified parameter the
        value should be assigned to. An accepted metadata keyword must
        be used otherwise the program will assign the information to
        the user_params attribute of the data. The csv file should be
        located in the same directory as the data.
        Accepted metadata keywords:
        ---------------------------
            "filename" (required)
            "sample_phi_deg"
            "sample_phi_offset_deg",
            "sample_omega_deg"
            "sample_chi_deg",
            "energy_ev"
            "wavelength_nm",
            "exposure_time_s"
            "sdd_cm"
            "pixel_size_um",
            "scaling_factor"
            "I0"
            "beam_current",
            "data_directory"
            "name"
            "center_px"
            'sample_phi_offset_deg'
    verbose : bool, optional
        If set to True, a progress bar will be displayed during the
        loading process. Set to False to turn off this feature.
        Default value is True
    detector_type : str, optional
        Specify the detector type to use built-in loader functions
        specific to the detector. Accepted detector types are:
            'Pilatus'
    """

    dataset = Dataset(name=dataset_name)

         
        
        
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
                    metadata[str(header[ii])] = correct_metadata_dtype(header[ii], value)
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
                    metadata[key] = correct_metadata_dtype(key, value)
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
                    metadata[keyword] = correct_metadata_dtype(keyword, value)

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

    print('Made dataset from ' + directory_path)

    return dataset
