from __future__ import annotations
import os
import warnings

import numpy as np
from tqdm import tqdm

from ..data.data2d import Data2D
from ..data.dataset import Dataset
from ..data.metadata import (
    METADATA_KEYWORDS,
    check_metadata,
    correct_metadata_dtype
)
from . import _loader_tools as loader_tools
from .detectors import read_pilatus
from .filetypes import (
    read_tiff,
    read_nist_bin,
    read_smi_h5
)


def filter_filenames(
        directory_path, filter_substrings=None, file_extension=None):
    """
    Filter the files found in the folder at the provided directory path.

    The filters should be a list of substrings to either ensure are
    included or excluded from the filename.

    The most outer level at the list are all substrings that are joined
    with an 'OR', meaning that only one of those keywords needs to
    be present in the filename. If there are a set of keywords that
    should be joined with 'AND', meaning that all of the keywords need
    to be present in the filename, they should be found in a nested list
    at any one or more of the outer list positions. Finally, if there
    are keywords to exclude at either of these levels, they can be
    nested in a tuple with the keyword "NOT".

    For example, if filter_substrings was provided as:
    [['red', 'apple', ('NOT', 'bad')], 'grape', ['orange', ('NOT', 'good')]]

    then the files that would be accepted into the filtered list include
    those with the word grape, those with both red and apple but not bad,
    and those with orange but not good. The list would also include
    any files that include any combinations of those three filters.

    Be careful with the OR level, as 'red_apple_bad_grape.txt' would
    make the cut in this case, even though it would have failed
    the first filter. If the file has to pass both checks, then an
    alternative filter_substrings could be:
    [['red', 'apple', ('NOT', 'bad')],
     ['grape', ('NOT', 'apple'),
     ['orange', ('NOT', 'good')]]

    In this case 'red_apple_bad_grape.txt' would not have passed, but
    'red_apple_good_grape.txt' would have passed.

    This is meant to only provide simple functionality and if a more
    complicated filter is required, then the filenames should be
    filtered by the user and provided directly to the loader.

    Parameters
    ----------
    directory_path : str
        Path to the directory where the files are located.
    filter_substrings : list, optional
        List of substring filters for the files. If not provided,
        this function will return a list of all filenames in the
        directory.
    file_extension : str, optional
        A file extension can be provided as an additional filter
        on the files. A file extension could also be provided in the
        filter_substrings keyword.

    Returns
    -------
    filenames : list[str]
        Filenames in directory_path that satisfy the requested filters.

    """
    directory_path = loader_tools.clean_filepath(directory_path)
    if file_extension is None:
        file_extension = "."
    filenames = [x for x in os.listdir(directory_path) if file_extension in x]

    if filter_substrings is not None:
        or_filtered = []

        for or_item in filter_substrings:
            if type(or_item) is str:
                or_filtered.extend([x for x in filenames if or_item in x])
            elif type(or_item) is list:
                and_filtered = filenames.copy()
                for and_item in or_item:
                    if type(and_item) is str:
                        and_filtered = [
                            x for x in and_filtered if and_item in x]
                    elif type(and_item) is tuple:
                        and_filtered = [
                            x for x in and_filtered if and_item[1] not in x]
                or_filtered.extend(and_filtered)

        filenames = list(set(or_filtered))

    return filenames


def LoadData(
    filepath,
    metadata=None,
    user_params=None,
    name=None,
    filetype=None,
    detector_type=None,
):
    """
    Create an instance of Data2D from a single data file.

    Parameters
    ----------
    filepath : str, path
        Path to the file to be loaded. The loader will try and determine
        the proper reader from the file extension. This can be
        overwritten by providing the keyword argument 'filetype'.
    metadata : dict, optional
        Metadata key-value pairs to add to the loaded 2D data object.
        See metadata.py for the full list of accepted metadata keys.
    user_params : dict, optional
        User-defined key-value pairs that store additional information
        outside the package's standard metadata set.
    name : str, optional
        Name for the two-dimensional data instance.
    filetype : str, optional
        Specify the filetype so that the proper reader is used. If not
        provided, the loader will try to determine the file type based
        on the extension. This can result in unexpected behavior.
        Currently, the accepted filetypes are:
            'tiff' or 'tif'
            'nist-bin'
            'smi-h5'
    detector_type : str
        Specify the type of detector used to collect the image. This is
        helpful if you know there is metadata stored in the file's
        header (or other location in the file depending on the type).
        Currently, the accepted detector types are:
            'Pilatus'

    Returns
    -------
    data : Data2D | list[Data2D]
        Loaded 2D data object. For SMI H5 input, a list of Data2D
        objects is returned, one per image in the scan.
    """

    # clean the filepath and try to determine filetype if not provided
    filepath = loader_tools.clean_filepath(filepath)
    if filetype is None:
        extension = os.path.basename(filepath).split(".")[-1]
        if extension == 'tif' or extension == 'tiff':
            filetype = 'tiff'
        elif extension == 'bin':
            filetype = 'nist-bin'
        else:
            raise ValueError(
                "Did not recognize the filtype extension:"
                f"{os.path.basename(filepath)}."
            )

    if metadata is None:
        metadata = {}
    else:
        check_metadata(metadata=metadata)

    if user_params is None:
        user_params = {}

    # try to use the right loader based on filetype
    if filetype.lower() in ['tiff', 'tif']:
        if detector_type is not None and detector_type.lower() in ["pilatus"]:
            image, data_filepath, metadata_add = read_pilatus(filepath)
            for key, value in metadata_add.items():
                if key in metadata.keys():
                    warnings.warn(
                        f"Metadata for {key} was provided by the user or"
                        " already extracted from reading the file."
                        " I will not overwrite the existing metadata with"
                        " the value extracted by knowing the detector type."
                    )
                else:
                    metadata[key] = value
            # handle negative values between detector panels in the images as nan
            image[image < 0] = np.nan
        else:
            image, data_filepath, _ = read_tiff(filepath)

    elif filetype.lower() in ['nist-bin', 'nist_bin']:
        image, data_filepath, metadata_add = read_nist_bin(filepath=filepath)
        for key, value in metadata_add.items():
            if key in metadata.keys():
                warnings.warn(
                    f"Metadata for {key} was provided by the user and"
                    "also extracted from the data files. I will not"
                    "overwrite the information provided by the user"
                    "but please make sure this is correct."
                )
            else:
                metadata[key] = value
        # handle negative values between detector panels in the images as nan
        image[image < 0] = np.nan

    elif filetype.lower() in ['smi_h5', 'smi-h5']:
        image_stack = read_smi_h5(filepath=filepath)
        # handle negative values between detector panels in the images as nan
        for image, _, _ in image_stack:
            image[image < 0] = np.nan

    else:
        raise ValueError(
            f"Did not recognize the filetype {filetype}."
        )

    if filetype.lower() not in ['smi_h5', 'smi-h5']:
        metadata['data_directory'] = os.path.dirname(data_filepath)
        metadata['filename'] = os.path.basename(data_filepath)

        if name is not None:
            metadata['name'] = name

        return Data2D(
            image, **metadata, **user_params)
    

    else:
        data2d_list = []
        for image, data_filepath, metadata_add in image_stack:
            temp_metadata = dict(metadata)
            temp_user_params = dict(user_params)
            
            for key, value in metadata_add.items():
                if key in METADATA_KEYWORDS:
                    if key in temp_metadata.keys():
                        warnings.warn(
                            f"Metadata for {key} was provided by the user and"
                            "also extracted from the data files. I will not"
                            "overwrite the information provided by the user"
                            "but please make sure this is correct."
                        )
                    else:
                        temp_metadata[key] = value
                else:
                    if key in temp_user_params.keys():
                        warnings.warn(
                            f"User parameter for {key} was provided by the user and"
                            "also extracted from the data files. I will not"
                            "overwrite the information provided by the user"
                            "but please make sure this is correct."
                        )
                    else:
                        temp_user_params[key] = value

            temp_metadata['data_directory'] = os.path.dirname(data_filepath)
            temp_metadata['filename'] = os.path.basename(data_filepath)

            if name is not None:
                new_name = loader_tools.generate_data_name_from_pattern(
                    name, temp_metadata, temp_user_params)
                temp_metadata['name'] = new_name
            
            data2d_list.append(Data2D(image, **temp_metadata, **temp_user_params))
        
        return data2d_list


def LoadDataset(
    dataset_name,
    directory_path,
    filenames=None,
    metadata_pattern=None,
    metadata_scales=None,
    data_name_pattern=None,
    metadata=None,
    user_params=None,
    verbose=True,
    filetype=None, 
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
        User-specified name given to the dataset.
    directory_path : str, path
        Path to the directory containing the data to load.
    filenames : list, optional
        List of specific files to load into the dataset from the
        directory at directory_path. Keep in mind that unless this is
        specified, all files in the directory that align with the
        filetype (if specified) will be loaded into the dataset.
        A set of substring filters can be applied using the
        'filter_filenames' function. The list of filenames returned
        can be used as the input to this keyword argument.
        Default is None.
    metadata_pattern : str, optional
        Extract metadata from information stored in the filenames.
        A pattern for the filenames can be provided where the
        keywords for either metadata or user parameters should be
        enclosed in {}. For example, if two images had filenames of:
            sample1_phi0_sdd_500_run001.tif
            sample1_phi-1_sdd_500_run002.tif
        The following pattern could be provided to extract metadata
        parameters of 'sample_phi_deg' and 'sdd_cm' as well as user
        parameter 'run' for each file:
            sample1_phi{sample_phi_deg}_sdd_{sdd_cm}_run{run}.tif
        NOTE: conflicts can arise if both this pattern and the
        CSV metadata loader are used. Metadata parameters specified in
        both places can result in one source overwriting the other.
    metadata_scales : dict, optional
        If any of the metadata was provided in incorrect units, a
        scaling value can be provided to perform unit conversions. The
        argument should be provided as a dictionary where the key
        is the metadata keyword or user_params keyword, and the
        value is the amount by which to scale or multiply the
        parameter's current value.
        NOTE: this will only apply to values extracted from the metadata
        pattern of the filenames.
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
    metadata : dict, optional
        Metadata key-value pairs to add to every loaded 2D data object.
        See metadata.py for the full list of accepted metadata keys.
    user_params : dict, optional
        User-defined key-value pairs that store additional information
        outside the package's standard metadata set.
    verbose : bool, optional
        If set to True, a progress bar will be displayed during the
        loading process. Set to False to turn off this feature.
        Default value is True
    filetype : str, optional
        Specify the filetype so that the proper reader is used.
        Currently, the accepted filetypes are:
            'tiff' or 'tif'
            'nist-bin'
            'smi-h5'
    detector_type : str
        Specify the type of detector used to collect the image. This is
        helpful if you know there is metadata stored in the file's
        header (or other location in the file depending on the type).
        Currently, the accepted detector types are:
            'Pilatus'

    Returns
    -------
    dataset : Dataset
        Dataset containing the loaded 2D data objects.
        """

    dataset = Dataset(name=dataset_name)

    directory_path = loader_tools.clean_filepath(directory_path)

    if filenames is None:
        filenames = [x for x in os.listdir(directory_path)]
    if not isinstance(filenames, list):
        filenames = [filenames]
    filenames = loader_tools.filter_filenames_by_filetype(filenames, filetype)

    if verbose:
        pbar = tqdm(range(len(filenames)), desc="Loading files: ",
                    position=0, leave=True)

    with warnings.catch_warnings(record=True) as warnings_output:
        for filename in filenames:
            filepath = os.path.join(directory_path, filename)
            if metadata is not None:
                metadata_i = {key: value for key, value in metadata.items()}
            else:
                metadata_i = {}
            if user_params is not None:
                user_params_i = {key: value for key, value in user_params.items()}
            else:
                user_params_i = {}

            # extract information from the metadata filename pattern
            metadata_extract, user_params_extract = loader_tools.extract_metadata_from_pattern(
                {}, {}, filename, metadata_pattern, metadata_scales
            )
            for key, value in metadata_extract.items():
                if key in metadata_i.keys():
                    warnings.warn(
                        f"Metadata for {key} was provided by the user but "
                        "also extracted from the filename using metadata pattern."
                        " I will use the data provided by the user rather than "
                        "the value extracted from the name but please ensure "
                        "this is correct."
                    )
                else:
                    metadata_i[key] = value
            for key, value in user_params_extract.items():
                if key in user_params_i.keys():
                    warnings.warn(
                        f"User params for {key} was provided by the user but "
                        "also extracted from the filename using metadata pattern."
                        " I will use the data provided by the user rather than "
                        "the value extracted from the name but please ensure "
                        "this is correct."
                    )
                else:
                    user_params_i[key] = value
            # print("METADATA", metadata_i, user_params_i)
            # generate the name for the two-dimensional data
            new_name = loader_tools.generate_data_name_from_pattern(
                data_name_pattern, metadata_i, user_params_i)

            data = LoadData(
                filepath=filepath,
                metadata=metadata_i,
                user_params=user_params_i,
                filetype=filetype,
                detector_type=detector_type,
                name=new_name
            )
            if isinstance(data, list):
                for d in data:
                    dataset.add_data(d)
            else:
                dataset.add_data(data)

            if verbose:
                pbar.update(1)

    for warning in warnings_output:
        print(warning.message)

    if verbose:
        pbar.close()
    print(
        f"Created dataset from data directory: {directory_path}")
    return dataset


def LoadDataset_MetadataCSV(
    dataset_name,
    metadata_csv_filepath,
    verbose=True,
    filetype=None,
    detector_type=None,
    data_name_pattern=None
):
    """
    General data loader to create a dataset from a CD-SAXS angle scan
    and load relevant metadata from a CSV file.
    TODO : currently only implemented for tiff files

    Parameters
    ----------
    dataset_name : str
        User-specified name given to the dataset.
    metadata_csv_filepath : str, path
        Filepath to the CSV file used to define metadata for each image
        loaded. Column headers should specify either an accepted
        metadata key or a user-defined parameter name. Values for
        unrecognized keys are stored in user_params. The CSV file should
        be located in the same directory as the data files.
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
        Default is True.
    filetype : str, optional
        Specify the filetype so that the proper reader is used.
        Currently, the accepted filetypes are:
            'tiff' or 'tif'
            'nist-bin'
    detector_type : str, optional
        Specify the type of detector used to collect the image. This is
        helpful if you know there is metadata stored in the file's
        header (or other location in the file depending on the type).
        Currently, the accepted detector types are:
            'Pilatus'
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

    Returns
    -------
    dataset : Dataset
        Dataset containing the loaded 2D data objects defined by the CSV
        metadata table.
    """

    dataset = Dataset(name=dataset_name)

    # load the csv metadata file
    metadata_csv_filepath = loader_tools.clean_filepath(filepath=metadata_csv_filepath)
    csv_data = np.loadtxt(metadata_csv_filepath, dtype='str', delimiter=',')
    csv_header = csv_data[0, :]
    csv_data = csv_data[1:, :]

    directory_path = os.path.dirname(metadata_csv_filepath)

    if verbose:
        pbar = tqdm(range(csv_data.shape[0]), desc="Loading files: ",
                    position=0, leave=True)

    with warnings.catch_warnings(record=True) as warnings_output:
        for i, row in enumerate(csv_data):
            metadata = {}
            user_params = {}
            filepath = None
            for ii, value in enumerate(row):
                if csv_header[ii] in METADATA_KEYWORDS:
                    if csv_header[ii] == 'filename':
                        filepath = os.path.join(directory_path, value)
                    else:
                        metadata[str(csv_header[ii])]\
                            = correct_metadata_dtype(csv_header[ii], value)
                else:
                    user_params[str(csv_header[ii])] = value

            new_name = loader_tools.generate_data_name_from_pattern(
                    data_name_pattern, metadata, user_params)

            data = LoadData(
                filepath=filepath,
                metadata=metadata,
                user_params=user_params,
                filetype=filetype,
                detector_type=detector_type,
                name=new_name
            )

            dataset.add_data(data)

            if verbose:
                pbar.update(1)

    for warning in warnings_output:
        print(warning.message)

    if verbose:
        pbar.close()
    print(
        f"Created dataset from data directory: {directory_path}\n"
        f"and csv file {os.path.basename(metadata_csv_filepath)}.")

    return dataset
