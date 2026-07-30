"""
Tools for the loader module.
"""
import os
import re
import warnings

from ..data.metadata import (
    METADATA_KEYWORDS,
    correct_metadata_dtype
)


def clean_filepath(filepath):
    """
    Normalize a file path to an absolute OS-formatted path.

    Parameters
    ----------
    filepath : str | os.PathLike
        File path to normalize.

    Returns
    -------
    cleaned_filepath : str
        Absolute file path formatted for the current operating system.
    """
    filepath = os.path.abspath(filepath)
    if len(filepath) > 256:
        warnings.warn(
            "Caution: your file path is quite long"
            f"({len(filepath)} characters) and might result in"
            "this loader failing.")
    return filepath


def filter_filenames_by_filetype(filenames, filetype):
    """
    Filter filenames by a supported file type.

    Parameters
    ----------
    filenames : list[str]
        Filenames to filter.
    filetype : str | None
        File type filter to apply. 
        Supported values are:
            'tiff' or 'tif'
            'nist-bin' or 'nist_bin' 
            'smi-h5' or 'smi_h5'
        If None, filenames is returned unchanged.

    Returns
    -------
    filtered_filenames : list[str]
        Filenames that match the requested file type.
    """
    if filetype is None:
        return filenames

    if filetype.lower() in ['tiff', 'tif']:
        filenames = [x for x in filenames if 'tif' == x.split('.')[-1]
                     or 'tiff' == x.split('.')[-1]]
    elif filetype.lower() in ['nist_bin', 'nist-bin']:
        filenames = [x for x in filenames if 'bin' == x.split('.')[-1]]
    elif filetype.lower() in ['smi-h5', 'smi_h5']:
        filenames = [x for x in filenames if 'h5' == x.split('.')[-1]]

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
    filename : str
        Filename from which to extract metadata values.
    metadata_pattern : str
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
    metadata_scales : dict | None
        If any extracted metadata was provided in incorrect units, a
        scaling value can be provided to perform unit conversions. The
        argument should be a dictionary where each key is a metadata or
        user_params key and each value is the factor used to scale the
        extracted value. This only applies to values extracted from
        metadata_pattern.

    Returns
    -------
    metadata : dict
        Updated metadata dictionary containing values extracted from the
        filename pattern.
    user_params : dict
        Updated user parameter dictionary containing extracted values for
        keys outside the accepted metadata set.

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
    Generate a data name from a pattern, metadata, and user parameters.

    Parameters
    ----------
    data_name_pattern : str | None
        Naming pattern containing placeholders in braces.
    metadata : dict, optional
        Metadata values available for placeholder substitution.
    user_params : dict, optional
        User-defined values available for placeholder substitution.

    Returns
    -------
    new_name : str | None
        Generated data name. If no pattern is provided, the filename
        metadata value is used when available.
    """

    # handle any None metadata or user_params that may get passed
    if metadata is None:
        metadata = {}
    if user_params is None:
        user_params = {}

    if data_name_pattern is not None:
        new_name = data_name_pattern
        metadata_not_found = []
        while '{' in new_name and '}' in new_name:
            start = new_name.find('{')
            stop = new_name.find('}')
            key = new_name[start+1:stop]
            if key in metadata.keys():
                value = metadata[key]
                if isinstance(value, float):
                    value = round(value, 2)
            elif key in user_params.keys():
                value = user_params[key]
                if isinstance(value, float):
                    value = round(value, 2)
            else:
                value = key
                metadata_not_found.append(key)
            old_str = "{"+key+"}"
            new_str = str(value)
            new_name = new_name.replace(old_str, new_str)
        for key in metadata_not_found:
            new_name = new_name.replace(key, "{"+key+"}")
    elif 'filename' in metadata.keys():
        new_name = metadata['filename']
    else:
        new_name = data_name_pattern

    return new_name
