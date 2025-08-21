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
        _, value, units = [
            x for x in header['ImageDescription'][0].split('#')
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
        _, value0, units0, _, value1, units1 = [
            x for x in header['ImageDescription'][0].split('#')
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


def filter_filenames(
        dir_path, filter_substrings=None, file_extension=None):
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
    dir_path : str
        Path to the directory where the files are located.
    filter_substrings : list, optional
        List of substring filters for the files. If not provided,
        this function will return a list of all filenames in the
        directory.
    file_extension  : str, optional
        A file extension can be provided as an additional filter
        on the files. A file extension could also be provided in the
        filter_substrings keyword.

    """
    dir_path = clean_filepath(dir_path)
    if file_extension is None:
        file_extension = "."
    filenames = [x for x in os.listdir(dir_path) if file_extension in x]

    if filter_substrings is not None:
        or_filtered = []

        for or_item in filter_substrings:
            if type(or_item) is str:
                or_filtered.extend([x for x in filenames if or_item in x])
            elif type(or_item) is list:
                and_filtered = filenames.copy()
                for and_item in or_item:
                    if and_item is str:
                        and_filtered = [
                            x for x in and_filtered if and_item in x]
                    elif and_item is tuple:
                        and_filtered = [
                            x for x in and_filtered if and_item not in x]
                or_filtered.extend(and_filtered)

        filenames = list(set(or_filtered))

    return filenames
