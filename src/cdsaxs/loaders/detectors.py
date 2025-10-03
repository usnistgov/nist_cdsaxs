import cdsaxs.loaders._loader_tools as lt
from cdsaxs.loaders.filetypes import read_tiff


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
    NDArray, None
        Image from the tiff file if the filepath is provided.
        Otherwise, this is None
    str, None
        Formatted filepath from which the image was loaded. If filepath
        is not provided, this is None.
    dict
        Metadata dictionary with accepted metadata keywords extracted
        from the tiff file header.
    """
    image, filepath, header = read_tiff(filepath=filepath)
    metadata = lt.pilatus_header_to_metadata(header)

    return image, filepath, metadata
