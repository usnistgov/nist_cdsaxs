"""
Load different data classes from HDF5 files.
"""
import os

import h5py

from cdsaxs.data.data2d import Data2D
from cdsaxs.loaders.load_data import LoadData


def load_data2d(hdf5_filepath,
                raw_image_directory=None, raw_image_filename=None):

    file = h5py.File(hdf5_filepath, "r")

    if raw_image_filename is None:
        raw_image_filename = file["data2d/metadata"].attrs["filename"]
    if raw_image_directory is None:
        raw_image_directory = os.path.abspath(
            file["data2d/metadata"].attrs["data_directory"])

    raw_image_filepath = os.path.join(raw_image_directory, raw_image_filename)

    data = LoadData(
        raw_image_filepath,
    )

    metadata = dict(file["data2d/metadata"].attrs)
    user_params = dict(file["data2d/user_params"].attrs)
    sample_rotation = dict(file["data2d/sample_rotation"].attrs)

    data.update_metadata(metadata=metadata, overwrite=True, hide_q_warnings=True)
    data.update_user_params(params=user_params, overwrite=True)
    data._set_sample_rotation(**sample_rotation)

    data.name = file["data2d"].attrs["name"]

    data._overwrite_mask(file["data2d/mask"])

    for key, value in file["data2d/q"].items():
        setattr(data, key, value)

    return data

