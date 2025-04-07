"""
This module contains three classes:

DataQyQxz : Class for storing a single scattering image, relevant
    scattering metadata, and any user-defined parameters.
Dataset : Class for managing a series of DataQyQxz objects. It is
    expected that this is a single sample angle sweep in CD-SAXS.
IntegratedDataset : Class for managing integrated DataQyQxz images in
    a Dataset instance in the form of I vs. q_xz.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from numpy.typing import NDArray

import cdsaxs.calculators as calculators
from cdsaxs.sample import Sample

METADATA_KEYWORDS = [
        "sample_phi_deg",
        "sample_chi_deg",

        "energy_ev",
        "wavelength_nm",
        "exposure_time_s",
        "sdd_cm",
        "pixel_size_um",

        "scaling_factor",
        "I0",
        "beam_current",
        "center_px",
    ]


class DataQyQxz():
    """
    This class contains the Qy-Qxz image, relevant metadata, and any
    additional user parameters.

    Attributes
    ----------
    imgdata : NDArray
        Scattering image as a 2D NumPy array. The first dimension should
        correspond to Qy and the second dimension should correspond to
        Qxz (with respect to the detector coordinates).
    qxzs : NDArray
        One-dimensional NumPy array of the scattering vector along the
        xz direction (with respect to detector coordinates).
    qys : NDArray
        One-dimensional NumPy array of the scattering vector along the
        y direction (with respect to detector coordinates).
    metadata : dict
        Contains any relevant scattering metadata. These are key : value
        pairs where the key must be in the list below and the value is
        formatted depending on requirements of the parameter.
        Note: Only one of wavelength and energy need to be specified.
            The other will be calculated upon entry.
    sample : Sample
        Instance of the Sample class that details the sample measured
        when collecting this dataset.
    user_params : dict
        Contains additional user-provided parameters. These may be
        relevant to the user and are shown in the data table of the GUI
        after the required metadata, but are not used for processing
        the data within the GUI and standard workflows. They key can
        be of any format/type desired by the user.

    TODO : add brief definitions to these keywords
    Metadata Keywords
    -----------------
    sample_phi_deg : sample rotation angle about positive y axis
    sample_chi_deg : rotation about the z-axis (beam direction)

    energy_ev : source energy, eV
    wavelength_nm : source wavelength, nanometers
    exposure_time_s : count time in seconds
    sdd_cm : sample to detector distance in cm
    pixel_size_um

    scaling_factor : data scaling factor, defaults to 1
    I0
    beam_current
    center_px : [y, xz] beam center pixel location in y and xz
    """

    _current_rotation = 0

    def __init__(
        self,
        imgdata: NDArray[np.floating],
        qys: NDArray[np.floating],
        qxzs: NDArray[np.floating],
        metadata: dict,
        sample: Sample = None,
        user_params: dict = None,
    ):
        """Create an instance of DataQyQxz"""

        if imgdata.shape[0] != len(qys) or imgdata.shape[1] != len(qxzs):
            raise ValueError(
                "Your image and scattering vector dimensions"
                f"don't match up. Your image is of shape {imgdata.shape}. The"
                "first dimension corresponds to qy and the second dimension"
                "corresponds to qxz. Your qy and qxz scattering vectors are of"
                f"length {len(qys)} and {len(qxzs)}, respectively."
            )

        self.imgdata = imgdata
        self.qxzs = qxzs
        self.qys = qys

        # TODO : implement checks for missing critical metadata
        self._check_metadata(metadata)
        if 'scaling_factor' not in metadata.keys():
            metadata['scaling_factor'] = 1
        metadata = self._metadata_wavelength_energy_calc(metadata=metadata)
        self.metadata = metadata

        self.user_params = user_params if user_params is not None else {}
        self.sample = sample if sample is not None else Sample({})

    def rotate_image(self, degrees, direction='ccw'):
        """
        Rotate the scattering image by a specified number of degrees
        in the direction specified.

        The current image rotation with respect to the original image
        is always saved and can be recalled using `reset_rotations`.

        The beam center location is tracked through the rotation and
        the scattering vectors are updated accordingly. Because the
        beam center position is maintained on the image, the vectors
        do not need to be recalculated. In some cases, such as a
        rotation of 180 degrees, the sign from positive to negative or
        negative to positive may change to maintain coordinate
        conventions.
        TODO: this will also be affected by detector_x; correct this

        TODO: unclear how this works if beam center is off detector

        Parameters
        ----------
        degrees : float
            Number of degrees to rotate the image, should be in
            increments of 90 degrees. A negative value will reverse the
            rotation direction specified by direction, use caution.
        direction : str, optional
            Rotation direction. Default is 'ccw' which indicates a
            counterclockwise rotation. Set as 'cw' to indicate a
            clockwise rotation.
        """
        # current parameters
        imgdata = np.copy(self.imgdata)
        qxzs = np.copy(self.qxzs)
        qys = np.copy(self.qys)
        center_px = self.metadata['center_px'].copy()

        if degrees < 0:
            warnings.warn(
                "You have provided a negative value for degrees of rotation. "
                "This will reverse the direction specified in the 'direction' "
                "argument. For example, a -90 degree rotation "
                "counter-clockwise is the same as a 90 degree clockwise "
                "rotation. Did you intend this?"
            )
        # determine number of 90 degree rotations clockwise
        k = int(degrees/90) % 4
        if direction == 'cw':
            k *= -1

        if k != 0:
            self.imgdata = np.rot90(imgdata, k=k, axes=(0, 1))

        if k == 1 or k == -3:
            self.qxzs = qys
            self.qys = -1*np.flip(qxzs)
            self.metadata['center_px'] = [
                len(qxzs)-center_px[1]-1, center_px[0]]

        if k == 2 or k == -2:
            self.qxzs = -1*np.flip(qxzs)
            self.qys = -1*np.flip(qys)
            self.metadata['center_px'] = [
                len(qys)-center_px[0]-1, len(qxzs)-center_px[1]-1]

        if k == 3 or k == -1:
            self.qxzs = -1*np.flip(qys)
            self.qys = qxzs
            self.metadata['center_px'] = [
                center_px[1], len(qys)-center_px[0]-1]

        self._current_rotation += k

    def reset_rotations(self):
        """
        Return the image to its original orientation removing any
        rotations that have been done. The beam center will be tracked
        through this rotation and scattering vectors updated.
        """
        if self._current_rotation != 0:
            degrees = -90*self._current_rotation
            while degrees < 0:
                degrees += 360
            self.rotate_image(degrees)

    def recalculate_q(self):
        """
        Recalculate scattering vectors qys and qxzs.

        This should be performed after any changes to the beam center
        position or detector position.
        """
        # TODO: implement when working on diffraction.py
        pass

    def update_metadata(self, metadata: dict, overwrite: bool = True):
        """
        Add accepted metadata to the class instance. Existing metadata
        parameters can be updated by keeping the overwrite argument
        to True.

        Parameters
        ----------
        metadata : dict
            Key : value pairs of accepted metadata (key) and their
            values. See class docstring for list of accepted keywords.
        overwrite : bool
            If set to True, any metadata provided to this method will
            overwrite the existing value in the instance if it already
            exists in self.metadata.
            Default value is True.
        """
        if self._check_metadata(metadata):
            for key, value in metadata.items():
                if key in self.metadata.keys() and not overwrite:
                    pass
                else:
                    self.metadata[key] = value

    def remove_metadata(self, metadata_keys: list):
        """
        Remove accepted metadata from this instance of the class.

        Parameters
        ----------
        metadata_keys : list
            List of metadata to remove from this class instance.
        """
        if self._check_metadata({key: 0 for key in metadata_keys}):
            self.metadata = {
                key: value for key, value in self.metadata
                if key not in metadata_keys
                }

    def _check_metadata(self, metadata):
        """
        Check if a metadata dictionary contains any unaccepted metadata
        keywords.

        Check if both wavelength and energy are being specified.
        """
        unaccepted_keywords = [
            x for x in metadata.keys() if x not in METADATA_KEYWORDS
        ]
        if len(unaccepted_keywords) > 0:
            raise ValueError(
                "The following metadata keywords are not accepted:\n" +
                f"{unaccepted_keywords}\n" +
                "The following are accepted metadata keywords:\n" +
                f"{METADATA_KEYWORDS}"
            )

        if "energy_ev" in metadata.keys() and "wavelength_nm" in metadata.keys():
            raise ValueError(
                "You have specified both the source energy and wavelength. "
                "Only one of these can be specified and the other is "
                "calculated. To avoid over-specifying or conflicting values, "
                "please only use one of these values. "
            )
        return True

    def _metadata_wavelength_energy_calc(self, metadata: dict = None):
        """
        Calculate missing wavelength or energy metadata from the other
        provided. For example, if wavelength was provided in metadata,
        the energy will be calculated and added to meatadata.

        This method will perform the calculation for the metadata
        attribute of this instance, unless the metadata argument is provided.
        The update metadata dictionary will be returned.
        """

        if metadata is None:
            metadata = self.metadata

        if 'wavelength_nm' in metadata.keys():
            metadata['energy_ev'] = calculators.wavelength_to_energy(
                metadata['wavelength_nm'])
        elif 'energy_ev' in self.metadata.keys():
            metadata['wavelength_nm'] = calculators.energy_to_wavelength(
                metadata['energy_ev'])
        else:
            raise KeyError(
                "There is no source wavelength or energy information in "
                "the metadata."
            )
        return metadata

    def update_user_params(self, params: dict, overwrite: bool = True):
        """
        Add key: value pairs to the user params of this class instance.
        Existing parameters can be updated by keeping the overwrite
        argument as True.

        Parameters
        ----------
        params : dict
            Key : value pairs of user-specified parameters for this
            data instance.
        overwrite : bool
            If set to True, any parameters provided to this method will
            overwrite the existing value in this instance if it already
            exists in self.uer_params.
            Default value is True.
        """
        for key, value in params.items():
            if key in self.user_params.keys() and not overwrite:
                pass
            else:
                self.user_params[key] = value

    def remove_user_params(self, param_keys: list):
        """
        Remove the identified parameters from user params of this
        class instance.

        Parameters
        ----------
        param_keys : list
            List of parameters to remove from user_params of this class
            instance.
        """
        self.user_params = {
            key: value for key, value in self.user_params
            if key not in param_keys
            }

    def define_sample(self, sample: Sample):
        """
        Define the measured sample with an instance of the Sample class.
        """
        # TODO: implement required sample checks
        self.sample = sample


class Dataset():
    """
    This class manages at least one instance of DataQyQxz as a dataset.

    Attributes
    ----------
    data_folder : absolute filepath to data directory
    name : default is data_folder but can be user-specified
    datas : dictionary of filename keys leading to DataQyQxz objects,
        one for each scattering file
    """

    def __init__(
        self,
        data_folder: str,
        name: str = None
    ):
        self.data_folder = os.path.abspath(data_folder)

        if name is not None:
            self.name = name
        else:
            self.name = self.data_folder

        self.datas = {}

    def add_data(self, filename: str, data: DataQyQxz):
        """Add a single DataQyQxz instance to the dataset."""
        if filename in self.datas.keys():
            raise ValueError(
                f"Data for {filename} is already part of this dataset")
        self.datas[filename] = data

    def remove_data(self, filename):
        """Removes a single DataQyQxz instance from the dataset."""
        try:
            del self.datas[filename]
        except KeyError:
            warnings.warn(f"Could not delete {filename} data as it was not "
                          "part of the dataset.")

    def update_metadata_for_all(self, metadata: dict, overwrite: bool = True):
        """
        Add accepted metadata to all DataQyQxz stored in this Dataset.
        Existing metadata parameters can be updated by keeping the
        overwrite argument to True.

        Parameters
        ----------
        metadata : dict
            Key : value pairs of accepted metadata (key) and their
            values. See class docstring for list of accepted keywords.
        overwrite : bool
            If set to True, any metadata provided to this method will
            overwrite the existing value in the instance if it already
            exists in self.metadata.
            Default value is True.
        """

        for data in self.datas.values():
            data.update_metadata(metadata=metadata, overwrite=overwrite)

    def remove_metadata_from_all(self, metadata_keys: list):
        """
        Remove accepted metadata from all DataQyQxz stored in this Dataset.

        Parameters
        ----------
        metadata_keys : list
            List of metadata to remove from this class instance.
        """

        for data in self.datas.values():
            data.remove_metadata(metadata_keys=metadata_keys)

    def update_user_params_for_all(self, params: dict, overwrite: bool = True):
        """
        Add key: value pairs to the user params for all DataQyQxz.
        Existing parameters can be updated by keeping the overwrite
        argument as True.

        Parameters
        ----------
        params : dict
            Key : value pairs of user-specified parameters for this
            data instance.
        overwrite : bool
            If set to True, any parameters provided to this method will
            overwrite the existing value in this instance if it already
            exists in self.uer_params.
            Default value is True.
        """

        for data in self.datas.values():
            data.update_user_params(params=params, overwrite=overwrite)

    def remove_user_params_from_all(self, param_keys: list):
        """
        Remove the identified parameters from user params for all
        DataQyQxz of this dataset.

        Parameters
        ----------
        param_keys : list
            List of parameters to remove from user_params of this class
            instance.
        """

        for data in self.datas.values():
            data.remove_user_params(param_keys=param_keys)


class IntegratedDataset():
    """
    This class manages a dataset of integrated DataQyQxz images in the
    form of I vs. Qxz.

    Attributes
    ----------
    dataset : Dataset
    integrated_data : dict
    integration_metadata : dict
    """

    def __init__(self, dataset: Dataset,
                 integrated_data: dict,
                 integration_metadata: dict):

        self.dataset = dataset
        self.integrated_data = integrated_data
        self.integration_metadata = integration_metadata
