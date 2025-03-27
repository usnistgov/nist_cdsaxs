"""
This module contains two classes:

DataQyQxz : Class for storing a single scattering image, relevant
    scattering metadata, and any user-defined parameters.
Dataset : Class for managing a series of DataQyQxz objects. A new
    Dataset instance is created upon each loading of data by the user
    in the GUI.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from numpy.typing import NDArray

METADATA_KEYWORDS = [
        'sample_phi_deg', 'energy_ev', 'wavelength_nm', 'scaling_factor',
        'detector_theta', 'detector_x', 'exposure_time_s', 'sdd_cm',
        'sample_chi_deg', 'div_photodiode', 'center_px', 'pixel_size_um'
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
    params : dict
        Contains additional user-provided parameters. These may be
        relevant to the user and are shown in the data table of the GUI
        after the required metadata, but are not used for processing
        the data within the GUI and standard workflows. They key can
        be of any format/type desired by the user.

    TODO : add brief definitions to these keywords
    Metadata Keywords
    -----------------
    sample_phi_deg : sample rotation angle about positive y axis
    energy_ev : source energy, eV
    wavelength_nm : source wavelength, nanometers
    scaling_factor : data scaling factor, defaults to 1
    detector_theta
    detector_x
    exposure_time_s : count time in seconds
    sdd_cm : sample to detector distance in cm
    sample_chi_deg : rotation about the z-axis (beam direction)
    div_photodiode
    center_px : [y, xz] beam center pixel location in y and xz
    """

    _current_rotation = 0

    def __init__(
        self,
        imgdata: NDArray[np.floating],
        qys: NDArray[np.floating],
        qxzs: NDArray[np.floating],
        metadata: dict,
        params: dict = None,
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

        # TODO : implement checks for missing critical metadata
        self.metadata = metadata
        if 'scaling_factor' not in self.metadata.keys():
            self.metadata['scaling_factor'] = 1

        self.params = params if params is not None else {}

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

    def update_beamcenter(self, qy, qxz):
        """Udpdate the beam center indices."""
        self.metadata['center_px'] = [qy, qxz]
        self.recalculate_q()

    def recalculate_q(self):
        """
        Recalculate scattering vectors qys and qxzs.

        This should be performed after any changes to the beam center
        position or detector position.
        """
        # TODO: implement when working on diffraction.py
        pass


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
