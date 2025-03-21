"""
This module contains the class for 2D image data and relevant metadata:
DataQxzQy. An instance of this class should be built for every
scattering image.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class DataQxzQy():
    """
    This class contains the Qxz-Qy image, relevant metadata, and any
    additional user parameters.

    Attributes
    ----------
    imgdata : NDArray
        Scattering image as a 2D NumPy array. The first dimension should
        correspond to Qxz and the second dimension should correspond to
        Qy (with respect to the detector coordinates).
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
    sample_phi_deg
    energy_ev
    wavelength_nm
    scaling_factor : data scaling factor, defaults to 1
    detector_theta
    detector_x
    exposure_time_s : count time in seconds
    sdd_cm : sample to detector distance in cm
    sample_chi_deg : rotation about the z-axis (beam directory)
    div_photodiode
    center_px : [xz, y] beam center pixel location in xz and y
    """

    accepted_keywords = [
        'sample_phi_deg', 'energy_ev', 'wavelength_nm', 'scaling_factor',
        'detector_theta', 'detector_x', 'exposure_time_s', 'sdd_cm',
        'sample_chi_deg', 'div_photodiode', 'center_px',
    ]

    _current_rotation = 0

    def __init__(
        self,
        imgdata: NDArray[np.floating],
        qxzs: NDArray[np.floating],
        qys: NDArray[np.floating],
        metadata: dict,
        params: dict = None,
    ):
        """Create an instance of DataQxzQy"""

        self.imgdata = imgdata
        self.qxzs = qxzs
        self.qys = qys

        unaccepted_keywords = [
            x for x in metadata.keys() if x not in self.accepted_keywords
        ]
        if len(unaccepted_keywords) > 0:
            raise ValueError(
                "The following metadata keywords are not accepted:\n" +
                f"{unaccepted_keywords}\n" +
                "The following are accepted metadata keywords:\n"+
                f"{self.accepted_keywords}"
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

        Parameters
        ----------
        degrees : float
            Number of degrees to rotate the image, should be in
            increments of 90 degrees.
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

        # determine number of 90 degree rotations
        k = int(degrees/90) % 4
        if direction == 'cw':
            k *= -1

        if k > 0:
            self.imgdata = np.rot90(imgdata, k=k, axes=(0, 1))

        if k == 1:
            self.qxzs = qys
            self.qys = -1*np.flip(qxzs)
            self.metadata['center_px'] = [len(qxzs)-center_px[1]-1, center_px[0]]

        if k == 2:
            self.qxzs = -1*np.flip(qxzs)
            self.qys = -1*np.flip(qys)
            self.metadata['center_px'] = [len(qys)-center_px[0]-1, len(qxzs)-center_px[1]-1]

        if k == 3:
            self.qxzs = -1*np.flip(qys)
            self.qys = qxzs
            self.metadata['center_px'] = [center_px[1], len(qys)-center_px[0]-1]

        self._current_rotation += k

    def reset_rotations(self):
        """
        Return the image to its original orientation removing any
        rotations that have been done. The beam center will be tracked
        through this rotation and scattering vectors updated.
        """
        if self._current_rotation > 0:
            self.rotate_image(-90*self._current_rotation)
