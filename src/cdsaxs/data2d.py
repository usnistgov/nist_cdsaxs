"""
This module contains the following classes for two-dimensional
scattering images.

Data2D : Generic two-dimensional data class not tied to diffraction.
DataQdyQdx(Data2D) : Child class of Data2D for detector images.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import linregress
from plotly.offline import iplot
from PIL import Image

import cdsaxs.calculators as calculators
from cdsaxs.data1d import IntegratedQSlice
from cdsaxs.metadata import METADATA_KEYWORDS
import cdsaxs.plotting as plotting
from cdsaxs_gui_legacy import diffraction

UPDATE_Q_TRIGGERS = [
    "energy_ev", "wavelength_nm", "sdd_cm", "pixel_size_um", "center_px",
    "detector_phi_deg", "detector_phi_omega"
]


class Data2D():
    """
    Generic 2D data class with basic image functionalities. This class
    is not tied to any diffraction information.

    Attributes
    ----------
    image : NDArray
        Two-dimensional array containing the image as pixel intensities.
        The first dimension corresponds to image rows from top to bottom
        and the second dimension corresponds to image columns from left
        to right.
    """

    # Will keep track of image rotations and flips.
    # R# indicates number of counter-clockwise 90 degree rotations
    # Example: R3 indicates 270 degree counter-clockwise rotation
    # VF indicates a vertical flip
    # HF indicates a horizontal flip
    _image_transformations = []

    def __init__(self, image: NDArray[np.floating]):
        """
        Parameters
        ----------
        image : NDArray
            Two-dimensional array of image intensities. The first
            dimension corresponds to image rows from top to bottom and
            the second dimension corresponds to image columns from left
            to right.
        """

        self.image = image

    def rotate_image(self, degrees, direction='ccw'):
        """
        Rotate the image by a specified numer of degrees in the
        direction specified.

        The original image is always saved and can be recalled by
        using 'reset_image_orientation'.

        Parameters
        ----------
        degrees : float
            Number of degrees to rotate the image. This should be in
            increments of 90 degrees. A negative value will reverse the
            rotation direction specified by the 'direction' parameter,
            use caution.
        direction : str, optional
            Rotation direction. Default is 'ccw' which indicates a
            counterclockwise rotation. Set as 'cw' to indicate a
            clockwise rotation.
        """

        if degrees < 0:
            warnings.warn(
                "You have provided a negative value for degrees of rotation. "
                "This will reverse the direction specified in the 'direction' "
                "argument. For example, a -90 degree rotation "
                "counter-clockwise is the same as a 90 degree clockwise "
                "rotation. Did you intend this?"
            )
        # switch to counterclockwise degrees
        if direction == 'cw':
            degrees = 360 - degrees % 360
        else:
            degrees = degrees % 360
        # determine number of 90 degree rotations counterclockwise
        # this will round down to the nearest 90 degree rotation
        k = int(degrees/90)

        if k != 0:
            self.image = np.rot90(self.image, k=k, axes=(0, 1))
            self._image_transformations.append("R"+str(k))

    def flip_horizontally(self):

        self.image = np.flip(self.image, axis=1)
        self._image_transformations.append("HF")

    def flip_vertically(self):

        self.image = np.flip(self.image, axis=0)
        self._image_transformations.append("VF")

    def reset_image_orientation(self, transformations=None):
        """
        Return the image to its original orientation removing any
        rotations or flips that have been done.
        """
        if transformations is None:
            transformations = self._image_transformations[::-1]

        for tf in transformations:
            if tf == "VF":
                self.flip_vertically()
            elif tf == "HF":
                self.flip_horizontally()
            else:
                k = int(tf[1:])
                self.rotate_image(degrees=90*k, direction="cw")

        self._image_transformations = []

    def integrate_box(
            self,
            limits_axis0,
            limits_axis1,
            mode,
            axis,
            box_angle_deg=0,
            rotation_center=[0, 0],
            rotation_sampling_mode='bicubic'
    ):
        """
        Simple integration in a box defined by the [min, max) limits
        for each axis.

        Parameters
        ----------
        limits_axis0 : tuple[int, int]
            Defines the limits (indices) of the box in the first
            dimension. This is a half open range [min, max).
        limits_axis1 : tuple[int, int]
            Defines the limits (indices) of the box in the second
            dimension. This is a half open range [min, max).
        mode : str
            Sets the integration mode. This can be set to 'sum' or
            'mean'.
        axis : int
            The axis along which the integration should be performed.
            This can be set to either 0 (rows) or 1 (columns).
        box_angle_deg : float
            Rotate the box by the set number of degrees clockwise
            about the center point. Rotating the box will maintain the
            size of the box.
            Units are in degrees.
            Default value is 0.
        rotation_center : list
            Center of rotation if the box_angle_deg is not 0. The
            default is the upper left of the image.
        rotation_sampling_mode : str
            Set the resampling method used when a box angle is provided.
            The box rotation works by rotating the image underneath then
            extracting the box for integration. Resampling of the
            image intensities can be performed with the 'nearest',
            'bilinear', or 'bicubic' methods in the PILLOW package.
            Default value is 'bicubic'.
        """
        image = np.copy(self.image)
        image[image < 0] = np.nan
        image[np.isinf(image)] = np.nan
        image[np.isneginf(image)] = np.nan
        if box_angle_deg != 0:
            if rotation_sampling_mode == 'nearest':
                resample = Image.Resampling.NEAREST
            elif rotation_sampling_mode == 'bilinear':
                resample = Image.Resampling.BILINEAR
            else:
                resample = Image.Resampling.BICUBIC
            image = Image.fromarray(image)
            image = image.rotate(box_angle_deg, resample=resample,
                                 center=(rotation_center[1], rotation_center[0]),
                                 fillcolor=-50)
            image = np.array(image)

        if mode == 'sum':
            integrated_i = np.nansum(
                image[limits_axis0[0]:limits_axis0[1],
                      limits_axis1[0]:limits_axis1[1]],
                axis=axis
            )
        elif mode == 'mean':
            integrated_i = np.nanmean(
                image[limits_axis0[0]:limits_axis0[1],
                      limits_axis1[0]:limits_axis1[1]],
                axis=axis
            )
        else:
            raise ValueError(
                f"Integration mode of {mode} is not recognized. Accepted modes"
                " include 'sum' and 'mean'."
            )

        return integrated_i.reshape(-1), {
                'mode': mode,
                'axis': axis,
                'limits_axis0': limits_axis0,
                'limits_axis1': limits_axis1,
                'box_angle_deg': box_angle_deg,
                'rotation_center': rotation_center,
                'rotation_sampling_mode': rotation_sampling_mode,
                'rotated_image': np.copy(image) if box_angle_deg != 0 else None
            }


class DataQdyQdx(Data2D):
    """
    This class contains 2D scattering images with coordinates of
    y vs x defined in the detector coordinate frame with positive y
    in the upward vertical direction and positive x in the left
    horizontal direction. The z axis is then defined as normal
    incidence to follow the right-hand rule.

    In many instances the detector coordinates will align with the lab
    frame, where the z axis aligns with the beam path and the detector
    is configured normal to the primary beam.

    In cases where the detector has moved from the position with the
    incident beam normal to the surface, two angles can be defined.
    detector_phi : Rotation counterclockwise about the y-axis
        originating at the sample position in the x-z plane (lab frame).
    detector_omega : Rotation counterclockwise about the x-axis
        originating at the sample position in the y-z plane (lab frame).

    Attributes
    ----------
    image : NDarray
        Scattering image as a two-dimensional numpy array. The first
        dimension corresponds to the y-axis (detector frame) and the
        second dimension corresponds to the x-axis (detector frame).
    qdy : NDArray
        Scattering vector for each pixel along the detector y-axis.
    qdx : NDArray
        Scattering vector for each pixel along the detector x-axis.
    metadata : dict
        Relevant scattering metadata to the image acquisition. These are
        key : value paris where the key must be selected from the
        metadata list below and the value format depends on the
        requirements of the specific parameter. See the user
        documentation for a thorough description of each of these
        parameters.
        Note: Only wavelength or energy should be specified, not both.
    user_params : dict
        Additional user-provided parameters or metadata relevant to the
        data workflow. These are not accessed by the cdsaxs package.
    name : str
        Identifier for this image acquisition. The default when using
        the cdsaxs loaders is the filename, but be cautious when
        creating a Dataset as the filenames alone may not always result
        in unique identifiers for each image. A custom name can be set
        by passing 'name' metadata.

    Metadata Keywords
    -----------------
    sample_kappa_deg
    sample_phi_deg
    sample_omega_deg

    energy_ev
    wavelength_nm
    exposure_time_s
    sdd_cm
    pixel_size_um

    detector_phi_deg
    detector_omega_deg

    scaling_factor
    I0
    beam_current
    center_px

    data_directory
    filename
    name

    """

    def __init__(
            self,
            image: NDArray[np.floating],
            metadata: dict = None,
            user_params: dict = None,
            name: str = None
    ):
        """Create an instance od DataQdyQdx"""

        # run base class init
        super().__init__(image)

        self.metadata = {}
        if metadata is not None:
            self.update_metadata(metadata)

        self.user_params = {}
        if user_params is not None:
            self.update_user_params(user_params)

        self.qdy = None
        self.qdx = None

        # calculate the q vectors if all required metadata is present
        self.calculate_q(suppress_errors=True)

        self.name = name if name is not None else\
            metadata['name'] if 'name' in metadata.keys() else\
            metadata['filename'] if 'filename' in metadata.keys() else 'name'

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
                    # handle special wavelength/energy relationship
                    if key == 'wavelength_nm':
                        self.metadata['energy_ev'] =\
                            calculators.wavelength_to_energy(value)
                    elif key == 'energy_ev':
                        self.metadata['wavelength_nm'] =\
                            calculators.energy_to_wavelength(value)
            if len([x for x in metadata.keys() if x in UPDATE_Q_TRIGGERS]) > 0:
                self.calculate_q(suppress_errors=True)

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

    def calculate_q(self, suppress_errors: bool = False):
        """
        Calculate the qdy and qdx vectors along the image axes if
        all required metadata is available.

        Parameters
        ----------
        suppress_errors : bool, optional
            If set to True, this method will try to calculate the
            q vectors if the required metadata is availabe, but it
            will not raise an error if the parameters are not available.
            Default value is False.
        """
        required_keywords = ["center_px", "sdd_cm", "wavelength_nm",
                             "pixel_size_um"]
        missing_keywords = []
        for word in required_keywords:
            if word not in self.metadata.keys():
                missing_keywords.append(word)
        if len(missing_keywords) > 0 and not suppress_errors:
            self.qdy = None
            self.qdx = None
            raise ValueError(
                "The following metadta is missing to calculate q: "
                f"{missing_keywords}"
            )
        elif len(missing_keywords) > 0:
            self.qdy = None
            self.qdx = None
        else:
            # TODO: update this when diffraction.py is refactored
            qdy = diffraction.qy_pixels_to_qy(
                -1*np.arange(0, self.image.shape[0])
                + self.metadata['center_px'][0],
                self.metadata["wavelength_nm"],
                self.metadata["pixel_size_um"],
                self.metadata["sdd_cm"],
            )
            qdx = diffraction.qxz_pixels_to_qxz(
                -1*np.arange(0, self.image.shape[1])
                + self.metadata['center_px'][1],
                self.metadata["wavelength_nm"],
                self.metadata["pixel_size_um"],
                self.metadata["sdd_cm"],
            )
            self.qdy = qdy
            self.qdx = qdx

    def rotate_image(self, degrees, direction='ccw'):
        """
        Rotate the scattering image by a specified numer of degrees in
        the direction specified. The scattering vectors qdy and qdx as
        well as the beam center position in metadata (center_px) will
        be updated to follow the rotation (if they exist).

        The current image rotation with respect to the original image
        is saved. The rotations can be undone with 'reset_rotations'.

        Parameters
        ----------
        degrees : float
            Number of degrees to rotate the image. This should be in
            increments of 90 degrees. A negative value will reverse the
            rotation direction specified by the 'direction' parameter,
            use caution.
        direction : str, optional
            Rotation direction. Default is 'ccw' which indicates a
            counterclockwise rotation. Set as 'cw' to indicate a
            clockwise rotation.
        """
        # temporarily store information about current state
        length0 = self.image.shape[0]
        length1 = self.image.shape[1]
        try:
            center_px = self.metadata['center_px']
        except KeyError:
            center_px = None

        # do the rotation and store original image information if needed
        super().rotate_image(degrees, direction=direction)
        k = int(self._image_transformations[-1][1:])

        if k == 1:
            if center_px is not None:
                self.metadata['center_px'] = [
                    length1-center_px[1]-1,
                    center_px[0]]

        if k == 2:
            if center_px is not None:
                self.metadata['center_px'] = [
                    length0-center_px[0]-1,
                    length1-center_px[1]-1]

        if k == 3:
            if center_px is not None:
                self.metadata['center_px'] = [
                    center_px[1],
                    length0-center_px[0]-1]

        # recalcualte q if possible
        self.calculate_q(suppress_errors=True)

    def flip_horizontally(self):
        super().flip_horizontally()
        try:
            center_px = self.metadata['center_px']
            self.metadata['center_px'] = [
                center_px[0],
                self.image.shape[1] - center_px[1] - 1
            ]
        except KeyError:
            pass

        # recalcualte q if possible
        self.calculate_q(suppress_errors=True)

    def flip_vertically(self):
        super().flip_vertically()
        try:
            center_px = self.metadata['center_px']
            self.metadata['center_px'] = [
                self.image.shape[0] - center_px[0] - 1,
                center_px[1]
            ]
        except KeyError:
            pass

        # recalcualte q if possible
        self.calculate_q(suppress_errors=True)

    def reset_image_orientation(self, transformations=None):
        """
        Return the scattering image to its original orientation
        removing any rotations that may have been done. The beam center
        and scattering vectors will be tracked and updated through this
        process (if they exist).
        """
        if transformations is None:
            transformations = self._image_transformations[::-1]

        for tf in transformations:
            if tf == "VF":
                self.flip_vertically()
            elif tf == "HF":
                self.flip_horizontally()
            else:
                k = int(tf[1:])
                self.rotate_image(degrees=90*k, direction="cw")

        self._image_transformations = []

    def integrate_box(
        self,
        limits_qdy_px: list | tuple,
        limits_qdx_px: list | tuple,
        mode: str,
        axis: str | int,
        show_plot=False,
        log_scale=True,
        box_angle_deg: float = 0,
        rotation_sampling_mode: str = 'bicubic',
        rotation_center_point: list | tuple = None,
        # interactive_plot=True
    ) -> IntegratedQSlice:
        """
        Integrate a box defined by indexing limits.

        Parameters
        ----------
        limits_qdy_px : iterable of int
            Pixel range along qdy axis for integration box.
            Half open range of [min, max).
        limits_qdx_px : iterable of int
            Pixel range along qdx axis for integration box.
            Half open range of [min, max).
        mode : str
            Integration mode, either 'sum' or 'mean'.
        axis : str, int
            Axis to integrate over, either 'qdy' or 'qdx'. The axis indices
            can also be used, 0 for 'qdy' or 1 for 'qdx'. For example, if
            axis is set to 'qdy', integration will return I vs. qdx data.
        show_plot : bool, optional
            If set to False, the scattering image overlaid with the
            integration box boundaries will be shown in a first figure
            and the one-dimensional data will be shown in a second figure.
            Default value is False.
        log_scale : bool, optional
            If set to True, the plots will show the scattering intensity
            on a log scale. If set to False, intensity will be displayed
            on a linear scale. This only applies to the plots and does
            not affect the data operation.
            Default value is True.
        box_angle_deg : float
            Rotate the box by the set number of degrees clockwise
            about the beam center point. Rotating the box will maintain 
            the size of the box.
            Units are in degrees.
            Default value is 0.
        rotation_sampling_mode : str
            Set the resampling method used when a box angle is provided.
            The box rotation works by rotating the image underneath then
            extracting the box for integration. Resampling of the
            image intensities can be performed with the 'nearest',
            'bilinear', or 'bicubic' methods in the PILLOW package.
            Default value is 'bicubic'.
        interactive_plot : bool, optional
            If set to True, the plots returned will be interactive plots
            built via Plotly. If set to False, the plots returned will be
            static matplotlib figures.
            TODO: currently this is disabled and only True is accepted.
            Default value is True.

        Returns
        -------
        IntegratedQSlice
            One-dimensional I vs. q data extracted from the integration.

        """

        if isinstance(axis, str):
            if axis == 'qdy':
                axis = 0
            elif axis == 'qdx':
                axis = 1
            else:
                raise ValueError(f"Invalid integration axis of {axis}.")

        # access parent method of box integration
        integrated_i, params = super().integrate_box(
            limits_axis0=limits_qdy_px,
            limits_axis1=limits_qdx_px,
            mode=mode,
            axis=axis,
            box_angle_deg=box_angle_deg,
            rotation_center=self.metadata['center_px'],
            rotation_sampling_mode=rotation_sampling_mode
        )

        # extract scattering vector for this integration
        if axis == 0:
            q = self.qdx[limits_qdx_px[0]:limits_qdx_px[1]]
        elif axis == 1:
            q = self.qdy[limits_qdy_px[0]:limits_qdy_px[1]]

        # create instance of IntegratedQSlice to hold integration metadata
        integrated_q_slice = IntegratedQSlice(
            q=q,
            Iq=integrated_i,
            q_axis='qdx' if axis == 0 else 'qdy',
            name=self.name,
            limits_axis0=params["limits_axis0"],
            limits_axis1=params["limits_axis1"],
            mode=mode,
            integration_axis=params["axis"],
            box_angle_deg=params['box_angle_deg'],
            rotation_sampling_mode=params['rotation_sampling_mode'],
            rotation_center=params['rotation_center'],
            rotated_image=params['rotated_image']
        )

        if show_plot:
            fig, fig_slice = plotting.plot_QdyQdx_integration(
                self,
                integrated_q_slice=integrated_q_slice,
                log_scale=log_scale
            )
            iplot(fig)
            iplot(fig_slice)

        return integrated_q_slice

    def integrate_box_of_size(
            self,
            size_qdy_px: int,
            size_qdx_px: int,
            mode: str,
            axis: str | int,
            shift_box_qdy_px: int = 0,
            shift_box_qdx_px: int = 0,
            box_angle_deg: float = 0,
            rotation_sampling_mode: str = 'bicubic',
            show_plot=False,
            log_scale=True,
    ):
        """
        Integrate a box of a specific size. By default this box is
        centered at the closest pixel to the beam center position (q=0),
        but it can be shifted in either qdy or qdx by a set number of
        pixels.

        Parameters
        ----------
        size_qdy_px : int
            Size of the integration box in pixels along qdy axis.
        size_qdx_px : int
            Size of the integration box in pixels along qdx axis.
        mode : str
            Integration mode, either 'sum' or 'mean'.
        axis : str, int
            Axis to integrate over, either 'qdy' or 'qdx'. The axis indices
            can also be used, 0 for 'qdy' or 1 for 'qdx'. For example, if
            axis is set to 'qdy', integration will return I vs. qdx data.
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            Default value is 0.
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.
        box_angle_deg : float
            Rotate the box by the set number of degrees clockwise
            about the center point. Rotating the box will maintain the
            size of the box.
            Units are in degrees.
            Default value is 0.
        rotation_sampling_mode : str
            Set the resampling method used when a box angle is provided.
            The box rotation works by rotating the image underneath then
            extracting the box for integration. Resampling of the
            image intensities can be performed with the 'nearest',
            'bilinear', or 'bicubic' methods in the PILLOW package.
            Default value is 'bicubic'.
        show_plot : bool, optional
            If set to False, the scattering image overlaid with the
            integration box boundaries will be shown in a first figure
            and the one-dimensional data will be shown in a second figure.
            Default value is False.
        log_scale : bool, optional
            If set to True, the plots will show the scattering intensity
            on a log scale. If set to False, intensity will be displayed
            on a linear scale. This only applies to the plots and does
            not affect the data operation.
            Default value is True.
        interactive_plot : bool, optional
            If set to True, the plots returned will be interactive plots
            built via Plotly. If set to False, the plots returned will be
            static matplotlib figures.
            TODO: currently this is disabled and only True is accepted.
            Default value is True.

        Returns
        -------
        IntegratedQSlice
            One-dimensional I vs. q data extracted from the integration.

        """

        # figure out where the box lies with respect to beam center
        # make sure that the box doesn't fall off the image
        center0, center1 = self.metadata['center_px']

        center0 = int(np.round(center0, 0))  # closest pixel
        min0 = center0 - int(size_qdy_px/2) - shift_box_qdy_px
        max0 = min0 + size_qdy_px
        min0 = max(min0, 0)
        max0 = min(max0, self.image.shape[0])

        center1 = int(np.round(center1, 0))  # closest pixel
        min1 = center1 - int(size_qdx_px/2) - shift_box_qdx_px
        max1 = min1 + size_qdx_px
        min1 = max(min1, 0)
        max1 = min(max1, self.image.shape[1])

        # integrate the box area of image
        integrated_q_slice = self.integrate_box(
            limits_qdy_px=[min0, max0],
            limits_qdx_px=[min1, max1],
            mode=mode,
            axis=axis,
            box_angle_deg=box_angle_deg,
            rotation_sampling_mode=rotation_sampling_mode,
        )

        if show_plot:
            fig, fig_slice = plotting.plot_QdyQdx_integration(
                self, integrated_q_slice=integrated_q_slice,
                log_scale=log_scale)
            iplot(fig)
            iplot(fig_slice)

        return integrated_q_slice

    def integrate_box_of_q_range(
            self,
            range_qdy: list | tuple,
            range_qdx: list | tuple,
            mode: str,
            axis: str | int,
            box_angle_deg: float = 0,
            rotation_sampling_mode: str = 'bicubic',
            show_plot=False,
            log_scale=True,
    ):
        """
        Integrate a box defined by scattering vector limits.

        Parameters
        ----------
        range_qdy : iterable of float
            Range of scattering vector qdy defining the integration box.
            Half open range of [min, max). Pixels with a q value that
            satisfies min <= q < max will be accepted into the box.
        range_qdx : iterable of float
            Range of scattering vector qdx definiing the integration box.
            Half open range of [min, max). Pixels with a q value that
            satisfies min <= q < max will be accepted into the box.
        mode : str
            Integration mode, either 'sum' or 'mean'.
        axis : str, int
            Axis to integrate over, either 'qdy' or 'qdx'. The axis indices
            can also be used, 0 for 'qdy' or 1 for 'qdx'. For example, if
            axis is set to 'qdy', integration will return I vs. qdx data.
        box_angle_deg : float
            Rotate the box by the set number of degrees clockwise
            about the center point. Rotating the box will maintain the
            size of the box.
            Units are in degrees.
            Default value is 0.
        rotation_sampling_mode : str
            Set the resampling method used when a box angle is provided.
            The box rotation works by rotating the image underneath then
            extracting the box for integration. Resampling of the
            image intensities can be performed with the 'nearest',
            'bilinear', or 'bicubic' methods in the PILLOW package.
            Default value is 'bicubic'.
        show_plot : bool, optional
            If set to False, the scattering image overlaid with the
            integration box boundaries will be shown in a first figure
            and the one-dimensional data will be shown in a second figure.
            Default value is False.
        log_scale : bool, optional
            If set to True, the plots will show the scattering intensity
            on a log scale. If set to False, intensity will be displayed
            on a linear scale. This only applies to the plots and does
            not affect the data operation.
            Default value is True.
        interactive_plot : bool, optional
            If set to True, the plots returned will be interactive plots
            built via Plotly. If set to False, the plots returned will be
            static matplotlib figures.s
            TODO: currently this is disabled and only True is accepted.
            Default value is True.

        Returns
        -------
        IntegratedQSlice
            One-dimensional I vs. q data extracted from the integration.

        """

        # fix the min, max order if the user provided them reversed
        range_qdy = [min(range_qdy), max(range_qdy)]
        range_qdx = [min(range_qdx), max(range_qdx)]

        qdy_indices = np.where((self.qdy >= range_qdy[0])
                               & (self.qdy < range_qdy[1]))[0]
        limits_qdy_px = (np.min(qdy_indices), np.max(qdy_indices)+1)

        qdx_indices = np.where((self.qdx >= range_qdx[0])
                               & (self.qdx < range_qdx[1]))[0]
        limits_qdx_px = (np.min(qdx_indices), np.max(qdx_indices)+1)

        integrated_q_slice = self.integrate_box(
            limits_qdy_px=limits_qdy_px,
            limits_qdx_px=limits_qdx_px,
            mode=mode,
            axis=axis,
            box_angle_deg=box_angle_deg,
            rotation_sampling_mode=rotation_sampling_mode,
        )

        if show_plot:
            fig, fig_slice = plotting.plot_QdyQdx_integration(
                self, integrated_q_slice=integrated_q_slice,
                log_scale=log_scale)
            iplot(fig)
            iplot(fig_slice)

        return integrated_q_slice

    def find_peaks1D(self,
                     box_mode,
                     box_params: dict,
                     peak_params: dict,
                     peak_find_scale='linear',
                     show_plot=True):
        """
        Simple peak finding function in 1D to determine appropriate
        rotation angle of the sample coordinate system in the x-y
        detector plane.

        The box used to search for peaks is defined in the same way as
        the integrator methods. This method assumes that there is only
        a one-dimensional line of peaks along the axis NOT defined as
        the integration axis in box_params. The location of the peaks
        along the integration axis are then determined at the max
        intensity value at a single position along the first axis.

        Parameters
        ----------
        box_mode : str
            Type of integration box to use. Options are:
                'box' : use DataQdyQdx.integrate_box
                'box_size' : use DataQdyQdx.integrate_box_of_size
                'q_range' : use DataQdyQdx.integrate_box_of_q_range
        box_params : dict
            Dictionary of keyword arguments for the selected integration
            method (box_mode). See the docstring of the corresponding
            integration method for more information of available
            arguments and their definitions.
        peak_params : dict
            Dictionary of keyword arguments for the scipy.find_peaks
            algorithm; see scipy documentation for more information.
        peak_find_scale : str
            The scale of the intesity data to use for peak finding.
            Can be set to 'linear' or 'log'.
            Default value is 'linear'.
        show_plot : bool
            If set to True, a first figure will display the scattering
            image overlaid with the integration box and markers on each
            detected peak while a second figure will show the 1D slice
            extracted from the integration and vertical lines at each
            peak position.

        Returns
        -------
        list[tuple[float, float]]
            List of peak positions in (qdy, qdx) scattering vector coordiantes.
        list[tuple[int, int]]
            List of peak positions in (px_dy, px_dx) pixel coordinates.
        float
            Angle of rotation of best line fit to the peaks clockwise
            from a line parallel to the qdx axis.
            Units are degrees.
        tuple[float, float]
            Results from linear fit to the peaks of (slope, intercept).
            The units are in pixels; keep in mind for images pixels
            are numbered from top to bottom and left to right (rows and
            columns).
        IntegratedQSlice
            Integrated I vs. Q slice used for peak finding.
        """

        # integrate over the box
        if box_mode == 'box':
            integrated_q_slice = self.integrate_box(**box_params)
        elif box_mode == 'box_size':
            integrated_q_slice = self.integrate_box_of_size(**box_params)
        elif box_mode == 'q_range':
            integrated_q_slice = self.integrate_box_of_q_range(**box_params)
        else:
            raise ValueError(
                f"The box_mode {box_mode} is not recognized."
            )

        # find peaks along the integrated I vs. q spectra
        if peak_find_scale == 'linear':
            peaks, _ = find_peaks(integrated_q_slice.Iq, **peak_params)
        elif peak_find_scale == 'log':
            peaks, _ = find_peaks(
                np.log10(integrated_q_slice.Iq), **peak_params)
        else:
            raise ValueError(
                f"The peak_find_scale {peak_find_scale} is not recognized."
            )

        min0, max0 = integrated_q_slice.limits_axis0
        min1, max1 = integrated_q_slice.limits_axis1
        image_box = self.image[min0:max0, min1:max1]

        if box_params['axis'] == 0 or box_params['axis'] == 'qdy':
            peaks_other = np.argmax(image_box[:, peaks], axis=0)
            peaks_px = [
                (y+min0, x+min1) for x, y in zip(peaks, peaks_other)]
        else:
            peaks_other = np.argmax(image_box[peaks, :], axis=1)
            peaks_px = [
                (y+min0, x+min1) for y, x in zip(peaks, peaks_other)]

        peaks_q = [
            (self.qdy[y], self.qdx[x]) for y, x in peaks_px]

        peaks_array = np.array(peaks_px)
        try:
            fit = linregress(peaks_array[:, 1], peaks_array[:, 0])
            angle = np.rad2deg(np.arctan(fit.slope))
            slope, intercept = (fit.slope, fit.intercept)
        except ValueError:
            # vertical line
            angle = 90
            slope = np.nan
            intercept = np.nan

        if show_plot:
            fig, fig_slice = plotting.plot_QdyQdx_find_peaks(
                self, integrated_q_slice, peaks_array)
            iplot(fig)
            iplot(fig_slice)

        return peaks_q, peaks_px, angle, (slope, intercept), integrated_q_slice

    def plot_data(
            self,
            show_pixels=False,
            log_scale=True,
            vmin=None,
            vmax=None,
            # interactive_plot=True
    ):
        """
        Plot the scattering image.

        Parameters
        ----------
        show_pixels : bool
            If set to True, instead of the scattering vector, pixel
            indices will be shown along the qdy and qdx axes.
            Default value is False.
        log_scale : bool
            If set to True, the scattering intensity will be displayed
            on a log sale. If set to False, the scattering intensity
            will be displayed on a linear scale.
            Default value is True
        vmin : float
            Manually set the minimum of the color bar range for
            plotting intensity.
        vmax : float
            Manually set the maximum of the color bar range for
            plotting intensity.
        interactive_plot : bool
            If set to True, an interactive plot built with Plotly will
            TODO: currently this is disabled and only accepts True.
        """

        if show_pixels or self.qdy is None:
            axis0 = None
            axis1 = None
            axis0_type = 'px_dy'
            axis1_type = 'px_dx'
        else:
            axis0 = self.qdy
            axis1 = self.qdx
            axis0_type = 'qdy'
            axis1_type = 'qdx'

        fig = plotting.plot2D(
            self.image,
            axis0=axis0, axis1=axis1,
            axis0_type=axis0_type, axis1_type=axis1_type,
            title=self.name,
            log_scale=log_scale,
            vmin=vmin,
            vmax=vmax
        )

        iplot(fig)

    def find_beam_center_from_peaks(
            self,
            beam_center_guess,
            size_qdy_px,
            size_qdx_px,
            peak_axis,
            peak_params: dict,
            peak_find_scale='linear'):
        """
        Attempt to locate the beam center position using simple
        1D peak finding. See DataQdyQdx.find_peaks1D for a more
        detailed description of the peak finding process. For this
        method, only an integration box of size can be used and it
        must be centered on the beam center guess so that you have
        equal number of peaks on each side of the beam (ideally).

        In many cases the beam center position is likely to fall on
        an integer pixel value. This is because the peak finding
        algorithm only returns the pixel on which the peak is and does
        not perform any additional fit of the local intensity to determine
        a float pixel location of the peak.
        TODO: implement local gaussian fits for more accurate positions

        The beam center is determined by matching the same order peaks
        in the negative and positive peak_axis direction. The peak_axis
        is not the integration axis.

        Parameters
        ----------
        beam_center_guess : iterable of int
            Initial guess of the beam center position in pixels along
            qdy and qdx [center_px_qdy, center_px_qdx].
        size_qdy_px : int
            Box size in pixels along the qdy axis.
        size_qdx_px : int
            Box size in pixels along the qdx axis.
        peak_axis : str, int
            Axis along which the peaks are present, either 'qdy' or 'qdx'.
            The axis indices can also be used, 0 for 'qdy' or 1 for 'qdx'.
            For example, if peak_axis is set to 'qdx', peaks will be
            detected along the qdx axis.
        peak_params : dict
            Dictionary of keyword arguments for the scipy.find_peaks
            algorithm; see scipy documentation for more information.
        peak_find_scale = 'linear'
            The scale of the intesity data to use for peak finding.
            Can be set to 'linear' or 'log'.
            Default value is 'linear'.
        show_plot : bool
            If set to True, a first figure will display the scattering
            image overlaid with the integration box and markers on each
            detected peak while a second figure will show the 1D slice
            extracted from the integration and vertical lines at each
            peak position. The determiend beam center will be shown
            with dashed red lines.

        Returns
        -------
        tuple[float, float]
            Beam center coordinates in units of pixels, [px_dy, px_dx].
        """

        if isinstance(peak_axis, str):
            if peak_axis == 'qdy':
                peak_axis = 0
            elif peak_axis == 'qdx':
                peak_axis = 1
            else:
                raise ValueError(
                    f"Invalid integration peak axis of {peak_axis}.")

        box_params = {
            "size_qdy_px": size_qdy_px,
            "size_qdx_px": size_qdx_px,
            "axis": 1 - peak_axis,
            "mode": 'sum',
        }
        # use the box shift to correctly center the box on the beam center
        # guess and not the default or current beam center
        box_params["shift_box_qdy_px"] = int(
            np.round(self.metadata['center_px'][0], 0) - beam_center_guess[0])
        box_params["shift_box_qdx_px"] = int(
            np.round(self.metadata['center_px'][1], 0) - beam_center_guess[1])

        peaks_q, peaks_px, angle, (slope, intercept), integrated_q_slice =\
            self.find_peaks1D(
                box_mode='box_size',
                box_params=box_params,
                peak_params=peak_params,
                peak_find_scale=peak_find_scale,
                show_plot=False
            )

        peaks = np.array(peaks_px)[:, peak_axis]
        low_peaks = peaks[peaks < beam_center_guess[peak_axis]]
        high_peaks = peaks[peaks > beam_center_guess[peak_axis]]
        centers = []
        for low, high in zip(np.flip(low_peaks), high_peaks):
            centers.append(np.mean([low, high]))

        if peak_axis == 1:
            center_qdx = np.mean(centers)
            center_qdy = slope*center_qdx + intercept
        elif peak_axis == 0:
            center_qdy = np.mean(centers)
            if np.isnan(slope):
                # this means the peaks form perfectly vertical line
                center_qdx = np.array(peaks_px)[0, 0]
            else:
                center_qdx = (center_qdy-intercept)/slope

        fig, fig_slice = plotting.plot_find_beam_center(
            self, integrated_q_slice, np.array(peaks_px),
            [center_qdy, center_qdx])
        iplot(fig)
        iplot(fig_slice)

        return center_qdy, center_qdx

    def find_sdd_from_peaks(
            self,
            expected_srm_pitch,
            size_qdy_px,
            size_qdx_px,
            peak_axis,
            peak_params: dict,
            peak_find_scale='linear'):
        """
        Calculate the sample to detector distance (SDD) from the
        known pitch of a SRM (standard reference material) sample
        using simple 1D peak finding. See DataQdyQdx.find_peaks1D
        for a more detailed description of the peak finding process.
        For this method, only an integration box of size can be used
        and it must be centered on the beam center as determined by 
        the user or the find_beam_center_from_peaks method. 
        
        Only the 1st order peaks will be compared to the expected pitch of
        the SRM sample to determine the SDD.
        TODO: implement error handling when determining the SDD
        
        In many cases the beam center position is likely to fall on
        an integer pixel value. This is because the peak finding
        algorithm only returns the pixel on which the peak is and does
        not perform any additional fit of the local intensity to determine
        a float pixel location of the peak.
        TODO: implement local gaussian fits for more accurate positions


        Parameters
        ----------
        expected_srm_pitch : float
            Expected pitch of the SRM sample in the units of Angstroms.
        size_qdy_px : int
            Box size in pixels along the qdy axis.
        size_qdx_px : int
            Box size in pixels along the qdx axis.
        peak_axis : str, int
            Axis along which the peaks are present, either 'qdy' or 'qdx'.
            The axis indices can also be used, 0 for 'qdy' or 1 for 'qdx'.
            For example, if peak_axis is set to 'qdx', peaks will be
            detected along the qdx axis.
        peak_params : dict
            Dictionary of keyword arguments for the scipy.find_peaks
            algorithm; see scipy documentation for more information.
        peak_find_scale = 'linear'
            The scale of the intesity data to use for peak finding.
            Can be set to 'linear' or 'log'.
            Default value is 'linear'.
        show_plot : bool
            If set to True, a first figure will display the scattering
            image overlaid with the integration box and markers on each
            detected peak while a second figure will show the 1D slice
            extracted from the integration and vertical lines at each
            peak position. The determiend beam center will be shown
            with dashed red lines.

        Returns
        -------
        calculated_sdd : float
            Sample detector distance (SDD) in the units of cm.
        """

        if isinstance(peak_axis, str):
            if peak_axis == 'qdy':
                peak_axis = 0
            elif peak_axis == 'qdx':
                peak_axis = 1
            else:
                raise ValueError(
                    f"Invalid integration peak axis of {peak_axis}.")

        box_params = {
            "size_qdy_px": size_qdy_px,
            "size_qdx_px": size_qdx_px,
            "axis": 1 - peak_axis,
            "mode": 'sum',
        }
        
        #Find peaks in the 1D slice
        peaks_q, peaks_px, angle, (slope, intercept), integrated_q_slice =\
            self.find_peaks1D(
                box_mode='box_size',
                box_params=box_params,
                peak_params=peak_params,
                peak_find_scale=peak_find_scale,
                show_plot=False
            )

        peaks = np.array(peaks_px)[:, peak_axis]
        peaks_pair = np.array(peaks_px)[:, 1 - peak_axis]
        low_peaks = peaks[peaks < self.metadata['center_px'][peak_axis]] - self.metadata['center_px'][peak_axis]
        low_peaks_pair = peaks_pair[peaks < self.metadata['center_px'][peak_axis]] - self.metadata['center_px'][1-peak_axis]
        high_peaks = peaks[peaks > self.metadata['center_px'][peak_axis]] - self.metadata['center_px'][peak_axis]
        high_peaks_pair = peaks_pair[peaks > self.metadata['center_px'][peak_axis]] - self.metadata['center_px'][1-peak_axis]
        
        sdds = []
        n = 1     
        for low, high , low_pair, high_pair in zip(np.flip(low_peaks), high_peaks, np.flip(low_peaks_pair), high_peaks_pair):
            sin_theta = n*self.metadata['wavelength_nm'] / expected_srm_pitch
            theta = np.arcsin(sin_theta)
            #theta = np.arcsin((n*self.sample_metadata['wavelength'])/(2*expected_srm_pitch))
            
            r_low =  np.abs(low)* self.metadata["pixel_size_um"]/1000
            r_low_pair = np.abs(low_pair) * self.metadata["pixel_size_um"]/1000
            r_high = np.abs(high) * self.metadata["pixel_size_um"]/1000
            r_high_pair = np.abs(high_pair) * self.metadata["pixel_size_um"]/1000
            
            #calculate magnitude of vector from beam center to peak
            r_low = np.sqrt(r_low**2 + r_low_pair**2)
            r_high = np.sqrt(r_high**2 + r_high_pair**2)
            
            sdd_low = r_low/np.tan(theta)
            sdd_high = r_high/np.tan(theta)
            
            sdds.append(sdd_low)
            sdds.append(sdd_high)
            n += 1
        
        #calculate average SDD from all peaks
        average_sdd = np.mean(sdds)/10  # convert to cm

        fig, fig_slice = plotting.plot_find_beam_center(
            self, integrated_q_slice, np.array(peaks_px),
            self.metadata['center_px'])
        iplot(fig)
        iplot(fig_slice)

        return average_sdd



    def integrate_autorotated_box(
            self,
            peak_find_box_mode: str,
            peak_find_box_params: dict,
            peak_axis: str,
            peak_params: dict,
            box_mode: str,
            box_params: dict,
            peak_find_scale: str,
            show_plot: True,
    ):
        """
        Integrate a 2D qdy vs qdx image with any of the standard
        integrator methods after performing an automated peak finding
        function to determine the rotation angle of the box.

        CAUTION: this method assumes that this is only a minor
        angular offset of the sample about the beam path axis. It
        functions by rotating the image underneath to align the qsx
        axis with the qdx axis. This may result in some unexpected
        behavior at large values of sample_phi_deg.
        TODO: figure out proper qd to qs operation for this rotation.

        This function requires two boxes:
        1. A larger box for the auto-peak finding function. This
           should encompass a series of peaks along a single direction
           only. See the peak_find1d method for more information.
        2. The actual box dimensions for integration. After the angle
           of rotaiton is determined via peak finding, the image will
           be rotated and the second box applied to extract the 1d
           slice.

        Parameters
        ----------
        peak_find_box_mode : str
            Type of box to use for the peak finding function. The box
            is defined the same way as the integrators:
                'box' : index ranges as in DataQdyQdx.integrate_box
                'box_size' : box size centered or offset from the beam
                    center as in DataQdyQdx.integrate_box_of_size
                'q_range' : scattering vector ranges as in
                    DataQdyQdx.integrate_box_of_q_range
        peak_find_box_params : dict
            Dictionary of keyword arguments for the selected
            integration method (peak_find_box_mode). See the docstring
            for the corresponding integration method for more details
            of available arguments and their definitions.
        peak_axis : str, int
            Axis along which the peaks are present, either 'qdy' or 'qdx'.
            The axis indices can also be used, 0 for 'qdy' or 1 for 'qdx'.
            For example, if peak_axis is set to 'qdx', peaks will be
            detected along the qdx axis.
        peak_params : dict
            Dictionary of keyword arguments for the scipy.find_peaks
            algorithm; see scipy documentation for more information.
        box_mode : str
            Type of box used for the integration step. Same options
            are available as peak_find_box_mode but the choice does
            not have to be the same.
        box_params : dict
            Dictionary of box keyword arguments for the box_mode
            chosen.
        peak_find_scale : str
            The scale of the intensity data to use for peak finding.
            Can be set to 'linear' or 'log'.
            Default value is 'linear'.
        show_plot : bool
            If set to True, the first figure will display the
            scattring image overlaid with the peak finding box and
            markers on each detected peak. The second figure will show
            the rotated image and overlaid integration box. The third
            figure will show the 1D slice extracted from the
            integration and vertical lines at each peak position.

        Returns
        -------
        IntegratedQSlice
            One-dimensional I vs. q data extracted from the integration.
        """

        peaks_q, peaks_px, angle, \
            (slope, intercept), integrated_q_slice_peak = self.find_peaks1D(
                box_mode=peak_find_box_mode,
                box_params=peak_find_box_params,
                peak_params=peak_params,
                peak_find_scale=peak_find_scale,
                show_plot=False
            )

        box_params['box_angle_deg'] = angle
        if box_mode == 'box':
            integrated_q_slice = self.integrate_box(**box_params)
        elif box_mode == 'box_size':
            integrated_q_slice = self.integrate_box_of_size(**box_params)
        elif box_mode == 'q_range':
            integrated_q_slice = self.integrate_box_of_q_range(**box_params)
        else:
            raise ValueError(
                f"The box_mode {box_mode} is not recognized."
            )

        if show_plot:
            fig_peak, _ = plotting.plot_QdyQdx_find_peaks(
                self, integrated_q_slice_peak, np.array(peaks_px))
            # TODO: look into what is correct here
            # vmin = np.log10(fig_peak.layout.coloraxis['cmin'])
            # vmax = np.log10(fig_peak.layout.coloraxis['cmax'])
            vmin = fig_peak.layout.coloraxis['cmin']
            vmax = fig_peak.layout.coloraxis['cmax']
            fig, fig_slice = plotting.plot_QdyQdx_integration(
                self, integrated_q_slice=integrated_q_slice,
                log_scale=True, vmin=vmin, vmax=vmax)
            iplot(fig_peak)
            iplot(fig)
            iplot(fig_slice)

        return integrated_q_slice

    def _check_metadata(self, metadata):
        """
        Check the metadata dictionary for:
        - unaccepted metadata keywords
        - overspecified wavelength/energy (onle one should be set)
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

        if "energy_ev" in metadata.keys() and\
                "wavelength_nm" in metadata.keys():
            raise ValueError(
                "You have specified both the source energy and wavelength. "
                "Only one of these can be specified and the other is "
                "calculated. To avoid over-specifying or conflicting values, "
                "please only use one of these values. "
            )
        return True
