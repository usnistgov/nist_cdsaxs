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

    _ccw_rotation_counter = 0

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

        if degrees < 0:
            warnings.warn(
                "You have provided a negative value for degrees of rotation. "
                "This will reverse the direction specified in the 'direction' "
                "argument. For example, a -90 degree rotation "
                "counter-clockwise is the same as a 90 degree clockwise "
                "rotation. Did you intend this?"
            )
        # determine number of 90 degree rotations counterclockwise
        k = int(degrees/90) % 4
        if direction == 'cw':
            k *= -1

        if k != 0:
            self.image = np.rot90(self.image, k=k, axes=(0, 1))

        self._ccw_rotation_counter = (self._ccw_rotation_counter + k) % 4

    def reset_rotations(self):
        """
        Return the image to its original orientation removing any
        rotations that have been done.
        """
        if self._ccw_rotation_counter != 0:
            degrees = -90*self._ccw_rotation_counter
            with warnings.catch_warnings():
                # ignore the warning meant for direct use of rotate()
                warnings.simplefilter('ignore')
                self.rotate_image(degrees)

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
            image = np.array(Image)

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
                'limits_axis1': limits_axis1
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
            raise ValueError(
                "The following metadta is missing to calculate q: "
                f"{missing_keywords}"
            )

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
        # do image rotation but keep track of the steps taken
        before_rotation = self._ccw_rotation_counter
        super().rotate_image(degrees, direction=direction)
        after_rotation = self._ccw_rotation_counter
        ccw_steps = after_rotation - before_rotation

        # update scattering vectors and beam center if they exist
        qdy_before = np.copy(self.qdy)
        qdx_before = np.copy(self.qdx)
        try:
            center_px_before = self.metadata['center_px']
        except KeyError:
            center_px_before = None

        if ccw_steps == 1 or ccw_steps == -3:
            if qdy_before is not None and qdx_before is not None:
                self.qdy = -1*np.flip(qdx_before)
                self.qdx = qdy_before
            if center_px_before:
                self.metadata['center_px'] = [
                    len(qdx_before)-center_px_before[1]-1,
                    center_px_before[0]]

        if ccw_steps == 2 or ccw_steps == -2:
            if qdy_before is not None and qdx_before is not None:
                self.qdy = -1*np.flip(qdy_before)
                self.qdx = -1*np.flip(qdx_before)
            if center_px_before:
                self.metadata['center_px'] = [
                    len(qdy_before)-center_px_before[0]-1,
                    len(qdx_before)-center_px_before[1]-1]

        if ccw_steps == 3 or ccw_steps == -1:
            if qdy_before is not None and qdx_before is not None:
                self.qdy = qdx_before
                self.qdx = -1*np.flip(qdy_before)
            if center_px_before:
                self.metadata['center_px'] = [
                    center_px_before[1],
                    len(qdy_before)-center_px_before[0]-1]

    def reset_rotations(self):
        """
        Return the scattering image to its original orientation
        removing any rotations that may have been done. The beam center
        and scattering vectors will be tracked and updated through this
        process (if they exist).
        """
        if self._ccw_rotation_counter != 0:
            degrees = -90*self._ccw_rotation_counter
            self.rotate_image(degrees)

    def integrate_box(
        self,
        limits_qdy_px: list | tuple,
        limits_qdx_px: list | tuple,
        mode: str,
        axis: str | int,
        show_plot=False,
        log_scale=True,
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
            axis=axis
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
            axis=axis
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
            axis=axis
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
            log_scale=log_scale
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
