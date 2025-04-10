"""
This module contains the following classes:

Data2D : Generic two-dimensional data class not tied to diffraction.
DataQdyQdx(Data2D) : Child class of Data2D for detector images.
Dataset : Container class for instances of DataQdyQdx making up a single
    CDSAXS measurement for a sample (e.g. single theta scan).
Data1D : General one-dimensional data classes for intensity vs. q.
IntegratedQSlice(Data1D) : Child class of Data1D for spectra
    extracted from integration across a defined area of an image.

"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import linregress

import cdsaxs.calculators as calculators
from cdsaxs.data1d import IntegratedQSlice
from cdsaxs.metadata import METADATA_KEYWORDS
import cdsaxs.plotting as plotting
from cdsaxs.sample import Sample
from cdsaxs_gui_legacy import diffraction
import cdsaxs._plotting_tools as plotting_tools

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
            self.rotate_image(degrees)

    def integrate_box(
            self,
            limits_axis0,
            limits_axis1,
            mode,
            axis,
    ):
        """
        Simple integration in a box defined by the [min, max) limits
        for each axis.

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
        """
        if mode == 'sum':
            integrated_i = np.nansum(
                self.image[limits_axis0[0]:limits_axis0[1],
                           limits_axis1[0]:limits_axis1[1]],
                axis=axis
            )
        elif mode == 'mean':
            integrated_i = np.nanmean(
                self.image[limits_axis0[0]:limits_axis0[1],
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
            -1*np.arange(0, self.image.shape[0]) + self.metadata['center_px'][0],
            self.metadata["wavelength_nm"],
            self.metadata["pixel_size_um"],
            self.metadata["sdd_cm"],
        )
        qdx = diffraction.qxz_pixels_to_qxz(
            -1*np.arange(0, self.image.shape[1]) + self.metadata['center_px'][1],
            self.metadata["wavelength_nm"],
            self.metadata["pixel_size_um"],
            self.metadata["sdd_cm"],
        )
        self.qdy = qdy
        self.qdx = qdx

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
            if qdy_before and qdx_before:
                self.qdy = -1*np.flip(qdx_before)
                self.qdx = qdy_before
            if center_px_before:
                self.metadata['center_px'] = [
                    len(qdx_before)-center_px_before[1]-1,
                    center_px_before[0]]

        if ccw_steps == 2 or ccw_steps == -2:
            if qdy_before and qdx_before:
                self.qdy = -1*np.flip(qdy_before)
                self.qdx = -1*np.flip(qdx_before)
            if center_px_before:
                self.metadata['center_px'] = [
                    len(qdy_before)-center_px_before[0]-1,
                    len(qdx_before)-center_px_before[1]-1]

        if ccw_steps == 3 or ccw_steps == -1:
            if qdy_before and qdx_before:
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

        if "energy_ev" in metadata.keys() and "wavelength_nm" in metadata.keys():
            raise ValueError(
                "You have specified both the source energy and wavelength. "
                "Only one of these can be specified and the other is "
                "calculated. To avoid over-specifying or conflicting values, "
                "please only use one of these values. "
            )
        return True

    def integrate_box(
            self,
            limits_qdy_px,
            limits_qdx_px,
            mode,
            axis,
    ):
        """
        Integrate a box defined by indexing limits.
        axis : str
            Define the axis to integrate over, either qdy or qdx.
        """
        integrated_i, params = super().integrate_box(
            limits_axis0=limits_qdy_px,
            limits_axis1=limits_qdx_px,
            mode=mode,
            axis=0 if axis == 'qdy' else 1
        )
        q = self.qdx[limits_qdx_px[0]:limits_qdx_px[1]]\
            if axis == 'qdy'\
            else self.qdy[limits_qdy_px[0]:limits_qdy_px[1]]

        integrated_q_slice = IntegratedQSlice(
            q=q,
            I=integrated_i,
            q_axis='qdx' if axis == 'qdy' else 'qdy',
            name=self.name,
            limits_axis0=params["limits_axis0"],
            limits_axis1=params["limits_axis1"],
            mode=mode,
            integration_axis=params["axis"]
        )

        return integrated_q_slice

    def integrate_box_of_size(
            self,
            size_qdy_px,
            size_qdx_px,
            mode,
            axis,
            offset_qdy_px=0,
            offset_qdx_px=0,
    ):
        # ""
        # Integrate a box defined by its size and offset from a defined
        # centerpoint.

        # trim : bool
        # If trim is set to True, only the box that overlays the image
        # will be returned. If set to False, the areas that fall off the
        # image will be filled with NAN.

        # """
        # min0 = center_px[0] - int(size0/2) - offset0
        # max0 = min0 + size0

        # min1 = center_px[1] - int(size1/2) - offset1
        # max1 = min1 + size1

        # min0_im = max(min0, 0)
        # min1_im = max(min1, 0)
        # max0_im = min(max0, self.image.shape[0]-1)
        # max1_im = min(max1, self.image.shape[1]-1)

        # integrated_i_im = self.integrate_box(
        #                  limits_axis0=(min0_im, min0_im),
        #                  limits_axis1=(min1_im, max1_im),
        #                  mode=mode, axis=axis
        #              )
        # if trim:
        #     return integrated_i_im
        # else:
        #     integrated_i = np.empty(size0 if axis == 1 else size1)
        #     integrated_i[:] = np.nan
        #     if axis == 0:
        #         integrated_i[min1_im-min1:max1_im-min1] = integrated_i_im[0]
        #     else:
        #         integrated_i[min0_im-min0:max0_im-min0] = integrated_i_im[0]
        #     return integrated_i.reshape(-1), integrated_i_im[1]  # params
        
        """
        Integrate a box defined by its size and offset from a the
        defined beam center.

        axis : str
            Define the axis to integrate over, either qdy or qdx.

        """

        # figure out where the box lies with respect to beam center
        center0, center1 = self.metadata['center_px']
        min0 = center0 - int(size_qdy_px/2) - offset_qdy_px
        max0 = min0 + size_qdy_px
        min1 = center1 - int(size_qdx_px/2) - offset_qdx_px
        max1 = min1 + size_qdx_px

        # make sure the box isn't falling off the image
        min0 = max(min0, 0)
        min1 = max(min1, 0)
        max0 = min(max0, self.image.shape[0]-1)
        max1 = min(max1, self.image.shape[1]-1)

        # integrate the box area of image
        integrated_q_slice = self.integrate_box(
            limits_qdy_px=[min0, max0],
            limits_qdx_px=[min1, max1],
            mode=mode,
            axis=axis
        )

        return integrated_q_slice

    def integrate_box_of_q_range(
            self,
            range_qdy,
            range_qdx,
            mode,
            axis
    ):
        """
        Integrate using q ranges along both axes (half open).
        """

        # fix the min, max order if the user provided them reversed
        range_qdy = [min(range_qdy), max(range_qdy)]
        range_qdx = [min(range_qdx), max(range_qdx)]

        qdy_indices = np.where((self.qdy >= range_qdy[0])
                               & (self.qdy < range_qdy[1]))[0]
        limits_axis0 = (np.min(qdy_indices), np.max(qdy_indices)+1)

        qdx_indices = np.where((self.qdx >= range_qdx[0])
                               & (self.qdx < range_qdx[1]))[0]
        limits_axis1 = (np.min(qdx_indices), np.max(qdx_indices)+1)

        integrated_q_slice = self.integrate_box(
            limits_axis0,s
            limits_axis1,
            mode=mode,
            axis=axis
        )

        return integrated_q_slice

    def find_peaks1D(self, box_mode, box_params: dict, peak_params: dict):
        """
        Simple peak finding function in 1D to determine appropriate
        rotation angle of the sample coordinate system in the x-y
        detector plane.

        The box used to search for peaks is defined in the same way as
        the integrator methods. This method assumes that there is only
        a one-dimensional line of peaks along the axis not defined as
        the integration axis in box_params.

        Parameters
        ----------
        box_mode : str
            Type of integration box to use. Options are:
                'box' : indcates use of DataQdyQdx.integrate_box
                'box_size' : indicates use of DataQdyQdx.integrate_box_of_size
                'q_range' : indicates use of DataQdyQdx.integrate_box_of_q_range
        box_params : dict
            Dictionary of keyword arguments for the selected integration
            method (box_mode).
        peak_params : dict
            Dictionary of keyword arguments for the scipy.find_peaks
            algorithm; see scipy documentation for more information.

        Returns
        -------
        list[tuple]
            List of peak positions in (qdy, qdx) coordinates.
        float
            Angle of rotation of best line fit to the peaks counterclockwise
            from the qdx axis.
        tuple[float, float]
            Results from linear fit to the peaks of (slope, intercept).
        """

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

        peaks, params = find_peaks(**peak_params)

        min0, max0 = integrated_q_slice.limits_axis0
        min1, max1 = integrated_q_slice.limits_axis1
        box_image = self.image[min0:max0, min1:max1]

        if box_params['axis'] == 0 or box_params['axis'] == 'qdy':
            peaks_other = np.argmax(box_image[:, peaks], axis=0)
            peak_coords = [(y, x) for y, x in zip(peaks_other, peaks)]
        else:
            peaks_other = np.argmax(box_image[peaks, :], axis=1)
            peak_coords = [(y, x) for y, x in zip(peaks, peaks_other)]

        peak_coords_array = np.array(peak_coords)
        fit = linregress(peak_coords_array[:, 1], peak_coords_array[:, 0])
        angle = np.arctan(fit.slope)

        return peak_coords, angle, (fit.slope, fit.intercept)

    def plot_data(self, show_pixels=False):

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
            title=self.name
        )

        return fig


class Dataset():
    """
    A container class for a set of DataQdyQdx instances that make up a
    single CD-SAXS measurement for a sample.

    Attributes
    ----------
    datas : dict
        Dictionary containing the DataQdyQdx objects. The key for each
        instance is DataQdyQdx.name. Be cautious if the name attribute
        was kept as default (filename) as it may result in non-unique
        keys. If you are pulling all data from the same data directory,
        however, this will not be a problem.
    name : str
        Custom name of the dataset.
    sample : Sample
        Instance of the Sample class that details the sample measured
        when collecting the dataset. Information such as sample
        thickness and attenuation coefficients should be added here to
        enable the relevant data corrections.

    Optional Attributes
    -------------------
    integrated_datasets : list
        List of dictionaries containing integrated data. For each
        ditionary (dataset), the keys align with the datas.keys() and
        the values are instances of IntegratedDataSlices. These
        dictionaries are produced by the cdsaxs integrators.

    """

    def __init__(
            self,
            datas: list[DataQdyQdx] = None,
            name: str = None,
            sample: str = None
    ):
        if datas:
            self.add_data(datas)
        else:
            self.datas = {}

        self.name = name
        self.sample = sample
        self.integrated_datasets = None

    def add_data(self, datas: DataQdyQdx | list[DataQdyQdx]):
        """Add one or more DataQdyQdx instances to the dataset."""
        datas = [datas] if isinstance(datas, DataQdyQdx) else datas
        for data in datas:
            if data.name in self.datas.keys():
                raise ValueError(
                    "You do not have unique names for DataQdyQdx instances."
                )
            else:
                self.datas[data.name] = data

    def remove_data(self, datas: DataQdyQdx | list[DataQdyQdx]):
        """Remove one or more DataQdyQdx instances from the dataset."""
        datas = [datas] if isinstance(datas, DataQdyQdx) else datas
        for data in datas:
            try:
                del self.datas[data.name]
            except KeyError:
                warnings.warn(f"Could not delete {data.name} data as it was"
                              "not part of the dataset.")

    def assign_sample(self, sample: Sample):
        """
        Assign the measured sample with an instance of the Sample class.
        """
        # TODO: implement required sample checks
        self.sample = sample

    def update_all_metadata(self, metadata: dict, overwrite: bool = True):
        """
        Add or update metadata for all DataQdyQdx stored in this Dataset.
        Existing metadata parameters can be updated by keeping the
        overwrite argument to True.

        Parameters
        ----------
        metadata : dict
            Key : value pairs of accepted metadata (key) and their
            values. See DataQdyQdx class docstring for list of accepted
            keywords.
        overwrite : bool
            If set to True, any metadata provided to this method will
            overwrite the existing value in the instance if it already
            exists in self.metadata.
            Default value is True.
        """

        for data in self.datas.values():
            data.update_metadata(metadata=metadata, overwrite=overwrite)

    def update_user_params_for_all(self, params: dict, overwrite: bool = True):
        """
        Add key: value pairs to the user params for all DataQdyQdx.
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

    def plot_datas(self, keys='ALL'):
        """
        Plot one or more DataQdyQdx in the dataset.

        Parameters
        ----------
        keys : list | str
            If set to 'ALL', a list of all figures for all datas will
            be returned. Otherwise, it can be a single data key or
            a list of data keys to plot.
        """

        figs = []

        if keys == 'ALL':
            keys = list(self.datas.keys())
        elif isinstance(keys, str):
            keys = [keys]
        else:
            pass

        for key in keys:
            figs.append(self.datas[key].plot_data())

        return figs

