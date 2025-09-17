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
from plotly.offline import iplot

import cdsaxs.calculators as calculators
from cdsaxs.data1d import QSlice
from cdsaxs.metadata import METADATA_KEYWORDS, check_metadata
import cdsaxs.plotting as plotting
from cdsaxs.tools import line_fit, find_peaks_2D_legacy, rotate_image
from cdsaxs.tools import find_peaks_2D, find_peaks_1D, find_peaks_2D_one_axis
import cdsaxs.diffraction as diffraction


# any changes to these metadata values should update calculated q values
UPDATE_Q_TRIGGERS = [
    "energy_ev", "wavelength_nm", "sdd_cm", "pixel_size_um", "center_px",
    "detector_phi_deg", "detector_phi_omega"
]


def _default_mask(image: NDArray):

    """
    Generate a default mask of points that are nan, inf, or -inf.
    """
    mask = np.isnan(image)
    mask += np.isinf(image)
    mask += np.isneginf(image)

    return mask


def combine_dataqdyqdx(*dataqdyqdx: DataQdyQdx, name=None):
    """
    Combine two or more instances of DataQdyQdx into a single instance
    of DataQdyQdx. This operation is not sensitive to any data
    transformations or orientation changes that have been performed and
    so the user should carefully consider when to perform this operation.

    The image intensities will be summed together. Any points that were
    masked in one or more of the data instances will be masked in the
    new combined instance. Any nan, inf, or -inf points will also
    be masked prior to this operation to ensure no unexpected behavior
    of nansum arises (e.g., sum of all nan values resulting in 0 for a
    single pixel).

    The user is responsible for ensuring that the scattering images
    can be summed together at their measurement conditions, including
    wavelength, sample to detector distance, sample configuration, etc.
    By default, the metadata will be transferred from the first data
    instance provided. The exception is exposure_time_s which will
    be summed across the instances to accurately reflect the total
    measurement time. If only some of the instances have exposure time
    in the metadata dictionary, this could give an artifically low value
    for exposure time.

    Parameters
    ----------
    *dataqdyqdx : DataQdyQdx
        Any number of DataQdyQdx instances can be passed to this
        function and summed together.
    """

    # initialize information from the first 2d data instance
    first_data = dataqdyqdx[0]
    metadata = first_data.metadata
    user_params = first_data.user_params
    mask = first_data.mask
    image = first_data.image
    if name is None:
        name = first_data.name

    for data in dataqdyqdx[1:]:
        if 'exposure_time_s' in data.metadata.keys():
            metadata['exposure_time_s'] += data.metadata['exposure_time_s']
        mask += data.mask
        image = np.nansum(image, data.image)

    new_data = DataQdyQdx(
        image=image,
        name=name,
        mask=mask,
        **metadata,
        **user_params,
    )

    return new_data


class Data2D():
    """
    Protected Attributes
    --------------------
    _data_transformations : list of tuples
        Will keep track of intensity data transformations, including
        a normalization, scaling, adding or subtracting by or of a
        specified value. Each item in the list is a tuple of
        (transformation, value) where transformation can be:
            normalize
            scale
            add
            subtract
        and where value can either be a single float or an array of
        floats with the same dimensions as the image.
    """

    def __init__(self, image: NDArray[np.floating],
                 mask: NDArray[np.bool] = None):
        """
        Generic 2D data class with basic image functionalities. This
        class is not tied to any diffraction information.

        Attributes
        ----------
        image : NDArray
            Two-dimensional array containing the image as pixel
            intensities. The first dimension corresponds to image rows
            from top to bottom and the second dimension corresponds to
            image columns from left to right.
        mask : NDArray
            Two-dimensional boolean array of same dimensions as image
            that are True at pixel values that should be masked out
            for all operations.
            All pixels that are nan, inf, or -inf will be masked by
            default.
        """
        self.image = image
        self._raw_image = np.copy(self.image)
        self.mask = _default_mask(self.image)
        if mask is not None:
            self.mask += mask  # apply user-provided mask
        self._data_transformations = []

    def mask_points(self, mask):
        """
        Add points to the data mask. This will not unmask any previously
        masked points in the image.

        Parameters
        ----------
        mask : NDArray
            Two-dimensional boolean array of same dimensions as the
            data image. Pixels that are True will be masked out for
            all data operations. This will NOT unmask any previously
            masked points.
        """
        self.mask += mask

    def overwrite_mask(self, mask):
        """
        Set a new mask for the data. This will unmask all previously
        masked points and only mask the points provided to this
        function call.

        Parameters
        ----------
        mask : NDArray
            Two-dimensional boolean array of same dimensions as the
            data image. Pixels that are True will be masked out for
            all data operations. This will unmask any previously
            masked points.
        """
        self.mask = self.mask*False + mask

    def reset_mask(self):
        """
        Reset the mask to only mask out pixels with values of nan, inf,
        or -inf.
        """
        self.mask = np.isnan(self.image)  # mask out nan
        self.mask += np.isinf(self.image) + np.isneginf(self.image)  # mask inf

    def rotate_image_ccw(self, steps=1):
        """
        Rotate the image counterclockwise by 90 degree, or by a
        specified number of 90 degree steps.

        The original image can be recalled by using
        'reset_image_orientation'.

        Parameters
        ----------
        steps : int
            Number of 90 degree rotations to be performed. If a negative
            value is provided, the rotations will be performed in the
            clockwise direction.
        """

        k = int(np.round(steps, 0))

        # determine counterclockwise steps to achieve same rotation
        while k < 0:
            k += 4
        if k != 0:
            self.image = np.rot90(self.image, k=k, axes=(0, 1))
            self.mask = np.rot90(self.mask, k=k, axes=(0, 1))

    def flip_horizontally(self):

        self.image = np.flip(self.image, axis=1)
        self.mask = np.flip(self.mask, axis=1)

    def flip_vertically(self):

        self.image = np.flip(self.image, axis=0)
        self.mask = np.flip(self.mask, axis=0)

    def scale_data(self, value):
        """
        Scale the data by the specified value or array of values
        that match the dimensions of the data image.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.image.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")

        self.image = self.image*value
        self._data_transformations.append(("scale", value))

    def normalize_data(self, value):
        """
        Scale the data by the recipricol of the specified value.or array
        of values that match the dimensions of the data image.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
            value_r = 1/value
        else:
            value_r = np.reciprocal(value)
            if value_r.shape != self.image.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")

        self.image = self.image*value_r
        self._data_transformations.append(("normalize", value))

    def subtract_from_data(self, value):
        """
        Subtract a specified single value or an array of values that
        matches the image dimensions from the image data.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.image.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")
        self.image = self.image-value
        self._data_transformations.append(("subtract", value))

    def add_to_data(self, value):
        """
        Add a specified single value or an array of values that
        matches the image dimensions to the image data.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.image.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")
        self.image = self.image+value
        self._data_transformations.append(("add", value))

    def reset_intensity(self):
        """
        Resets any normailzation, scaling, added or subtracted values
        applied to the image data intensity.
        """
        for transform, value in reversed(self._data_transformations):
            if transform == "add":
                self.subtract_from_data(value)
            elif transform == "subtract":
                self.add_to_data(value)
            elif transform == "normalize":
                self.scale_data(value)
            elif transform == "scale":
                self.normalize_data(value)
        self._data_transformations = []

    def reset_image(self):
        """
        Resets the image to the raw image and also resets the applied
        mask. This will undo any orientation transformations to the
        image as well as any scaling, normalization, additions or
        subtractions applied to the image.
        """
        self._data_transformations = []
        self.image = np.copy(self._raw_image)
        self.reset_mask()

    def sum_box(
            self,
            limits_axis0,
            limits_axis1,
            axis,
    ):
        """
        Select a region of interest (box shape) and sum over the
        selected axis (or axes).

        Be careful if you have any pixels with a value of nan that are
        not masked by the mask attribute. If the summation algorithm
        encounters all nan values, it will return 0 rather than nan. By
        default, all nan values are masked out unless the user
        overwrites this behavior.

        Parameters
        ----------
        limits_axis0: tuple[int, int]
            Defines the limits (indices) of the box in the first
            dimension. This is a half open range [min, max).
        limits_axis1 : tuple[int, int]
            Defines the limits (indices) of the box in the second
            dimension. This is a half open range [min, max).
        axis : int | tuple
            The axis or axes over which to perform the sum.
            Setting axis to 0 will sum over each row.
            Setting axis to 1 will sum over each column.
            Setting axis to (0, 1) will sum over all axes and return a
            single value.

        Returns
        -------
        ndarray
            Intensity of the selected box summed over the selected axis
            or axes.
        ndarray
            The two dimensional box selected from the image data used
            in the summation.
        """
        image_box = self.image[limits_axis0, limits_axis1]
        sum_intensity = np.nansum(
            image_box,
            axis=axis,
            where=~self.mask
        )

        return sum_intensity.reshape(-1), image_box

    def mean_box(
            self,
            limits_axis0,
            limits_axis1,
            axis,
    ):
        """
        Select a region of interest (box shape) and perform an
        arithmetic mean over the selected axis (or axes).

        Parameters
        ----------
        limits_axis0: tuple[int, int]
            Defines the limits (indices) of the box in the first
            dimension. This is a half open range [min, max).
        limits_axis1 : tuple[int, int]
            Defines the limits (indices) of the box in the second
            dimension. This is a half open range [min, max).
        axis : int | tuple
            The axis or axes over which to perform the mean.
            Setting axis to 0 will average over each row.
            Setting axis to 1 will average over each column.
            Setting axis to (0, 1) will average over all axes and return
            a single value.

        Returns
        -------
        ndarray
            Intensity of the selected box averaged over the selected
            axis or axes.
        ndarray
            The two dimensional box selected from the image data used
            in the summation.
        """
        image_box = self.image[limits_axis0, limits_axis1]
        mean_intensity = np.nanmean(
            image_box,
            axis=axis,
            where=~self.mask
        )

        return mean_intensity.reshape(-1), image_box

    def slice_box(
            self,
            limits_axis0,
            limits_axis1,
            axis,
            mode,
    ):
        """
        Select a region of interest (box shape) and perform an
        arithmetic mean or sum over the selected axis (or axes).

        Parameters
        ----------
        limits_axis0: tuple[int, int]
            Defines the limits (indices) of the box in the first
            dimension. This is a half open range [min, max).
        limits_axis1 : tuple[int, int]
            Defines the limits (indices) of the box in the second
            dimension. This is a half open range [min, max).
        axis : int | tuple
            The axis or axes over which to perform the mean or sum.
            Setting axis to 0 will average/sum over each row.
            Setting axis to 1 will average/sum over each column.
            Setting axis to (0, 1) will average/sum over all axes and return
            a single value.
        mode : str
            Select whether to perform a 'mean' or 'sum'.

        Returns
        -------
        ndarray
            Intensity of the selected box averaged over the selected
            axis or axes.
        ndarray
            The two dimensional box selected from the image data used
            in the summation.
        """

        if mode == 'sum':
            slice_i, slice_box = self.sum_box(
                limits_axis0=limits_axis0,
                limits_axis1=limits_axis1,
                axis=axis
            )
        elif mode == 'mean':
            slice_i, slice_box = self.mean_box(
                limits_axis0=limits_axis0,
                limits_axis1=limits_axis1,
                axis=axis
            )
        else:
            raise ValueError(
                f"Did not recognize slice mode {mode}. Use 'sum' or 'mean'."
            )

        return slice_i, slice_box

    def rotate_image(self,
                     rotation_angle_deg,
                     rotation_center=(0, 0),
                     resampling_mode="bicubic"):

        """
        Rotate the image counterclockwise by the specified angle about
        the rotation center.
        NOTE: This operation will convert any masked points in your
        array to nan prior to the image rotation so they are not used
        in the resampling algorithms. The mask will then be reset to
        mask out any nan pixels after the rotation.

        Parameters
        ----------
        rotation_angle_deg : float
            Angle in degrees by which to rotate the image
            counterclockwise.
        rotation_center : tuple
            Center of rotatation.
            Default is the upper left pixel.
        resampling_mode: str
            Set the resampling method used during the rotation.
            The box rotation works by rotating the image underneath then
            extracting the box for integration. Resampling of the
            image intensities can be performed with the 'nearest',
            'bilinear', or 'bicubic' methods in the PILLOW package.
            Default value is 'bicubic'.
        """

        self.image[self.mask] = np.nan
        self.image = rotate_image(self.image,
                                  degrees=rotation_angle_deg,
                                  rotation_center=rotation_center,
                                  resampling_mode=resampling_mode)
        self.reset_mask()


class DataQdyQdx(Data2D):
    """
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
        data workflow. These are not recognized and therefore, not
        accessed by the cdsaxs package in any built-in operations. They
        can, however, be used for scaling or normalization of the data
        if the user requests them in the function call.
    name : str
        Identifier for this image acquisition. The default when using
        the cdsaxs loaders is the filename, but be cautious when
        creating a Dataset as the filenames alone may not always result
        in unique identifiers for each image. A custom name can be set
        by passing 'name' metadata. This name is used as a key in the
        dictionaries storing the data objects within the dataset class.
    """

    def __init__(
            self,
            image: NDArray[np.floating],
            name: str = None,
            mask: NDArray[np.bool] = None,
            **kwargs
    ):
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

        Parameters
        ----------
        image : NDarray
            Scattering image as a two-dimensional numpy array. The first
            dimension corresponds to the y-axis (detector frame) and the
            second dimension corresponds to the x-axis (detector frame).
        name : str
            Identifier for this image acquisition. The default when using
            the cdsaxs loaders is the filename, but be cautious when
            creating a Dataset as the filenames alone may not always result
            in unique identifiers for each image. A custom name can be set
            by passing 'name' metadata. This name is used as a key in the
            dictionaries storing the data objects within the dataset class.
        mask : NDArray
            Two-dimensional boolean array of same dimensions as image
            that are True at pixel values that should be masked out
            for all operations.
            All pixels that are nan, inf, or -inf will be masked by
            default.

        Other Parameters
        ----------------
        **kwargs
            Relevant scattering metadata to the image acquisition can
            be passed as additional keyword argument. Any keywords
            recognized as metadata by the cdsaxs code will be saved in
            the metadata attribute. The remaining information will be
            stored in the user_params dictionary.
        """

        # run base class init
        super().__init__(image=image, mask=mask)

        self.metadata = {}
        self.update_metadata(
            {x: y for x, y in kwargs.items() if x in METADATA_KEYWORDS})

        self.user_params = {}
        self.update_user_params(
            {x: y for x, y in kwargs.items() if x not in METADATA_KEYWORDS}
        )

        self.qdy = None
        self.qdx = None

        # calculate the q vectors if all required metadata is present
        try:
            self.calculate_q(suppress_errors=False)
        except ValueError as e:
            print(f"WARNING: insufficient metadata for q calculation:\n{e}")

        self.name = name if name is not None else\
            self.metadata['name'] if 'name' in self.metadata.keys() else\
            self.metadata['filename'] if 'filename' in self.metadata.keys()\
            else 'name'

        # set required default metadata values not required by user
        self.update_metadata({'sample_phi_offset_deg': 0}, overwrite=False)
        self.update_metadata({'center_px': (0, 0)}, overwrite=False)

        self.data_transformations = []

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
        if check_metadata(metadata):
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
                try:
                    self.calculate_q(suppress_errors=False)
                except ValueError as e:
                    print(f"WARNING: insufficient metadata for q calculation:\n{e}")

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

    def calculate_q(self,
                    suppress_errors: bool = False):
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

    def scale_data(self, value, keyword=None):
        """
        Scale the data by the specified value or array of values
        that match the dimensions of the data image.
        """
        super().scale_data(value)
        self.data_transformations.append(
            ("scale", value, keyword))

    def normalize_data(self, value, keyword=None):
        """
        Scale the data by the recipricol of the specified value.or array
        of values that match the dimensions of the data image.
        """

        super().normalize_data(value)
        self.data_transformations.append(
            ("normalize", value, keyword))

    def subtract_from_data(self, value, keyword=None):
        """
        Subtract a specified single value or an array of values that
        matches the image dimensions from the image data.
        """
        super().subtract_from_data(value)
        self.data_transformations.append(
            ("subtract", value, keyword))

    def add_to_data(self, value, keyword=None):
        """
        Add a specified single value or an array of values that
        matches the image dimensions to the image data.
        """
        super().add_to_data(value)
        self.data_transformations.append(
            ("add", value, keyword))

    def normalize_by_metadata(self, normalize_by):
        """
        Normalize (divide) the image by the selected metadata or user
        parameters. This function will check if the normalization was
        already performed using any of the requested metadata. If so,
        this function will skip that normalization as to not 'double up'
        on the requested transformation.

        If the user has updated any metadata values and would like to
        apply the new value to the data, the user should reset the
        data intensity transformations and perform them again in the
        required order. This is because these transformation functions
        do not account for order of operations.

        Parameters
        ----------
        normalize_by : list, str
            List of accepted metadata keywords or user parameter keys
            that should be used to normalize the data. A single string
            keyword can also be provided.
        """
        if type(normalize_by) is str:
            normalize_by = [normalize_by]
        elif type(normalize_by) is not list:
            raise ValueError(
                "The 'normalize_by' argument should be a single string "
                "keyword or a list of string keywords, not type "
                f"{type(normalize_by)}.")

        for key in normalize_by:
            if type(key) is float or type(key) is int:
                value = float(key)
            elif key in METADATA_KEYWORDS:
                value = self.metadata[key]
            elif key in self.user_params.keys():
                value = float(self.user_params[key])
            else:
                warnings.warn(f"Did not recognize {key} as an available"
                              "parameter in either metadata or user_params.")
                value = None
            if type(key) is str:
                for transform, _, keyword in self.data_transformations:
                    if key == keyword and transform == "normalize":
                        warnings.warn(
                            f"{key} was already used in a normalization data "
                            "transformation. Skipping for now.")
                        value = None
            if value is not None:
                self.normalize_data(value, keyword=key)

    def scale_by_metadata(self, scale_by):
        """
        Scale (multiple) the image by the selected metadata or user
        parameters. This function will check if the normalization was
        already performed using any of the requested metadata. If so,
        this function will skip that scaling as to not 'double up' on
        the requested transformation.

        If the user has updated any metadata values and would like to
        apply the new value to the data, the user should reset the
        data intensity transformations and perform them again in the
        required order. This is because these transformation functions
        do not account for order of operations.

        Parameters
        ----------
        scale_by : list, str
            List of accepted metadata keywords or user parameter keys
            that should be used to scale the data. A single string
            keyword can also be provided.
        """
        if type(scale_by) is str:
            scale_by = [scale_by]
        elif type(scale_by) is not list:
            raise ValueError(
                "The 'scale_by' argument should be a single string "
                "keyword or a list of string keywords, not type "
                f"{type(scale_by)}.")

        for key in scale_by:
            if type(key) is float or type(key) is int:
                value = float(key)
            if key in METADATA_KEYWORDS:
                value = self.metadata[key]
            elif key in self.user_params.keys():
                value = float(self.user_params[key])
            else:
                warnings.warn(f"Did not recognize {key} as an available "
                              "parameter in either metadata or user_params.")
                value = None
            if type(key) is str:
                for transform, _, keyword in self.data_transformations:
                    if key == keyword and transform == "scale":
                        warnings.warn(
                            f"{key} was already used in a scaling data "
                            "transformation. Skipping for now."
                        )
                    value = None
            if value is not None:
                self.scale_data(value, keyword=key)

    def reset_intensity(self):
        """
        Resets any normailzation, scaling, added or subtracted values
        applied to the image data intensity values.
        """
        super().reset_intensity()
        self.data_transformations = []

    def rotate_image_ccw(self, steps):
        """
        Rotate the scattering image in 90 degree counterclockwise steps.
        The scattering vectors qdy and qdx as well as the beam center
        position in metadata(center_px) will be updated with the
        rotation.

        Parameters
        ----------
        steps : int
            Number of 90 degree steps to rotation the image in the
            counterclockwise direction.
        """
        # temporarily store information about current state
        length0 = self.image.shape[0]
        length1 = self.image.shape[1]
        center_px = (
            self.metadata['center_px'][0], self.metadata['center_px'][1])

        # do the rotation
        k = int(np.round(steps, 0))
        super().rotate_image_ccw(steps=k)

        if k == 1:
            if center_px is not None:
                self.metadata['center_px'] = (
                    length1-center_px[1]-1,
                    center_px[0])

        if k == 2:
            if center_px is not None:
                self.metadata['center_px'] = (
                    length0-center_px[0]-1,
                    length1-center_px[1]-1)

        if k == 3:
            if center_px is not None:
                self.metadata['center_px'] = (
                    center_px[1],
                    length0-center_px[0]-1)

        # recalcualte q if possible
        self.calculate_q(suppress_errors=True)

    def flip_horizontally(self):
        super().flip_horizontally()
        center_px = self.metadata['center_px']
        self.metadata['center_px'] = [
            center_px[0],
            self.image.shape[1] - center_px[1] - 1
        ]
        # recalcualte q if possible
        self.calculate_q(suppress_errors=True)

    def flip_vertically(self):
        super().flip_vertically()
        center_px = self.metadata['center_px']
        self.metadata['center_px'] = [
            self.image.shape[0] - center_px[0] - 1,
            center_px[1]
        ]
        # recalcualte q if possible
        self.calculate_q(suppress_errors=True)

    def reset_image(self):
        """
        Return the scattering image to its original orientation
        removing any rotations or flips that may have been performed.

        All transformations to the scattering intensity will also be
        undone, including scale, normalize, add and subtract functions.

        This function will reset the image to the raw image and the beam
        center will be reset back to default of (0, 0). The mask will
        be reset to the default conditions of masking any nan, inf, or
        -inf values.
        """
        self.update_metadata({'center_px': (0, 0)})
        super().reset_image()
        self.data_transformations = []

    def get_box_dims_size(self, size_qdy_px, size_qdx_px,
                          shift_box_qdy_px=0, shift_box_qdx_px=0):
        """
        Find the pixel index limits in half open ranges [min, max) that
        define a region of interest based on a box with a specific width
        along the two axes.

        Parameters
        ----------
        size_qdy_px : int
            Size of the box in pixels along qdy axis (axis 0)
        size_qdx_px : int
            Size of the box in pixels along qdx axis (axis 1)
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

        Returns
        -------
        tuple(int, int)
            Half open range along the qdy axis (axis 0).
        tuple(int, int)
            Half open range along the qdx axis (axis 0).
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

        return (min0, max0), (min1, max1)

    def get_box_dims_qrange(self, range_qdy, range_qdx):
        """
        Find the pixel index limits in half open ranges [min, max) that
        define a region of interest based on a box with set q ranges
        on both axes.

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

        return limits_qdy_px, limits_qdx_px

    def integrate_box(
        self,
        limits_qdy_px: list | tuple,
        limits_qdx_px: list | tuple,
        mode: str,
        axis: str | int,
        show_plot=False,
        log_scale=True,
        subtract_background_offset: int | list[int] = None,
        # interactive_plot=True
    ) -> QSlice:
        """
        Integrate a region of interest defined by the limits along both
        axes qdy and qdx (0 and 1, respectively).

        The limits along the qdy and qdx axes with respect to the
        pixel indices are required. The method get_box_dims can be
        used to retrieve the box dimensions based on a q-range or a
        specific box of size if desired. It returns the two axis limits
        in the correct format and can be fed directly into this function.

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
            Axis over which the summation or mean will be
            performed. Either the q name or numpy index can be provided
            here.
            If axis is set to 'qdy' or 0, this integration will be
            performed over all rows in each column and return I vs. qdx.
            If axis is set to 'qdx' or 1, this integration will be
            performed over all columns in each row and return I vs. qdy.
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
        subtract_background_offset: int, list[int], optional
            If set to a number of pixels greater than or equal to the
            width of the region of interest to be integrated over, a
            background subtraction will be performed by subtracting the
            intensity of an integrated box offset by the set number of
            pixels. If a single integer is provided, only one offset
            box will be used in the subtraction. If multiple are
            provided, the average signal from mulitple integrated offset
            boxes will be used in the subtraction.
            Note that the integration mode for these boxes will align
            with the selected mode for this integration function.
        interactive_plot : bool, optional
            If set to True, the plots returned will be interactive plots
            built via Plotly. If set to False, the plots returned will be
            static matplotlib figures.
            TODO: currently this is disabled and only True is accepted.
            Default value is True.

        Returns
        -------
        QSlice
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
        integrated_i, image_box = super().slice_box(
            limits_axis0=limits_qdy_px,
            limits_axis1=limits_qdx_px,
            axis=axis,
            mode=mode,
        )

        # extract scattering vector for this integration
        if axis == 0:
            q = self.qdx[limits_qdx_px[0]:limits_qdx_px[1]]
            q_int = np.mean(self.qdy[limits_qdy_px[0]:limits_qdy_px[1]])
            dq_int = np.std(self.qdy[limits_qdy_px[0]:limits_qdy_px[1]])
            q_axis = 'qdx'
            q_int_axis = 'qdy'
        elif axis == 1:
            q = self.qdy[limits_qdy_px[0]:limits_qdy_px[1]]
            q_int = np.mean(self.qdx[limits_qdx_px[0]:limits_qdx_px[1]])
            dq_int = np.mean(self.qdx[limits_qdx_px[0]:limits_qdx_px[1]])
            q_axis = 'qdy'
            q_int_axis = 'qdx'

        # extract background intensity
        if subtract_background_offset is not None:
            backgrounds = []
            if type(subtract_background_offset) is int:
                subtract_background_offset = [subtract_background_offset]
            for offset in subtract_background_offset:
                if offset < image_box.shape[axis]:
                    warnings.warn(
                        f"A background subtraction offset of {offset} "
                        "is less than the integrated axis width and so"
                        "it will be skipped in the subtraction.")
                else:
                    limits_qdy_px_sub = (
                        limits_qdy_px[0] + (offset if axis == 0 else 0),
                        limits_qdy_px[1] + (offset if axis == 0 else 0)
                    )
                    limits_qdx_px_sub = (
                        limits_qdx_px[0] + (offset if axis == 1 else 0),
                        limits_qdy_px[1] + (offset if axis == 1 else 0)
                    )
                    background_i, _ = super().sum_box(
                        limits_axis0=limits_qdy_px_sub,
                        limits_axis1=limits_qdx_px_sub,
                        axis=axis
                    )
                    backgrounds.append(
                        [background_i, limits_qdy_px_sub, limits_qdx_px_sub])

            background_i_avg = np.array([
                background_i for background_i, _, _ in backgrounds])
            background_i_avg = np.nanmean(background_i_avg, axis=0)

            integrated_i -= background_i_avg
        else:
            background_i_avg = None
            backgrounds = None

        # create instance of QSlice to hold integration metadata
        integrated_q_slice = QSlice(
            q=q,
            Iq=integrated_i,
            q_axis=q_axis,
            name=self.name,
            limits_axis0=limits_qdy_px,
            limits_axis1=limits_qdx_px,
            mode=mode,
            axis=axis,
            image_box=image_box,
            q_int=q_int,
            dq_int=dq_int,
            q_int_axis=q_int_axis,
            background=background_i_avg
        )

        if show_plot:
            fig, fig_slice = plotting.plot_QdyQdx_integration(
                self,
                integrated_q_slice=integrated_q_slice,
                log_scale=log_scale,
                background_subtractions=backgrounds
            )
            iplot(fig)
            iplot(fig_slice)

        return integrated_q_slice

    def find_peaks2D(
            self, box_dims=None, log_scale=True, refinement_size=7, **kwargs):
        """
        Find peaks across a two-dimensional image or region of interest
        using the scikit-image.feature peak_local_max() function and
        then further refined with local Gaussian fits across the two
        axes. Refinement is required for more accurate peak positions as
        the peak_local_max() only retuns the nearest pixel.

        Parameters
        ----------
        box_dims : tuple[int, int], tuple[int, int]
            Tuples that define the bounds along axis 0 and axis 1 of the
            image, respectively, in pixel indices. The get_box_dims...
            methods can be used to determine these bounds based on a
            q-range or a specific box size.
            The ranges are half open intervals [min, max).
        log_scale : bool, optional
            If set to True, the image will be passed to the peak finding
            algorithm on a log sale of intensity. If set to False, the image
            will be sent to the peak finding algorithm with its original
            values.
            Default value is True.
        refinement_size : int
            Define the box size around the peaks in which to peform the
            Gaussian refinement.
            Default value is 7. Minimum value is 4.

        Other Parameters
        ----------------
        **kwargs
            The keyword arguments for scikit-image's peak_local_max()
            function can be passed through. Please refer to the scikit-image
            documentation for detailed information on the parameters.
            A brief list is provided here:
                min_distance
                threshold_abs
                threshold_rel
                exclude_border
                num_peaks
                footprint
                labels
                num_peaks_per_label
                p_norm
            The threshold_abs keyword will always be set to 0 if no other
            value is provided by the user. This is to account for the -inf
            values after the log transform of the image.

        Returns
        -------
        NDArray
            An n x 2 array of peak coordinate positions will be returned
            for n number of peaks found.
        NDArray
            An n x 2 array of peak coordinate positions in (qdy, qdx)
            will be returned for n number of peaks found. If the
            scattering vector has not yet been calculated, this will be
            None.
        """

        if box_dims is not None:
            (min0, max0), (min1, max1) = box_dims
        else:
            min0 = 0
            max0 = self.image.shape[0]
            min1 = 0
            max1 = self.image.shape[1]

        peaks = find_peaks_2D(
            self.image[min0:max0, min1:max1],
            log_scale=log_scale,
            refinement_size=refinement_size,
            **kwargs)

        if self.qdy is not None and self.qdx is not None:
            peaks_q = np.ones_like(peaks).astype(np.float64)

            sort_qdy = np.argsort(self.qdy)
            peaks_q[:, 0] = np.interp(
                peaks[:, 0], np.arange(0, len(self.qdy)), self.qdy[sort_qdy])

            sort_qdx = np.argsort(self.qdx)
            peaks_q[:, 1] = np.interp(
                peaks[:, 1], np.arange(0, len(self.qdx)), self.qdx[sort_qdx])
        else:
            peaks_q = None

        return peaks, peaks_q

    def find_peaks2D_one_axis(
            self, box_dims=None, peak_axis=None, integration_mode='sum',
            log_scale=True, refinement_size=7, algorithm='scikit', **kwargs):
        """
        Find peaks along one axis of a two-dimensional image using
        the scikit-image.feature peak_local_max() function. The peaks
        are further refined with local Gaussian fits across the two axes
        at the peak location. Refinement is required for more accurate
        peak positions as the peak_local_max() only returns the positions
        to the nearest pixel.

        The old version of this function used scipy.signal find_peaks()
        to determine the intiial peak position. It is possible to use
        this algorithm by siwtching the algorithm keyword argument to
        'scipy'.

        This function differs from find_peaks_2D() in that it only
        allows for the primary peaks to be found along a single axis.
        For example, if axis 1 is the peak axis, the image provided will
        be integrated along axis 0 (summed or averaged) to find the
        primary peak location along axis 1. Then the peak location in
        axis 0 will be determined as the highest intensity pixel at each
        peak location along axis 1. This is then refined by the Gaussian
        fits. This function will assume that the peak axis is the
        axis with the longest dimensions. If the region of interest
        is square, then this function will assume peak axis is 1 unless
        otherwise specified.

        Parameters
        ----------
        box_dims : tuple[int, int], tuple[int, int]
            Tuples that define the bounds along axis 0 and axis 1 of the
            image, respectively, in pixel indices. The get_box_dims...
            methods can be used to determine these bounds based on a
            q-range or a specific box size.
            The ranges are half open intervals [min, max).
        peak_axis : int, optional
            The axis along which the peaks are found. If axis 0 (qdy) is
            selected, the image will be integrated along axis 1 (qdx).
            If axis 1 (qdx) is selected, the image will be integrated
            along axis 0 (qdy).
            The peak_axis will default to the longer axis of the
            image or region of interest. If the axes are the same
            length, peak_axis will default to axis 1.
        integration_mode : str, optional
            Integration mode to be performed along the axis not set as
            peak_axis. Options are 'mean' and 'sum'.
            Default value is 'sum'.
        log_scale : bool, optional
            If set to True, the image will be passed to the peak finding
            algorithm on a log sale of intensity. If set to False, the image
            will be sent to the peak finding algorithm with its original
            values.
            Default value is True.
        refinement_size : int
            Define the box size around the peaks in which to peform the
            Gaussian refinement.
            Default value is 7.
        algorithm: str
            Specify which peak finding algorithm is used. Default value is
            'scikit' which uses scikit-image.feature peak_local_max() to
            locate the peaks. If set instead to 'scipy', the scipy.signal
            find_peaks() algorithm will be used instead.

        Other Parameters
        ----------------
        **kwargs
            The keyword arguments for the specified peak finding algorithm
            can be passed through.
            If using scikit-image's peak_local_max() function (algorithm set
            to 'scikit'), keyword arguments include:
                min_distance
                threshold_abs
                threshold_rel
                exclude_border
                num_peaks
                footprint
                labels
                num_peaks_per_label
                p_norm
            The threshold_abs keyword will always be set to 0 if no other
            value is provided by the user. This is to account for the -inf
            values after the log transform of the image.

            If using scipy's find_peaks() function (algorithm set to
            'scipy'), keyword arguments include:
                height
                threshold
                distance
                prominence
                width
                wlen
                rel_height
                pleateau_size

            Note that these argument lists are not always kept up to date
            and we encourage the user to reference the scikit-image or
            scipy documentation directly.

        Returns
        -------
        NDArray
            An n x 2 array of peak coordinate positions will be returned for
            n number of peaks found.
        NDArray
            An n x 2 array of peak coordinate positions in (qdy, qdx)
            will be returned for n number of peaks found. If the
            scattering vector has not yet been calculated, this will be
            None.
        """

        if box_dims is not None:
            (min0, max0), (min1, max1) = box_dims
        else:
            min0 = 0
            max0 = self.image.shape[0]
            min1 = 0
            max1 = self.image.shape[1]

        if peak_axis is None:
            if (max0 - min0) > (max1 - min1):
                peak_axis = 0
            else:
                peak_axis = 1

        peaks = find_peaks_2D_one_axis(
            self.image[min0:max0, min1:max1],
            peak_axis=peak_axis,
            integration_mode=integration_mode,
            log_scale=log_scale,
            refinement_size=refinement_size,
            algorithm=algorithm,
            **kwargs)

        if self.qdy is not None and self.qdx is not None:
            peaks_q = np.ones_like(peaks).astype(np.float64)

            sort_qdy = np.argsort(self.qdy)
            peaks_q[:, 0] = np.interp(
                peaks[:, 0], np.arange(0, len(self.qdy)), self.qdy[sort_qdy])

            sort_qdx = np.argsort(self.qdx)
            peaks_q[:, 1] = np.interp(
                peaks[:, 1], np.arange(0, len(self.qdx)), self.qdx[sort_qdx])
        else:
            peaks_q = None

        return peaks, peaks_q

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
            size_qdy_px,
            size_qdx_px,
            update=True,
            beam_center_guess=None,
            peak_axis=None,
            **kwargs):
        """
        Attempt to locate the beam center position using the
        find_peaks2D_one_axis() method. Please refer to the method
        doc string for more information about the required arguments.

        For this method, the box dimensions are found internally for
        a box with specific widths along each axis centered around the
        starting beam center guess.

        The peaks located will need to be symmetric about the beam
        center and so this may require some careful consideration of
        the peak finding algorithm parameters.

        Parameters
        ----------

        size_qdy_px : int
            Box size in pixels along the qdy axis.
        size_qdx_px : int
            Box size in pixels along the qdx axis.
        update : bool
            If set to True, the found beam center will be updated in
            the data metadata as well as returned. If set to False,
            the center will only be returned and the data metadata will
            remain at the last beam center position.
        beam_center_guess : tuple[int, int], optional
            Initial guess of the beam center position in pixels along
            qdy and qdx (center_px_qdy, center_px_qdx). If provided,
            this method will update the current metadata so that the
            beam_center_guess is the beam_center_px.
            If no beam_center_guess is provided, this method will use
            the existing beam_center_px which will result in an error
            if it is not close enough to the actual center.

        Other Parameters
        ----------------
        **kwargs
            Any additional keyword arguments for the find_peaks2D_one_axis()
            method can be passed through to the underlying function.
            This includes:
                box_dims
                peak_axis
                integration_mode
                log_scale
                refinement_size
                algorithm
                any keyword arguments for the fitting algorithm

        Returns
        -------
        tuple[float, float]
            Beam center coordinates in units of pixels, [px_dy, px_dx].
        """

        if beam_center_guess is not None:
            self.update_metadata({'center_px': beam_center_guess},
                                 overwrite=True)
        box_dims = self.get_box_dims_size(
            size_qdy_px=size_qdy_px,
            size_qdx_px=size_qdx_px
        )

        (min0, max0), (min1, max1) = box_dims
        if peak_axis is None:
            if (max0 - min0) > (max1 - min1):
                peak_axis = 0
            else:
                peak_axis = 1
        peaks, peaks_q = self.find_peaks2D_one_axis(
            box_dims=box_dims,
            peak_axis=peak_axis,
            **kwargs
        )

        # check to make sure we found equal number of peaks on either
        # side of the guessed beam center position
        peaks_peak_axis = peaks[:, peak_axis]
        peaks_peak_axis = peaks_peak_axis[np.argsort(peaks_peak_axis)]
        low_peaks = peaks[peaks < self.metadata['center_px'][peak_axis]]
        high_peaks = peaks[peaks > self.metadata['center_px'][peak_axis]]

        # determine whether the right number of peaks was found
        no_peak_warning = False
        symmetric_warning = False
        if peaks.shape[0] == 0:
            no_peak_warning = True
        elif peaks.shape[0] % 2 != 0 or len(low_peaks) != len(high_peaks):
            symemtric_warning = True

        # determine the slope and intercept if more than 1 peak
        if peaks.shape[0] > 1:
            _, slope, intercept = line_fit(peaks[:, 1], peaks[:, 0])
        else:
            slope = np.nan
            intercept = np.nan

        if not no_peak_warning and not symmetric_warning:
            center = np.mean(peaks_peak_axis)

            if peak_axis == 1:
                center_qdx = center
                center_qdy = slope*center_qdx + intercept
            elif peak_axis == 0:
                center_qdy = center
                if np.isnan(slope):
                    # this means the peaks form perfectly vertical line
                    center_qdx = np.array(peaks_px)[0, 0]
                else:
                    center_qdx = (center_qdy-intercept)/slope

        if show_plot:
            fig, fig_slice = plotting.plot_find_beam_center(
                self, integrated_q_slice, np.array(peaks_px), np.array(peaks_q),
                [center_qdy, center_qdx])
            iplot(fig)
            iplot(fig_slice)

        # move this check to after the plots so even if we didn't find
        # the right peaks we can see the visualization
        if len(low_peaks) != len(high_peaks):
            raise ValueError(
                "Found peaks were not symmetric about the beam center.")

        return center_qdy, center_qdx

    def find_sdd_from_reference_peaks(
            self,
            pitch,
            size_qdy_px,
            size_qdx_px,
            peak_axis,
            peak_params: dict = {},
            peak_find_scale='linear',
            peak_orders: list = None,
            show_plot=True):
        """
        Calculate the sample to detector distance (SDD) from the
        known pitch of reference sample using simple 1D peak finding.
        See DataQdyQdx.find_peaks1D for a more detailed description of
        the peak finding process.
        For this method, only an integration box of size can be used
        and it must be centered on the beam center as determined by
        the user or the find_beam_center_from_peaks method.

        Only the 1st order peaks will be compared to the expected pitch of
        the SRM sample to determine the SDD.
        TODO: implement error handling when determining the SDD
        TODO: use provided error to filter out non-reference peaks,
        currently this is set to a 5% window as the accepted range

        In many cases the beam center position is likely to fall on
        an integer pixel value. This is because the peak finding
        algorithm only returns the pixel on which the peak is and does
        not perform any additional fit of the local intensity to determine
        a float pixel location of the peak.
        TODO: implement local gaussian fits for more accurate positions



        Parameters
        ----------
        pitch : float
            Known pitch of a reference sample in nanometers.
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
        peak_orders : list
            A list of integers that specfies the peak orders found.
            Default behavior is orders will start at n=1 and increase
            by one order for every peak found.
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
            if peak_axis == 'qdy' or peak_axis == 0:
                peak_axis = 0
            elif peak_axis == 'qdx' or peak_axis == 1:
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

        # Find peaks in the 1D slice
        peaks_q, peaks_px, peaks_px_int, integrated_q_slice =\
            self.find_peaks1D(
                box_mode='box_size',
                box_params=box_params,
                peak_params=peak_params,
                peak_find_scale=peak_find_scale,
                show_plot=False
            )

        if show_plot:
            fig, fig_slice = plotting.plot_find_beam_center(
                self, integrated_q_slice, np.array(peaks_px),
                np.array(peaks_q),
                self.metadata['center_px'])
            iplot(fig)
            iplot(fig_slice)

        # convert to pixel distances relative to beam center
        peaks = np.array(peaks_px)
        peaks = peaks[np.argsort(peaks[:, peak_axis]), :]
        low_peaks = peaks[
            peaks[:, peak_axis] < self.metadata['center_px'][peak_axis]
            ] - self.metadata['center_px']
        high_peaks = peaks[
            peaks[:, peak_axis] > self.metadata['center_px'][peak_axis]
            ] - self.metadata['center_px']
        if len(low_peaks[:, 0]) != len(high_peaks[:, 0]):
            raise ValueError(
                "Found peaks were not symmetric about the beam center.")

        # assume peak orders start at 1 unless told otherwise
        if peak_orders is None:
            peak_orders = np.arange(0, len(low_peaks[:, 0])) + 1

        sin_theta = peak_orders * self.metadata['wavelength_nm'] / (2 * pitch)
        theta = np.arcsin(sin_theta)

        # calculate magnitude of vector from beam center to peak in cm
        r_low_px = np.sqrt(low_peaks[:, 0]**2 + low_peaks[:, 1]**2)
        r_low = r_low_px * self.metadata["pixel_size_um"]/10000
        r_low = np.flip(r_low)  # flip to match order of peak orders
        r_high_px = np.sqrt(high_peaks[:, 0]**2 + high_peaks[:, 1]**2)
        r_high = r_high_px * self.metadata["pixel_size_um"]/10000

        sdd_low = r_low/np.tan(2*theta)
        sdd_high = r_high/np.tan(2*theta)

        # calculate average SDD from all peaks
        average_sdd = np.mean(np.concatenate((sdd_low, sdd_high)))

        return average_sdd

    def find_detector_rotation_correction_from_peaks(
        self,
        peak_find_box_mode: str,
        peak_find_box_params: dict,
        peak_params: dict,
        peak_find_scale: str,
        show_plot=True,
    ):
        """
        Wrapper function that just pulls the rotation from the find_peaks1D
        function. This function is designed to make the more complicated
        find_peaks1D function accesible to users when performing a rotation
        correction during the integration step.
        TODO: this assumes there is no rotation of the detector with
        respect to the beam coordinate system. Update in the future
        to include a different position of the detector.

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
        peak_params : dict
            Dictionary of keyword arguments for the scipy.find_peaks
            algorithm; see scipy documentation for more information.
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
        float
            Angle kappa in degrees. This angle is a counterclockwise rotation
            about the primary beam path (qbz). It can be used to align the
            detector x and y coordinates with the sample x and y coordinates.
        """

        _, peaks_px, _, _ = self.find_peaks1D(
            box_mode=peak_find_box_mode,
            box_params=peak_find_box_params,
            peak_params=peak_params,
            peak_find_scale=peak_find_scale,
            show_plot=show_plot
        )

        # fit a line to the peaks
        peaks_array = np.array(peaks_px)
        if peaks_array.shape[0] > 1:
            angle, _, _ = line_fit(peaks_array[:, 1], peaks_array[:, 0])
        else:
            warnings.warn(
                "WARNING: Only one peak found for:\n"
                + f"{self.metadata['filename']}"
                + "\n Setting angle, slope, and intercept to nan."
                )
            angle = np.nan

        return angle

