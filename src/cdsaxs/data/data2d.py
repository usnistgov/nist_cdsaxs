from __future__ import annotations
import warnings

import numpy as np
from numpy.typing import NDArray

import cdsaxs.calculators as calculators
from cdsaxs.data.data_image import DataImage
from cdsaxs.data.metadata import (
    METADATA_KEYWORDS,
    check_metadata
)
from cdsaxs.data.qslice import QSlice
import cdsaxs.plotting.plotting as plotting
import cdsaxs.diffraction as diffraction
from cdsaxs.tools import (
    find_peaks_2D,
    find_peaks_2D_one_axis,
    line_fit
)

# any changes to these metadata values should update calculated q values
UPDATE_Q_TRIGGERS = [
    "energy_ev", "wavelength_nm", "sdd_cm", "pixel_size_um", "center_px",
    "detector_phi_deg", "detector_phi_omega"
]


def combine_data2d(*data2d: Data2D, name=None):
    """
    Combine two or more instances of Data2D into a single instance
    of Data2D. This operation is not sensitive to any data
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
    *data2d : Data2D
        Any number of Data2D instances can be passed to this
        function and summed together.
    """

    # initialize information from the first 2d data instance
    first_data = data2d[0]
    metadata = first_data.metadata
    user_params = first_data.user_params
    mask = first_data.mask
    image = first_data.image
    if name is None:
        name = first_data.name

    for data in data2d[1:]:
        if 'exposure_time_s' in data.metadata.keys():
            metadata['exposure_time_s'] += data.metadata['exposure_time_s']
        mask += data.mask
        image = np.nansum(image, data.image)

    new_data = Data2D(
        image=image,
        name=name,
        mask=mask,
        **metadata,
        **user_params,
    )

    return new_data


class Data2D(DataImage):
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

    Protected Attributes
    --------------------
    _raw_image : NDArray
        Original image provided during initialization of an
        instance of this class. This enables the user to fully
        reset to the original image regardless of any data
        transformations performed.
    _masked_image : NDArray
        Retrieve the current image of the DataImage instance with
        all masked points replaced with np.nan.
    """

    def __init__(
            self,
            image: NDArray[np.floating],
            name: str = None,
            mask: NDArray[np.bool] = None,
            hide_q_warnings=False,
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
        hide_q_warnings : bool, optional
            Hide any warnings that may occur attempting to
            calculate the scattering vector during init.
            Default value is False.

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

        # set required metadata defaults if not present
        self.update_metadata(
            {'sample_phi_offset_deg': 0},
            overwrite=False, hide_q_warnings=True)
        self.update_metadata(
            {'center_px': (0, 0)},
            overwrite=False, hide_q_warnings=True)

        self.user_params = {}
        self.update_user_params(
            {x: y for x, y in kwargs.items() if x not in METADATA_KEYWORDS}
        )

        self.name = name if name is not None else\
            self.metadata['name'] if 'name' in self.metadata.keys() else\
            self.metadata['filename'] if 'filename' in self.metadata.keys()\
            else 'name'

        self.data_transformations = []

        self.qdy = None
        self.qdx = None

        # calculate the q vectors if all required metadata is present
        # we will suppress the warning here but will give a single warning
        # at the end of this init if we can't calculate q
        self.calculate_q(suppress_errors=True)

        if not hide_q_warnings and self.qdy is None:
            warnings.warn(
                "Insufficient metadata to calculate q. "
                "Check your metadata to ensure the correct center "
                "position, wavelength, sample to detector distance, and "
                "pixel size are provided.")

    def update_metadata(self,
                        metadata: dict,
                        overwrite: bool = True,
                        hide_q_warnings=False):
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
        hide_q_warnings : bool, optional
            Hide errors and warnings from the q calculation in the case
            that insufficient metadata is available to calculate q.
            Default is False.

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
                    self.calculate_q(suppress_errors=hide_q_warnings)
                except ValueError as e:
                    warnings.warn(f"{e}")

    def update_user_params(self,
                           params: dict,
                           overwrite: bool = True):
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
                    if key == keyword and transform == "scale":
                        warnings.warn(
                            f"{key} was already used in a scaling data "
                            "transformation. Skipping for now.")
                        value = None
            if value is not None:
                self.scale_data(value, keyword=key)

    def apply_footprint_correction(self):
        """
        Apply a footprint correction to the scattering intensity as a
        scaling parameter. The footprint factor will be stored in the
        metadata as "footprint_factor" and will be recalculated every
        time this correction is called.

        The footprint correction requires the sample phi angle be
        known. It will also apply the offset to this angle, so any
        offset angle should be determined prior to applying this
        correction.
        """
        # check that the proper metadata has been provided
        required_metadata = ["sample_phi_deg",
                             "sample_phi_offset_deg"]
        self._check_for_keywords_in_metadata(required_metadata)

        sample_phi_deg = self.metadata["sample_phi_deg"]
        sample_phi_deg += self.metadata["sample_phi_offset_deg"]
        footprint_factor = calculators.footprint_correction(sample_phi_deg)

        self.update_metadata({"footprint_factor": float(footprint_factor)},
                             overwrite=True)
        self.scale_by_metadata("footprint_factor")

    def apply_sample_size_correction(self):
        """
        Apply a sample size correction to the scattering intensity as a
        scaling parameter. The scaling factor will be stored in the
        metadata as "sample_size_factor" and will be recalculated every
        time this correction is called.

        The sample size correction requires the following metadata:
            "sample_size_mm"
            "beam_center_mm"
            "beam_fwhm_mm"
            "sample_phi_deg"
        """
        # check that the proper metadata has been provided
        required_metadata = ["sample_size_mm", "beam_center_mm",
                             "beam_fwhm_mm", "sample_phi_deg"]
        self._check_for_keywords_in_metadata(required_metadata)

        fwhm_mm = self.metadata["beam_fwhm_mm"]
        center_mm = self.metadata["beam_center_mm"]
        sample_size_mm = self.metadata["sample_size_mm"]

        sample_phi_deg = self.metadata["sample_phi_deg"]
        sample_phi_deg += self.metadata["sample_phi_offset_deg"]

        sample_size_factor = calculators.sample_size_correction(
            sample_phi_deg=sample_phi_deg,
            fwhm_mm=fwhm_mm,
            center_mm=center_mm,
            sample_size_mm=sample_size_mm
        )

        self.update_metadata({
            "sample_size_factor": float(sample_size_factor)},
            overwrite=True)
        self.scale_by_metadata("sample_size_factor")

    def apply_substrate_absorption_correction(self):
        """
        Applies a correction due to substrate absorption as a scaling
        factor to the scattering intensity. The scaling factor will be
        stored in the metadata as "substracte_absorption_factor" and
        will be recalculated every time this correction is called.

        The substrate absorption correction requires the following
        metadata:
            "substrate_thickness_um"
            "substrate_attenuation_coeff_um-1"
            "sample_phi_deg"
        """
        required_metadata = ["substrate_thickness_um",
                             "substrate_attenuation_coeff_um-1",
                             "sample_phi_deg"]
        self._check_for_keywords_in_metadata(required_metadata)

        thickness = self.metadata["substrate_thickness_um"]
        atten_coeff = self.metadata["substrate_attenuation_coeff_um-1"]

        sample_phi_deg = self.metadata["sample_phi_deg"]
        sample_phi_deg += self.metadata["sample_phi_offset_deg"]

        substrate_absorption_factor =\
            calculators.substrate_absorption_correction(
                sample_phi_deg=sample_phi_deg,
                substrate_thickness_um=thickness,
                substrate_attenuation_coeff_um_m1=atten_coeff
            )

        self.update_metadata({
            "substrate_absorption_factor":
            float(substrate_absorption_factor)},
            overwrite=True)
        self.scale_by_metadata("substrate_absorption_factor")

    # def apply_sample_absorption_correction(self):
    #     TODO: implement the sample absorption correction
    #     """
    #     Applies a correction due to sample absorption as a scaling
    #     factor to the scattering intensity. The scaling factor will be
    #     stored in the metadata as "sample_absorption_factor" and
    #     will be recalculated every time this correction is called.

    #     The substrate absorption correction requires the following
    #     metadata:
    #         "sample_thickness_um"
    #         "sample_attenuation_coeff_um-1"
    #         "sample_phi_deg"
    #     """
    #     pass

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
        while k < 0:
            k += 4
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

    def rotate_image(self,
                     rotation_angle_deg,
                     rotation_center=None,
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
            Default is the beam center if available, otherwise (0, 0).
        resampling_mode: str
            Set the resampling method used during the rotation.
            The box rotation works by rotating the image underneath then
            extracting the box for integration. Resampling of the
            image intensities can be performed with the 'nearest',
            'bilinear', or 'bicubic' methods in the PILLOW package.
            Default value is 'bicubic'.
        """

        if rotation_center is None:
            rotation_center = self.metadata.get('center_px', (0, 0))

        super().rotate_image(rotation_angle_deg=rotation_angle_deg,
                             rotation_center=rotation_center,
                             resampling_mode=resampling_mode)

    def flip_horizontally(self):
        super().flip_horizontally()
        center_px = self.metadata['center_px']
        self.metadata['center_px'] = (
            center_px[0],
            self.image.shape[1] - center_px[1] - 1
        )
        # recalcualte q if possible
        self.calculate_q(suppress_errors=True)

    def flip_vertically(self):
        super().flip_vertically()
        center_px = self.metadata['center_px']
        self.metadata['center_px'] = (
            self.image.shape[0] - center_px[0] - 1,
            center_px[1]
        )
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
        limits_qdy_px = (int(np.min(qdy_indices)),
                         int(np.max(qdy_indices)+1))

        qdx_indices = np.where((self.qdx >= range_qdx[0])
                               & (self.qdx < range_qdx[1]))[0]
        limits_qdx_px = (int(np.min(qdx_indices)),
                         int(np.max(qdx_indices)+1))

        return limits_qdy_px, limits_qdx_px

    def integrate_box(
        self,
        limits_qdy_px: list | tuple | int,
        limits_qdx_px: list | tuple | int,
        mode: str,
        axis: str | int = None,
        shift_box_qdy_px=0,
        shift_box_qdx_px=0,
        show_plot=True,
        subtract_background_offset: int | list[int] = None,
        plotting_kwargs={},
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
            If an integer value is given instead, the box limits will
            be determined internally for a box of that width centered
            around the beam center and offset by shift_box_qdy_px.
        limits_qdx_px : iterable of int
            Pixel range along qdx axis for integration box.
            Half open range of [min, max).
            If an integer value is given instead, the box limits will
            be determined internally for a box of that width centered
            around the beam center and offset by shift_box_qdx_px.
            If an integer was given for qdy, an integer must be given
            for qdx.
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
            If no axis is provided, the function will assume the data
            should be integrated over the shorter box dimension.
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
            Default value is True.
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
        if type(limits_qdy_px) is int and type(limits_qdx_px) is int:
            limits_qdy_px, limits_qdx_px = self.get_box_dims_size(
                limits_qdy_px, limits_qdx_px,
                shift_box_qdy_px=shift_box_qdy_px,
                shift_box_qdx_px=shift_box_qdx_px)

        if axis is None:
            if np.diff(limits_qdy_px) <= np.diff(limits_qdx_px):
                axis = 0
            else:
                axis = 1
        elif axis == 'qdy':
            axis = 0
        elif axis == 'qdx':
            axis = 1
        elif axis in [0, 1]:
            pass
        else:
            raise ValueError(f"Invalid integration axis of {axis}.")

        # access parent method of box integration
        integrated_i, image_box, mask_box = super().slice_box(
            limits_axis0=limits_qdy_px,
            limits_axis1=limits_qdx_px,
            axis=axis,
            mode=mode,
        )

        # extract scattering vector for this integration
        if axis == 0:
            q = self.qdx[limits_qdx_px[0]:limits_qdx_px[1]]
            q_int = np.mean(self.qdy[limits_qdy_px[0]:limits_qdy_px[1]])
            q_axis = 'qdx'
            q_int_axis = 'qdy'
        else:
            q = self.qdy[limits_qdy_px[0]:limits_qdy_px[1]]
            q_int = np.mean(self.qdx[limits_qdx_px[0]:limits_qdx_px[1]])
            q_axis = 'qdy'
            q_int_axis = 'qdx'

        # extract background intensity
        if subtract_background_offset is not None:
            backgrounds = []
            backgrounds_iq = []
            if type(subtract_background_offset) is int:
                subtract_background_offset = [subtract_background_offset]
            for offset in subtract_background_offset:
                if np.abs(offset) < image_box.shape[axis]:
                    warnings.warn(
                        f"A background subtraction offset of {offset} "
                        "is less than the integrated axis width and so "
                        "it will be skipped in the subtraction.")
                else:
                    limits_qdy_px_sub = (
                        limits_qdy_px[0] + (offset if axis == 0 else 0),
                        limits_qdy_px[1] + (offset if axis == 0 else 0)
                    )
                    limits_qdx_px_sub = (
                        limits_qdx_px[0] + (offset if axis == 1 else 0),
                        limits_qdx_px[1] + (offset if axis == 1 else 0)
                    )
                    background_i, b_image, b_mask = super().slice_box(
                        limits_axis0=limits_qdy_px_sub,
                        limits_axis1=limits_qdx_px_sub,
                        axis=axis,
                        mode=mode,
                    )

                    b_slice = QSlice(
                        q=q,
                        Iq=background_i,
                        q_axis=q_axis,
                        data2d=self,
                        limits_axis0=limits_qdy_px_sub,
                        limits_axis1=limits_qdx_px_sub,
                        integration_mode=mode,
                        integration_axis=axis,
                        image_roi=b_image,
                        image_mask=b_mask,
                    )

                    backgrounds_iq.append(background_i)
                    backgrounds.append(b_slice)

            background_i_avg = np.array(backgrounds_iq)
            # even if some points are masked in some background offsets
            # we will use the background points
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
            data2d=self,
            limits_axis0=limits_qdy_px,
            limits_axis1=limits_qdx_px,
            integration_mode=mode,
            integration_axis=axis,
            image_roi=image_box,
            image_mask=mask_box,
            background_Iq=background_i_avg,
            background_qslices=backgrounds,
        )
        # set the q-axis that was integrated over to the mean value
        integrated_q_slice.__setattr__(q_int_axis, q_int)

        if show_plot:
            fig = plotting.plot_data2d_integrate_box(
                integrated_q_slice,
                **plotting_kwargs
            )
        else:
            fig = None

        return integrated_q_slice, fig

    def find_peaks2D(
            self,
            limits_qdy_px=None,
            limits_qdx_px=None,
            exclude_qdy=None,
            exclude_qdx=None,
            shift_box_qdy_px=0,
            shift_box_qdx_px=0,
            log_scale=True,
            refinement_size=7,
            show_plot=True,
            zoom_plot=True,
            plotting_kwargs={},
            **kwargs
    ):
        """
        Find peaks across a two-dimensional image or region of interest
        using the scikit-image.feature peak_local_max() function and
        then further refined with local Gaussian fits across the two
        axes. Refinement is required for more accurate peak positions as
        the peak_local_max() only retuns the nearest pixel.

        Parameters
        ----------
        limits_qdy_px : iterable of int | int
            Pixel range along qdy axis for integration box.
            Half open range of [min, max).
            If an integer value is given instead, the box limits will
            be determined internally for a box of that width centered
            around the beam center and offset by shift_box_qdy_px.
        limits_qdx_px : iterable of int | int
            Pixel range along qdx axis for integration box.
            Half open range of [min, max).
            If an integer value is given instead, the box limits will
            be determined internally for a box of that width centered
            around the beam center and offset by shift_box_qdx_px.
            If an integer was given for qdy, an integer must be given
            for qdx.
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
        if type(limits_qdy_px) is int and type(limits_qdx_px) is int:
            limits_qdy_px, limits_qdx_px = self.get_box_dims_size(
                limits_qdy_px, limits_qdx_px,
                shift_box_qdy_px=shift_box_qdy_px,
                shift_box_qdx_px=shift_box_qdx_px)

        if limits_qdx_px is None:
            limits_qdy_px = (0, self.image.shape[0])
            limits_qdx_px = (0, self.image.shape[1])

        min0, max0 = limits_qdy_px
        min1, max1 = limits_qdx_px

        peaks = find_peaks_2D(
            self.image[min0:max0, min1:max1],
            log_scale=log_scale,
            refinement_size=refinement_size,
            mask=self.mask[min0:max0, min1:max1],
            **kwargs)

        if len(peaks.shape) < 2:
            peaks = np.empty((0, 2))
        peaks[:, 0] = peaks[:, 0] + min0
        peaks[:, 1] = peaks[:, 1] + min1

        if self.qdy is not None and self.qdx is not None:
            peaks_q = np.ones_like(peaks).astype(np.float64)

            peaks_q[:, 0] = np.interp(
                peaks[:, 0],
                np.arange(0, len(self.qdy)),
                self.qdy)

            peaks_q[:, 1] = np.interp(
                peaks[:, 1],
                np.arange(0, len(self.qdx)),
                self.qdx)

            if exclude_qdy is not None:
                if isinstance(exclude_qdy, tuple):
                    exclude_qdy = [exclude_qdy]
                for (ex_min, ex_max) in exclude_qdy:
                    keep = (peaks_q[:, 0] < ex_min) | (peaks_q[:, 0] > ex_max)
                    peaks = peaks[keep, :]
                    peaks_q = peaks_q[keep, :]

            if exclude_qdx is not None:
                if isinstance(exclude_qdx, tuple):
                    exclude_qdx = [exclude_qdx]
                for (ex_min, ex_max) in exclude_qdx:
                    keep = (peaks_q[:, 1] < ex_min) | (peaks_q[:, 1] > ex_max)
                    peaks = peaks[keep, :]
                    peaks_q = peaks_q[keep, :]

        if show_plot:
            fig = plotting.plot_data2d_find_peaks2d(
                self,
                peaks=peaks,
                limits_axis0=limits_qdy_px,
                limits_axis1=limits_qdx_px,
                zoom_plot=zoom_plot,
                **plotting_kwargs
            )
        else:
            fig = None

        return peaks, peaks_q, fig

    def find_peaks2D_one_axis(
            self,
            limits_qdy_px=None,
            limits_qdx_px=None,
            exclude_q=None,
            peak_axis=None,
            integration_mode='sum',
            shift_box_qdy_px=0,
            shift_box_qdx_px=0,
            log_scale=True,
            refinement_size=7,
            algorithm='scikit',
            zoom_plot=True,
            show_plot=True,
            **kwargs):
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

        if type(limits_qdy_px) is int and type(limits_qdx_px) is int:
            limits_qdy_px, limits_qdx_px = self.get_box_dims_size(
                limits_qdy_px, limits_qdx_px,
                shift_box_qdy_px=shift_box_qdy_px,
                shift_box_qdx_px=shift_box_qdx_px)

        if limits_qdx_px is None:
            limits_qdy_px = (0, self.image.shape[0])
            limits_qdx_px = (0, self.image.shape[1])

        min0, max0 = limits_qdy_px
        min1, max1 = limits_qdx_px

        if peak_axis is None:
            if (max0 - min0) > (max1 - min1):
                peak_axis = 0
            else:
                peak_axis = 1
        elif peak_axis == 'qdx':
            peak_axis = 1
        elif peak_axis == 'qdy':
            peak_axis = 0

        peaks = find_peaks_2D_one_axis(
            self.image[min0:max0, min1:max1],
            peak_axis=peak_axis,
            integration_mode=integration_mode,
            log_scale=log_scale,
            refinement_size=refinement_size,
            mask=self.mask[min0:max0, min1:max1],
            algorithm=algorithm,
            **kwargs)

        if len(peaks.shape) < 2:
            peaks = np.empty((0, 2))
        peaks[:, 0] = peaks[:, 0] + min0
        peaks[:, 1] = peaks[:, 1] + min1

        if self.qdy is not None and self.qdx is not None:
            peaks_q = np.ones_like(peaks).astype(np.float64)

            peaks_q[:, 0] = np.interp(
                peaks[:, 0],
                np.arange(0, len(self.qdy)),
                self.qdy)

            peaks_q[:, 1] = np.interp(
                peaks[:, 1],
                np.arange(0, len(self.qdx)),
                self.qdx)

            # can only exclude q range along the peak axis
            if exclude_q is not None:
                if isinstance(exclude_q, tuple):
                    exclude_q = [exclude_q]
                for (ex_min, ex_max) in exclude_q:
                    keep = (peaks_q[:, peak_axis] < ex_min) | (peaks_q[:, peak_axis] > ex_max)
                    peaks = peaks[keep, :]
                    peaks_q = peaks_q[keep, :]

        else:
            peaks_q = None

        if show_plot:
            fig = plotting.plot_data2d_find_peaks2d(
                self,
                peaks=peaks,
                limits_axis0=limits_qdy_px,
                limits_axis1=limits_qdx_px,
                zoom_plot=zoom_plot,
                **kwargs
            )
        else:
            fig = None

        return peaks, peaks_q, fig

    def plot_data(
            self,
            **kwargs
    ):
        """
        Plot the scattering image.
        """
        fig = plotting.plot_data2d(
            self,
            **kwargs
        )

        return fig

    def find_beam_center_from_peaks(
            self,
            size_qdy_px,
            size_qdx_px,
            update=True,
            beam_center_guess=None,
            exclude_q=None,
            show_plot=True,
            zoom_plot=True,
            ignore_peaks=[],
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
        try:
            peak_axis = kwargs.pop('peak_axis')
        except KeyError:
            if (max0 - min0) > (max1 - min1):
                peak_axis = 0
            else:
                peak_axis = 1
        peaks, _, _ = self.find_peaks2D_one_axis(
            limits_qdy_px=box_dims[0],
            limits_qdx_px=box_dims[1],
            peak_axis=peak_axis,
            exclude_q=exclude_q,
            show_plot=False,
            **kwargs
        )

        peaks = peaks[np.argsort(peaks[:, peak_axis]), :]
        if len(ignore_peaks) > 0:
            keep_index = [x for x in np.arange(0, peaks.shape[0])
                          if x not in ignore_peaks]
            peaks = peaks[keep_index, :]

        # check to make sure we found equal number of peaks on either
        # side of the guessed beam center position
        peaks_peak_axis = peaks[:, peak_axis]
        low_peaks = peaks_peak_axis[
            peaks_peak_axis < self.metadata['center_px'][peak_axis]]
        high_peaks = peaks_peak_axis[
            peaks_peak_axis > self.metadata['center_px'][peak_axis]]

        # determine whether the right number of peaks was found
        no_peak_warning = False
        symmetric_warning = False
        if peaks.shape[0] == 0:
            no_peak_warning = True
        elif len(low_peaks) != len(high_peaks):
            symmetric_warning = True

        # determine the slope and intercept if more than 1 peak
        if peaks.shape[0] > 1:
            _, slope, intercept = line_fit(peaks[:, 1], peaks[:, 0])
        else:
            slope = np.nan
            intercept = np.nan

        if not no_peak_warning and not symmetric_warning:
            center = np.mean(peaks_peak_axis)

            if peak_axis == 1:
                center_qdx = float(center)
                center_qdy = slope*center_qdx + intercept
            elif peak_axis == 0:
                center_qdy = float(center)
                if np.isnan(slope):
                    # this means the peaks form perfectly vertical line
                    center_qdx = float(np.nanmean(peaks)[:, 1])
                else:
                    center_qdx = (center_qdy-intercept)/slope

        if update and not symmetric_warning and not no_peak_warning:
            self.update_metadata({'center_px': (center_qdy, center_qdx)},
                                 overwrite=True)

        if show_plot:
            fig = plotting.plot_data2d_find_beam_center(
                self,
                peaks=peaks,
                limits_axis0=box_dims[0],
                limits_axis1=box_dims[1],
                zoom_plot=zoom_plot,
                **kwargs,
                show_beam_center=False\
                if symmetric_warning or no_peak_warning else (
                    True if update else (center_qdy, center_qdx))
            )
        else:
            fig = None

        if symmetric_warning:
            raise ValueError(
                "Found peaks were not symmetric about the beam center. "
                "Try changing the peak finding keyword arguments "
                "or check to make sure your beam center guess is "
                "reasonable close to the actual position.")
        if no_peak_warning:
            raise ValueError(
                "No peaks detected. Try changing the peak finding "
                "keyword arguments."
            )

        if update:
            self.update_metadata({'center_px': (center_qdy, center_qdx)},
                                 overwrite=True)

        return (round(center_qdy, 2), round(center_qdx, 2)), fig

    def find_sdd_from_reference_peaks(
            self,
            pitch_nm,
            size_qdy_px,
            size_qdx_px,
            update=True,
            show_plot=True,
            peak_orders=None,
            exclude_q=None,
            zoom_plot=True,
            ignore_orders=[],
            **kwargs
    ):
        """
        Attempt to calculate the sample to detector distance using the
        find_peaks2D_one_axis() method. Please refer to the method
        doc string for more information about the required arguments.

        For this method, the box dimensions are found internally for
        a box with specific widths along each axis centered around the
        starting beam center guess.

        The algorithm will assume that each order peak is found on
        either side of the beam center position, orders 1, 2, 3, etc.
        They don't have to be symmetric on both sides, just continuous,
        so finding orders -2, -1, 1, 2, 3, 4 would be acceptable. If
        not all order peaks are available, the peak_orders can be
        provided as a list of integers. In this example, if only orders
        -2, 1, 2, 3 were located then you would provide [-2, 1, 2, 3]
        for the peak_orders keyword argument.

        Parameters
        ----------
        pitch_nm : float
            Known pitch of a reference sample in nanometers.
        size_qdy_px : int
            Box size in pixels along the qdy axis.
        size_qdx_px : int
            Box size in pixels along the qdx axis.
        update : bool
            If set to True, the found sdd will be updated in
            the data metadata as well as returned. If set to False,
            the sdd will only be returned and the data metadata will
            remain at the last sdd value.
        peak_orders : list
            A list of integers that specfies the peak orders found.
            Default behavior is orders will start at n=1 and increase
            by one order for every peak found on either side of beam
            center.
        show_plot : bool
            If set to True, a first figure will display the scattering
            image overlaid with the integration box and markers on each
            detected peak while a second figure will show the 1D slice
            extracted from the integration and vertical lines at each
            peak position. The determiend beam center will be shown
            with dashed red lines.

        Other Parameters
        ----------------
        **kwargs
            Any additional keyword arguments for the find_peaks2D_one_axis()
            method can be passed through to the underlying function.
            This includes:
                peak_axis
                integration_mode
                log_scale
                refinement_size
                algorithm
                any keyword arguments for the fitting algorithm

        Returns
        -------
        float
            Sample detector distance (SDD) in the units of cm. This is
            the average SDD calculated using all found peak orders.
        float
            Standard deviation of the average sample detector distance
            returned in units of cm.
        """

        box_dims = self.get_box_dims_size(
            size_qdy_px=size_qdy_px,
            size_qdx_px=size_qdx_px
        )

        (min0, max0), (min1, max1) = box_dims
        try:
            peak_axis = kwargs.pop('peak_axis')
        except KeyError:
            if (max0 - min0) > (max1 - min1):
                peak_axis = 0
            else:
                peak_axis = 1
        peaks, peaks_q, _ = self.find_peaks2D_one_axis(
            limits_qdy_px=box_dims[0],
            limits_qdx_px=box_dims[1],
            peak_axis=peak_axis,
            exclude_q=exclude_q,
            show_plot=False,
            **kwargs
        )
        # sort the peaks by index along peak_axis
        peaks = peaks[np.argsort(peaks[:, peak_axis]), :]
        # peaks_q = peaks_q[np.argsort(peaks[:, peak_axis]), :]

        # figure out peak orders unless otherwise provided
        low_peaks = peaks[
            peaks[:, peak_axis] < self.metadata['center_px'][peak_axis],
            peak_axis
            ].reshape(-1)
        high_peaks = peaks[
            peaks[:, peak_axis] > self.metadata['center_px'][peak_axis],
            peak_axis
            ].reshape(-1)
        if peak_orders is None:
            peak_orders = np.concatenate([
                np.flip(np.arange(0, len(low_peaks)))+1,
                np.arange(0, len(high_peaks))+1
            ])

        if len(ignore_orders) > 0:
            keep_index = [y for x, y in
                          zip(peak_orders, np.arange(0, len(peak_orders)))
                          if x not in ignore_orders]
            peaks = peaks[keep_index, :]
            peak_orders = peak_orders[keep_index]

        pixel_distances_cm = np.sqrt(np.sum(
            (peaks - self.metadata['center_px'])**2, axis=1
            )) * self.metadata['pixel_size_um'] / 10000
        q_orders = 2*np.pi*peak_orders/(pitch_nm)  # keep in inverse nm
        theta_rad = np.arcsin(
            q_orders * self.metadata['wavelength_nm'] / (4 * np.pi)) * 2
        sdd_cm_orders = pixel_distances_cm / np.tan(theta_rad)

        # calculate average SDD from all peaks
        average_sdd = np.round(np.mean(sdd_cm_orders), 2)
        std_sdd = np.round(np.std(sdd_cm_orders), 2)

        if update:
            self.update_metadata({'sdd_cm': average_sdd}, overwrite=True)

        if show_plot:
            fig = plotting.plot_data2d_find_sdd(
                self,
                peaks=peaks,
                limits_axis0=box_dims[0],
                limits_axis1=box_dims[1],
                zoom_plot=zoom_plot,
                sdd_cm=(average_sdd, std_sdd),
                **kwargs,
            )
        else:
            fig = None

        return average_sdd, std_sdd, fig

    def find_detector_rotation_correction(
        self,
        size_qdy_px,
        size_qdx_px,
        show_plot=True,
        zoom_plot=True,
        exclude_q=None,
        **kwargs
    ):
        """
        Find the rotation angle of the sample coordinates x and y about the
        primary beam axis (qz).

        For this method, the box dimensions are found internally for
        a box with specific widths along each axis centered around the
        starting beam center guess.

        Parameters
        ----------
        size_qdy_px : int
            Box size in pixels along the qdy axis.
        size_qdx_px : int
            Box size in pixels along the qdx axis.
        peak_orders : list
            A list of integers that specfies the peak orders found.
            Default behavior is orders will start at n=1 and increase
            by one order for every peak found on either side of beam
            center.
        show_plot : bool
            If set to True, a first figure will display the scattering
            image overlaid with the integration box and markers on each
            detected peak while a second figure will show the 1D slice
            extracted from the integration and vertical lines at each
            peak position. The determiend beam center will be shown
            with dashed red lines.

        Other Parameters
        ----------------
        **kwargs
            Any additional keyword arguments for the find_peaks2D_one_axis()
            method can be passed through to the underlying function.
            This includes:
                peak_axis
                integration_mode
                log_scale
                refinement_size
                algorithm
                any keyword arguments for the fitting algorithm

        Returns
        -------
        float
            Angle kappa in degrees. This angle is a counterclockwise rotation
            about the primary beam path (qbz) from alignment along qdx. It can
            be used to align the detector x and y coordinates with the sample 
            x and y coordinates.
        """

        box_dims = self.get_box_dims_size(
            size_qdy_px=size_qdy_px,
            size_qdx_px=size_qdx_px
        )
        limits_qdy_px, limits_qdx_px = box_dims

        (min0, max0), (min1, max1) = box_dims
        try:
            peak_axis = kwargs.pop('peak_axis')
        except KeyError:
            if (max0 - min0) > (max1 - min1):
                peak_axis = 0
            else:
                peak_axis = 1
        peaks, _, _ = self.find_peaks2D_one_axis(
            limits_qdy_px=limits_qdy_px,
            limits_qdx_px=limits_qdx_px,
            peak_axis=peak_axis,
            show_plot=False,
            exclude_q=exclude_q,
            **kwargs
        )

        # determine whether the right number of peaks was found
        if peaks.shape[0] < 2:
            warnings.warn(
                "Insuffient peaks found to determine rotation angle.")
            angle, slope, intercept = np.nan, np.nan, np.nan
        else:
            angle, slope, intercept = line_fit(
                peaks[:, 1], peaks[:, 0],
                force_intercept=(self.metadata['center_px'][1],
                                 self.metadata['center_px'][0]))

        if show_plot:
            fig = plotting.plot_data2d_find_detector_rotation_correction(
                self,
                peaks=peaks,
                line=(angle, slope, intercept),
                limits_axis0=limits_qdy_px,
                limits_axis1=limits_qdx_px,
                zoom_plot=zoom_plot,
                **kwargs
            )
        else:
            fig = None

        return angle, fig

    def _check_for_keywords_in_metadata(self, keywords):
        """
        Check that the list of keywords provided are all present in
        the metadata.
        """
        missing_keywords = [
            x for x in keywords if x not in self.metadata.keys()
        ]
        if len(missing_keywords) > 0:
            raise ValueError(
                "The following metadata is missing for:\n" +
                f"{missing_keywords}\n" +
                "The following metadata are all required:\n" +
                f"{keywords}"
            )

        return True
