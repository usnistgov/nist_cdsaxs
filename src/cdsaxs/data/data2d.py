from __future__ import annotations
from typing import Tuple
import warnings

import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure

from .. import calculators
from .data_image import DataImage
from .metadata import (
    METADATA_KEYWORDS,
    check_metadata,
    correct_metadata_dtype,
    ACCEPTED_Q_AXES
)
from .reduced_data1d import ReducedData1D
from ..plotting import plotting
from .. import diffraction
from .. import tools
from ..tools import (
    find_peaks_2D,
    find_peaks_2D_one_axis,
    line_fit,
)

# any changes to these metadata values should update calculated q values
UPDATE_QB_TRIGGERS = [
    "energy_ev", "wavelength_nm", "sdd_cm", "pixel_size_um", "center_px",
    "detector_phi_deg", "detector_phi0_deg", "detector_phi_scale",
    "detector_y_mm", "detector_y0_mm"
]

UPDATE_QS_TRIGGERS = [
    "sample_phi_deg", "sample_phi_offset_deg",
    "sample_omega_deg", "sample_omega_offset_deg",
    "sample_chi_deg", "sample_chi_offset_deg",
]

ACCEPTED_Q_KEYWORDS = ACCEPTED_Q_AXES


def combine_data2d(*data2d: Data2D, name=None):
    """
    Combine two or more instances of Data2D into a single instance
    of Data2D.

    NOTE: This operation is not sensitive to any data
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
    in the metadata dictionary, this could give an artificially low value
    for exposure time.

    Parameters
    ----------
    *data2d : Data2D
        Any number of Data2D instances to combine.

    Returns
    -------
    new_data : Data2D
        New Data2D instance containing the combined image, mask,
        metadata, and user parameters.
    """

    # initialize information from the first 2d data instance
    first_data = data2d[0]
    metadata = dict(first_data.metadata)
    user_params = dict(first_data.user_params)
    mask = np.array(first_data.mask, copy=True)
    image = np.array(first_data.image, copy=True)
    if name is None:
        name = first_data.name

    # Data2D derives energy from wavelength and vice versa, so avoid
    # passing both back into the constructor when cloning metadata.
    if 'wavelength_nm' in metadata and 'energy_ev' in metadata:
        del metadata['energy_ev']

    for data in data2d[1:]:
        if data.image.shape != first_data.image.shape:
            raise ValueError(
                "All Data2D images must have the same shape to be combined. "
                f"Expected {first_data.image.shape}, got {data.image.shape}."
            )
        if data.mask.shape != first_data.mask.shape:
            raise ValueError(
                "All Data2D masks must have the same shape to be combined. "
                f"Expected {first_data.mask.shape}, got {data.mask.shape}."
            )
        if 'exposure_time_s' in metadata.keys() and 'exposure_time_s' in data.metadata.keys():
            metadata['exposure_time_s'] = np.sum([metadata['exposure_time_s'], data.metadata['exposure_time_s']])
        else:
            warnings.warn(
                f"One or more Data2D instances do not have an exposure_time_s "
                "metadata value. The combined exposure time has been removed "
                "from the metadata."
            )
            metadata.pop('exposure_time_s', None)

        mask += data.mask
        # Sum the two images elementwise while propagating nans.
        image = np.sum(np.stack([image, data.image]), axis=0)

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
    qby_1d : NDArray
        Scattering vector for each pixel along the detector y-axis at
        qdx = 0. Used for plotting detector image only.
    qbx_1d : NDArray
        Scattering vector for each pixel along the detector x-axis at
        qdy = 0. Used for plotting detector image only.
    qb : NDArray
        Scattering vector for each pixel in beam coordinate space.
    qby : NDArray
        The y-component of qb.
    qbx : NDArray
        The x-component of qb.
    qbz : NDArray
        The z-component of qb.
    qbr : NDArray
        The radial q component of qb derived from sqrt(qbx^2 + qby^2).
    qs : NDArray
        Scattering vector for each pixel in sample coordinate space.
    qsy : NDArray
        The y-component of qs.
    qsx : NDArray
        The x-component of qs.
    qsz : NDArray
        The z-component of qs.
    qsr : NDArray
        The radial q component of qs derived from sqrt(qsx^2 + qsy^2).
    sample_rotation : dict
        Metadata that describes the sample rotations in the CD-SAXS
        experiment. Includes keys of:
            rotation_type : 'extrinsic' or 'intrinsic', default is 'extrinsic'
            first_axis : 'x', 'y', or 'z'; default is 'x'
            second_axis : 'x', 'y', or 'z'; default is 'z'
            third_axis : 'x', 'y', or 'z'; default is 'y'
        This is required for the proper calculation of the q components,
        especially when chi and omega are not zero.
    metadata : dict
        Relevant scattering metadata to the image acquisition. These are
        key : value pairs where the key must be selected from the
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
        This class stores 2D scattering images with coordinates defined
        as y vs x in the detector coordinate frame, with positive y
        in the upward vertical direction and positive x in the left
        horizontal direction. The z axis is defined as the surface normal
        to follow the right-hand rule.

        In many instances the detector coordinates will align with the lab
        frame, where the z axis aligns with the beam path and the detector
        is configured normal to the primary beam.

        In cases where the detector has moved from the position with the
        incident beam normal to the surface, two angles can be defined.
        detector_phi : Rotation counterclockwise about the y-axis
            originating at the sample position in the x-z plane (lab frame).
        detector_omega : Rotation counterclockwise about the x-axis
            originating at the sample position in the y-z plane (lab frame).

        This class assumes that the sample rotation that occurs during
        a CD-SAXS experiment is primarily a counterclockwise rotation
        about the positive y-axis in sample coordinate space by
        'sample_phi_deg' + 'sample_phi_offset' degrees from normal
        incidence. If chi and omega rotations are present, this class
        assumes that the rotation sequence is extrinsic in the order
        of omega (rotation about the x-axis), chi (rotation about the z-axis),
        and phi (rotation about y-axis).

        If the sample undergoes a different series of rotation, this
        can be changed using the _set_sample_rotation method, but we caution
        the user to only use this with a full understanding of its
        implications on the conversion from detector/beam-based
        coordinate system q to the sample coordinate system q.

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
            Two-dimensional boolean array with the same shape as image.
            True values mark pixels that should be masked in all
            operations.
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
            be passed as additional keyword arguments. Any keywords
            recognized as metadata by the cdsaxs code will be saved in
            the metadata attribute. The remaining information will be
            stored in the user_params dictionary.
        """

        # run base class init
        super().__init__(image=image, mask=mask)

        self.metadata = {}
        self._sample_rotation = {}

        self.update_metadata(
            {x: y for x, y in kwargs.items() if x in METADATA_KEYWORDS})

        # set the sample rotation metadata at default
        # default rotations are extrinsic in order of x, z, y
        self._set_sample_rotation()

        # set required metadata defaults if not present
        self.update_metadata(
            {'sample_phi_offset_deg': 0,
             'sample_chi_deg': 0,
             'sample_chi_offset_deg': 0,
             'sample_omega_deg': 0,
             'sample_omega_offset_deg': 0,
             "detector_phi_deg": 0,
             "detector_phi0_deg": 0,
             "detector_phi_scale": 1,
             "detector_y_mm": 0,
             "detector_y0_mm": 0},
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

        # initialize all q attributes as None
        q_attributes = ['qby_1d', 'qbx_1d',
                        'qb', 'qby', 'qbx', 'qbz', 'qbr',
                        'qs', 'qsy', 'qsx', 'qsz', 'qsr']
        for q_key in q_attributes:
            setattr(self, q_key, None)

        # calculate the q vectors if all required metadata is present
        # we will suppress the warning here but will give a single warning
        # at the end of this init if we can't calculate q
        self.calculate_q(suppress_errors=True)

        if not hide_q_warnings and self.qby_1d is None:
            warnings.warn(
                "Insufficient metadata to calculate q. "
                "Check your metadata to ensure the correct center "
                "position, wavelength, sample to detector distance, and "
                "pixel size are provided.")

    def _set_sample_rotation(self,
                             rotation_type=None,
                             first_axis=None,
                             second_axis=None,
                             third_axis=None):
        """
        Set the self.sample_rotation dictionary with details of the
        sample rotations. Carefully consider if changing the default
        settings that the selected rotation type and order matches the
        experimental conditions.

        If all keyword arguments are left as default values of None,
        the system will assume an extrinsic rotation type with the
        first, second, and third axes of x, z, y, respectively.

        Parameters
        ----------
        rotation_type : str
            Define the series of sample rotations as 'extrinsic' or
            'intrinsic'.
            An 'intrinsic' rotation is performed on the
            coordinate system after the previous rotation is performed.
            An 'extrinsic' rotation is performed on the original
            coordinate system prior to any rotations.
        first_axis : str
            Axis about which the first rotation is performed.
            Options are 'x', 'y', or 'z'.
        second_axis : str, optional
            Axis about which the second rotation is performed.
            Options are 'x', 'y', or 'z'.
        third_axis : str, optional
            Axis about which the first rotation is performed.
            Options are 'x', 'y', or 'z'.
        """
        accepted_axes = ['x', 'y', 'z']

        if rotation_type is None:
            rotation_type = 'extrinsic'
            if first_axis is not None or second_axis is not None or third_axis is not None:
                raise ValueError(
                    "The rotation type is required if any axes have been"
                    "assigned."
                )
            # the conditions to assume default has been reached
            first_axis = 'x'
            second_axis = 'z'
            third_axis = 'y'

        if first_axis is None:
            if second_axis is not None or third_axis is not None:
                raise ValueError(
                    "Axes should be assigned in order. Currently the first"
                    "axis is set to None but the second or third axis is"
                    "assigned. If there is only a second and/or third axis,"
                    "please assign the first axis keyword argument first."
                )

        if second_axis is None and third_axis is not None:
            raise ValueError(
                "Axes should be assigned in order. Currently the second"
                "axis is set to None but the third axis is assigned. Please"
                "reasign the second axis with the third axis value."
            )

        # check that the axes provided are accepted
        if first_axis.lower() not in accepted_axes:
            raise ValueError(
                f"Did not recognize first_axis of {first_axis}. Use "
                "'x', 'y', or 'z'."
            )
        if second_axis is not None and second_axis.lower() not in accepted_axes:
            raise ValueError(
                f"Did not recognize second_axis of {second_axis}. Use "
                "'x', 'y', or 'z'."
            )
        if third_axis is not None and third_axis.lower() not in accepted_axes:
            raise ValueError(
                f"Did not recognize third_axis of {third_axis}. Use "
                "'x', 'y', or 'z'."
            )
        if rotation_type.lower() not in ['intrinsic', 'extrinsic']:
            raise ValueError(
                f"Did not recognize rotation type of {rotation_type}. "
                "Use 'intrinsic' or 'extrinsic'."
            )

        self._sample_rotation = {
            'rotation_type': rotation_type.lower(),
            'first_axis': first_axis.lower(),
            'second_axis': second_axis.lower(),
            'third_axis': third_axis.lower(),
        }

        self.calculate_q(suppress_errors=True)

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
            Key-value pairs of accepted metadata and their values. See
            the class docstring for the list of accepted keywords.
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
                    value = correct_metadata_dtype(key, value)
                    self.metadata[key] = value
                    # handle special wavelength/energy relationship
                    if key == 'wavelength_nm':
                        self.metadata['energy_ev'] =\
                            calculators.wavelength_to_energy(value)
                    elif key == 'energy_ev':
                        self.metadata['wavelength_nm'] =\
                            calculators.energy_to_wavelength(value)
            if len([x for x in metadata.keys()
                    if x in UPDATE_QB_TRIGGERS]) > 0:
                try:
                    self._calculate_qb(suppress_errors=hide_q_warnings)
                except ValueError as e:
                    warnings.warn(f"{e}")
            if len([x for x in metadata.keys()
                    if x in UPDATE_QS_TRIGGERS or x in UPDATE_QB_TRIGGERS]) > 0:
                try:
                    self._calculate_qs(suppress_errors=hide_q_warnings)
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
            Key-value pairs of user-specified parameters for this data
            instance.
        overwrite : bool
            If set to True, any parameters provided to this method will
            overwrite the existing value in this instance if it already
            exists in self.user_params.
            Default value is True.
        """
        for key, value in params.items():
            if key in self.user_params.keys() and not overwrite:
                pass
            else:
                self.user_params[key] = value

    def _reset_q_attributes(self):
        """
        Reset all q attributes to None.
        """
        q_attributes = ['qby_1d', 'qbx_1d',
                        'qb', 'qby', 'qbx', 'qbz',
                        'qs', 'qsy', 'qsx', 'qsz']
        for q_key in q_attributes:
            setattr(self, q_key, None)

    def calculate_q(self, suppress_errors: bool = False):
        """
        Calculate the scattering vectors qb and qs in the beam and
        sample coordinate spaces, respectively, as well as their
        y-axis, x-axis, and z-axis components.

        Parameters
        ----------
        suppress_errors : bool, optional
            If set to True, this method will try to calculate the
            q vectors if the required metadata is available, but it
            will not raise an error if the parameters are not available.
            Default value is False.
        """

        self._calculate_qb(suppress_errors=suppress_errors)
        self._calculate_qs(suppress_errors=suppress_errors)

    def _calculate_qb(self, suppress_errors: bool = False):
        """
        Calculate the scattering vectors qb and qs in the beam and
        sample coordinate spaces, respectively, as well as their
        y-axis, x-axis, and z-axis components.

        Parameters
        ----------
        suppress_errors : bool, optional
            If set to True, this method will try to calculate the
            q vectors if the required metadata is availabe, but it
            will not raise an error if the parameters are not available.
            Default value is False.
        """

        # first check if we can calculate beam coordinate q
        missing_keywords = [x for x in UPDATE_QB_TRIGGERS
                            if x not in self.metadata.keys()]
        if len(missing_keywords) > 0:
            self.qb = None
            self.qby = None
            self.qbx = None
            self.qbz = None
            self.qbr = None
            self.qby_1d = None
            self.qbx_1d = None

            if not suppress_errors:
                raise ValueError(
                    "The following metadata is missing to calculate the "
                    "beam coordinate scattering vector: "
                    f"{missing_keywords}. Therefore, the sample coordinate"
                    "scattering vector also could not be calculated"
                    )
            else:
                pass
        else:
            qb, qby, qbx, qbz, _, cd = diffraction.detector_px_to_qbyxz(
                center_px=self.metadata['center_px'],
                detector_shape_px=self.image.shape,
                pixel_size_um=self.metadata['pixel_size_um'],
                wavelength_nm=self.metadata['wavelength_nm'],
                sdd_cm=self.metadata['sdd_cm'],
                center_coordinate_space='beam',
                detector_phi_deg=self.metadata['detector_phi_deg'],
                detector_y_mm=self.metadata['detector_y_mm'],
                detector_phi0_deg=self.metadata['detector_phi0_deg'],
                detector_y0_mm=self.metadata['detector_y0_mm'],
                detector_phi_scale=self.metadata['detector_phi_scale'],
            )
            qbr = diffraction.qyx_to_qr(qby, qbx)

            self.qb = qb
            self.qby = qby
            self.qbx = qbx
            self.qbz = qbz
            self.qbr = qbr

            self.update_metadata({'center_px_detector': cd})

            self.qby_1d = self.qby[
                :, int(round(self.metadata['center_px'][1], 0))
                ].reshape(-1)
            self.qbx_1d = self.qbx[
                int(round(self.metadata['center_px'][0], 0)), :
                ].reshape(-1)

    def _calculate_qs(self, suppress_errors: bool = False):
        """
        Calculate the scattering vectors qb and qs in the beam and
        sample coordinate spaces, respectively, as well as their
        y-axis, x-axis, and z-axis components.

        Parameters
        ----------
        suppress_errors : bool, optional
            If set to True, this method will try to calculate the
            q vectors if the required metadata is availabe, but it
            will not raise an error if the parameters are not available.
            Default value is False.
        """
        # first check if we can calculate beam coordinate q
        missing_keywords = [x for x in UPDATE_QS_TRIGGERS
                            if x not in self.metadata.keys()]
        if len(missing_keywords) > 0:
            self.qs = None
            self.qsy = None
            self.qsx = None
            self.qsz = None
            if not suppress_errors:
                raise ValueError(
                    "The following metadata is missing to calculate the "
                    "beam coordinate scattering vector: "
                    f"{missing_keywords}. Therefore, the sample coordinate"
                    "scattering vector also could not be calculated"
                    )
            else:
                pass
        elif self.qb is None:
            self.qs = None
            self.qsy = None
            self.qsx = None
            self.qsz = None
            self.qsr = None
            if not suppress_errors:
                raise ValueError(
                    "The beam-based scattering vectors qb are not"
                    "calculated and so sample-based scattering vectors"
                    "qs could not be calculated."
                )
        else:
            # check that the rotation information is present
            # this check is most relevant during the init
            if 'rotation_type' not in self._sample_rotation.keys() or\
                    'first_axis' not in self._sample_rotation.keys() or\
                    'second_axis' not in self._sample_rotation.keys() or\
                    'third_axis' not in self._sample_rotation.keys():
                if not suppress_errors:
                    raise ValueError(
                        "Sample rotation information is missing"
                    )
                else:
                    pass

            else:
                qs, qsy, qsx, qsz = diffraction.calculate_q_beam_to_sample(
                    qby=self.qby,
                    qbx=self.qbx,
                    qbz=self.qbz,
                    sample_phi_deg=self.metadata['sample_phi_deg']\
                        + self.metadata['sample_phi_offset_deg'],
                    sample_chi_deg=self.metadata['sample_chi_deg']\
                        + self.metadata['sample_chi_offset_deg'],
                    sample_omega_deg=self.metadata['sample_omega_deg']\
                        + self.metadata['sample_omega_offset_deg'],
                    rotation=self._sample_rotation['rotation_type'],
                    first_axis=self._sample_rotation['first_axis'],
                    second_axis=self._sample_rotation['second_axis'],
                    third_axis=self._sample_rotation['third_axis'],
                )
                qsr = diffraction.qyx_to_qr(qsy, qsx)
                self.qs = qs
                self.qsy = qsy
                self.qsx = qsx
                self.qsz = qsz
                self.qsr = qsr

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
        Scale the data by the reciprocal of the specified value.or array
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
        if (
            isinstance(steps, (float, np.floating))
            and not float(steps).is_integer()
        ):
            warnings.warn(
                "rotate_image_ccw only accepts integer quarter turns.",
                UserWarning,
            )
            raise ValueError(
                "rotate_image_ccw only accepts integer quarter turns."
            )

        k = int(steps)
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
                     resampling_mode="bilinear",
                     resampling_mode_q="bilinear",
                     fill_mode="constant",
                     fill_constant=np.nan,
                     use_pillow=False,
                     **kwargs):

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
            Center of rotation (y, x).
            Default is the upper left pixel, (0,0).
        resampling_mode : str, optional
            Set the resampling method used during the rotation.
            The box rotation works by rotating the image underneath then
            extracting the box for integration. Resampling modes are
            chosen from the sklearn.transform.warp method. Options are:
                nearest_neighbor
                bilinear (default)
                biquadratic
                bicubic
                biquartic
                biquintic
            Default value is 'bilinear'.
            If use_pillow is set to True, then the options for the
            PILLOW package rotation algorithm are different:
                nearest
                bilinear
                bicubic
        resampling_mode_q : str, optional
            Set the resampling method used during the rotation of the
            q component arrays.
            The box rotation works by rotating the array underneath then
            extracting the box for integration. Resampling modes are
            chosen from the sklearn.transform.warp method. Options are:
                nearest_neighbor
                bilinear (default)
                biquadratic
                bicubic
                biquartic
                biquintic
            Default value is 'bilinear'.
            If use_pillow is set to True, then the options for the
            PILLOW package rotation algorithm are different:
                nearest
                bilinear
                bicubic
        fill_mode : str, optional
            Determine how pixels outside the boundaries of the input image
            are filled after the rotation. Options match those from np.pad.
            Options are:
                constant (default)
                edge
                symmetric
                reflect
                wrap
            Default value is "constant".
        fill_constant : float, optional
            Specifies the constant value used to fill pixels outside the
            image boundaries after rotation. Only applies when resampling_mode
            is set to 'constant'.
        use_pillow : bool, optional
            If set to True, the algorithm will use the PILLOW package
            image rotation function instead of sklearn.transform.rotate.
            The fill_mode argument is not used and the resampling_mode
            options are slightly different, see the above description.
        """

        if rotation_center is None:
            rotation_center = self.metadata.get('center_px', (0, 0))

        for q in ['qs', 'qsx', 'qsy', 'qsz', 'qsr']:
            q_image = getattr(self, q)
            if q_image is not None:
                if not use_pillow:
                    q_image_rot = tools.rotate_image(
                        q_image, degrees=rotation_angle_deg,
                        rotation_center=rotation_center,
                        resampling_mode=resampling_mode_q,
                        fill_constant=fill_constant,
                        fill_mode=fill_mode,
                        **kwargs
                    )
                else:
                    q_image_rot = tools.rotate_image_pillow(
                        q_image, degrees=rotation_angle_deg,
                        rotation_center=rotation_center,
                        resampling_mode=resampling_mode_q,
                        **kwargs
                    )
            setattr(self, q, q_image_rot)

        super().rotate_image(rotation_angle_deg=rotation_angle_deg,
                             rotation_center=rotation_center,
                             resampling_mode=resampling_mode)

    def calculate_omega(self, qsy0_angle):

        """
        Calculate the sample rotation angle omega with knowledge of
        the sample rotation angles phi and chi, as well as the angle
        at which the peaks along qsy appear in the detector image.

        The angle should be calculated by fitting a line of:
            qby = - slope * qbx + constant
            angle = np.atan(slope)
        The built in function for finding the peak angle can be used
        to find the correct angle.
        A positive angle will appear as if the detector image had
        been rotated clockwise as this is a counterclockwise rotation
        about the positive z-axis.

        Parameters
        ----------
        qsy0_angle : float
            Angle in degrees of the line formed by peaks along qsy in
            the detector image.

        Returns
        -------
        float
            Sample rotation angle omega in degrees.
        """

        phi = np.deg2rad(
            self.metadata['sample_phi_deg']
            + self.metadata['sample_phi_offset_deg']
        )
        chi = np.deg2rad(
            self.metadata['sample_chi_deg']
            + self.metadata['sample_chi_offset_deg']
        )

        omega_rad = np.atan(
            np.cos(chi)*(
                np.tan(chi)*np.cos(phi)-np.tan(np.deg2rad(qsy0_angle))
                )/np.sin(phi)
        )

        return np.rad2deg(omega_rad)

    def flip_horizontally(self):
        """
        Flip the scattering image horizontally and update the beam
        center metadata and q values.
        """
        super().flip_horizontally()
        center_px = self.metadata['center_px']
        self.metadata['center_px'] = (
            center_px[0],
            self.image.shape[1] - center_px[1] - 1
        )
        # recalcualte q if possible
        self.calculate_q(suppress_errors=True)

    def flip_vertically(self):
        """
        Flip the scattering image vertically and update the beam
        center metadata and q values.
        """
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

    def get_box_dims_size(
            self,
            size_qdy_px: int | float,
            size_qdx_px: int | float,
            shift_box_qdy_px: int = 0,
            shift_box_qdx_px: int = 0,
            center: tuple = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Find the pixel index limits in half open ranges [min, max) that
        define a region of interest based on a box with a specific width
        along the two axes.

        Parameters
        ----------
        size_qdy_px : int
            Height of the box in pixels along qdy axis (axis 0)
        size_qdx_px : int
            Height of the box in pixels along qdy axis (axis 0)
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
        center : tuple[int | float]
            Set the pixel center of the box.
            Default is the center_px_detector (detector based beam
            center coordinates) or next the center_px (beam based beam
            center coordinates).
            First element should be center along axis 0 and second
            element should be center along axis 1.

        Returns
        -------
        tuple(int, int)
            Half open range along the qdy axis (axis 0).
        tuple(int, int)
            Half open range along the qdx axis (axis 0).
        """

        limits_axis0 = self._get_box_dims_size_y(
            size_qdy_px=size_qdy_px, shift_box_qdy_px=shift_box_qdy_px,
            center=center if center is None else center[0]
        )

        limits_axis1 = self._get_box_dims_size_x(
            size_qdx_px=size_qdx_px, shift_box_qdx_px=shift_box_qdx_px,
            center=center if center is None else center[1]
        )

        return limits_axis0, limits_axis1

    def _get_box_dims_size_y(
            self,
            size_qdy_px: int,
            shift_box_qdy_px: int = 0,
            center: int | float = None,
            allow_overflow=False) -> Tuple[int, int]:
        """
        Find pixel index limits that defines the height and position of
        a rectangular region of interest (limits along y or first axis
        of the image).

        Parameters
        ----------
        size_qdy_px : int
            Height of the box in pixels along qdy axis (axis 0)
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            Default value is 0.
        center : int
            Set the pixel center of the box.
            Default is the center_px_detector (detector based beam
            center coordinates) or next the center_px (beam based beam
            center coordinates).

        Returns
        -------
        tuple(int, int)
            Half open range along the qdy axis (axis 0).
        """
        if center is None:
            try:
                center_float = self.metadata['center_px_detector'][0]
            except KeyError:
                center_float = self.metadata['center_px'][0]
            center = round(center_float)
        else:
            center_float = center
            center = round(center)

        # since the beam center can fall not on the center of a pixel
        # shift the box in the case that the box width is even to be
        # as close to centered around the beam center as possible
        if size_qdy_px % 2 == 0 and center_float > center:
            shift_box_qdy_px -= 1

        min0 = center - int(size_qdy_px/2) - shift_box_qdy_px
        max0 = min0 + size_qdy_px

        if not allow_overflow:
            min0 = max(min0, 0)
            max0 = min(max0, self.image.shape[0])

        return min0, max0

    def _get_box_dims_size_x(
            self,
            size_qdx_px: int,
            shift_box_qdx_px: int = 0,
            center: int | float = None,
            allow_overflow=False) -> Tuple[int, int]:
        """
        Find pixel index limits that defines the width and position of
        a rectangular region of interest (limits along x or second
        axis of the image).

        Parameters
        ----------
        size_qdx_px : int
            Height of the box in pixels along qdy axis (axis 0)
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.
        center : int
            Set the pixel center of the box.
            Default is the center_px_detector (detector based beam
            center coordinates) or next the center_px (beam based beam
            center coordinates).

        Returns
        -------
        tuple(int, int)
            Half open range along the qdx axis (axis 1).
        """
        if center is None:
            try:
                center_float = self.metadata['center_px_detector'][1]
            except KeyError:
                center_float = self.metadata['center_px'][1]
            center = round(center_float)
        else:
            center_float = center
            center = round(center)

        # since the beam center can fall not on the center of a pixel
        # shift the box in the case that the box width is even to be
        # as close to centered around the beam center as possible
        if size_qdx_px % 2 == 0 and center_float > center:
            shift_box_qdx_px -= 1

        min1 = center - int(size_qdx_px/2) - shift_box_qdx_px
        max1 = min1 + size_qdx_px

        if not allow_overflow:
            min1 = max(min1, 0)
            max1 = min(max1, self.image.shape[1])

        return min1, max1

    def get_box_dims_qrange(
            self, **ranges) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Find the pixel index limits in half open ranges [min, max) that
        define a region of interest based on a box with set q ranges
        on both axes.

        If your q-range selection results in pixels that do not form
        a rectangular region of interest, the algorithm will try to
        find the largest rectangular region of interest that still
        meets all the provided criteria and bounds.

        If you are looking for a mask that finds all pixels that meet
        the provided qrange criteria, please use:
        Data2D.get_pixels_qrange().

        Parameters
        ----------
        *ranges : tuple | list
            Ranges for any of the q-component attributes of this class
            can be provided as keyword arguments. For example,
            providing qsy=(-0.01, 0.01) would select pixels that have
            a qsy values >= -0.01 and <= 0.01.
            Accepted q components include:
                qb
                qby
                qbx
                qbz
                qs
                qsy
                qsx
                qsz

        Returns
        -------
        tuple[int, int]
            Half open range along axis 0 of the rectangular region of
            interest.
        tuple[int, int]
            Half open range along axis 1 of the rectangular region of
            interest.
        """
        invalid_kwargs = [x for x in ranges.keys()
                          if x not in ACCEPTED_Q_KEYWORDS]
        if len(invalid_kwargs) > 0:
            raise KeyError(
                "The following keyword arguments are not allowed for this"
                f" function: {invalid_kwargs}. Please see the relevant "
                "doc strings for more details."
            )

        selection_mask = self.get_pixels_qrange(**ranges)

        limits = tools.find_maximum_rectangular_roi(selection_mask)

        return limits

    def get_pixels_qrange(self, **ranges) -> NDArray:
        """
        Returns a selection mask that includes image pixels that have
        q-components within the provided ranges. It will not return
        any points that are currently masked.

        Parameters
        ----------
        *ranges : tuple | list
            Ranges for any of the q-component attributes of this class
            can be provided as keyword arguments. For example,
            providing qsy=(-0.01, 0.01) would select pixels that have
            a qsy values >= -0.01 and <= 0.01.
            Accepted q components include:
                qb
                qby
                qbx
                qbz
                qs
                qsy
                qsx
                qsz

        Returns
        -------
        NDArray
            Boolean array the same size as the current data image that
            is True for pixels that meet all of the provided q ranges
            and are not already masked by the instance of this class
            (unless the mask is ignored).
        """

        invalid_kwargs = [x for x in ranges.keys()
                          if x not in ACCEPTED_Q_KEYWORDS]
        if len(invalid_kwargs) > 0:
            raise KeyError(
                "The following keyword arguments are not allowed for this"
                f" function: {invalid_kwargs}. Please see the relevant "
                "doc strings for more details."
            )

        selected = np.ones_like(self.image).astype(bool)

        for q_comp, limits in ranges.items():
            q_test = getattr(self, q_comp)
            selected_q = (q_test >= min(limits)) & (q_test <= max(limits))
            selected = selected * selected_q

        return selected

    def get_box_dims(
            self,
            width_qdy_px: int = None,
            width_qdx_px: int = None,
            range_qdy_px: tuple = None,
            range_qdx_px: tuple = None,
            center_qdy: tuple = None,
            center_qdx: tuple = None,
            shift_box_qdy_px: int = 0,
            shift_box_qdx_px: int = 0,
            **kwargs
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Find the index limits that define a rectangular region of
        interest in the scattering image based on a range of optional
        criteria.

        Not all keyword arguments should be used at the same time.
        For each axis, the routes to determine the pixel ranges are
        listed below. You can use two different methods to select ranges
        along the y and x axes by providing different keyword arguments
        for each axis.

        1. box size
            Set the width of the box along one of the axes using
            'width_qdy_px' along the vertical axis or 'width_qdx_px'
            along the horizontal axis.

            The box will be centered at the detector space beam center
            position if available otherwise the beam coordinate space
            beam center position ('center_px' in metadata) by default.

            The center position can be set to a custom value in either
            the beam or sample-coordinate spaces. For example, you can
            set the box center along the vertical y-axis by setting
            'center_qdy' to either ('qby', value) or ('qsy', value).
            For example, you can selection a region of interest along
            the line where 'qsy' is equal to 0.1 by setting ('qsy', 0.1).
            The pixel with the closest value to 0.1 will be set as the
            center pixel.
            CAUTION: This function assumes that you have aligned the
            sample space x and y axes parallel with the detector space
            x and y axes, respectively, by applying a rotation correction
            for easier region of interest selection.

            Accepted keyword arguments:
                width_qdy_px
                center_qdy
                shift_box_qdy_px
                width_qdx_px
                center_qdx
                shift_box_qdx_px

        2. pixel range
            Set the pixel indexing range with 'range_qdy_px' or
            'range_qdx_px'. These tuples of (start, stop) represent
            half open ranges of [start, stop), following the Python
            indexing rules.

            For example, a range of (5, 10) along qdy will select rows
            in the image with pixel indices of 5, 6, 7, 8, and 9. Keep
            in mind that indexing follows the Python Numpy convention
            of first axis counts rows from top to bottom and second
            axis counts columns from left to right.

            Accepted keyword arguments:
                range_qdy_px
                range_qdx_px

        3. q-ranges
            Use any of the q attributes to select a set of pixels that
            meet all the provided criteria. For example, setting
            'qs' = (0, 0.1) would select all pixels with a qs value
            >= 0 and <= 0.1. Note that these are closed ranges of
            [min, max].

            Also, using a combination of these ranges or
            the detector configuration may result in selected pixels
            that do not form a rectangular region of interest. In this
            case, the algorithm will try to find the largest rectangular
            continuous region of interest within these selected pixels.

            Accepted keyword arguments: **kwargs listed below.

        Parameters
        ----------
        width_qdy_px : int
            Set the box width along the vertical axis of the
            detector (qdy). It will be centered at the beam center
            unless otherwise set.
        width_qdx_px : int
            Set the box width along the horizontal axis of the
            detector (qdx). It will be centered at the beam center
            unless otherwise set.
        center_qdy : (keyword, value)
            Center the horizontal positioning of the box at another
            value other than qdy=0.
            This assumes that qby and qsy align with the horizontal
            image axis.
            The keyword should be 'qby' or 'qsy'.
        center_qdx : (keyword, value)
            Center the vertical positioning of the box at another
            value other than qdx=0.
            This assumes that qbx and qsx align with the vertical
            image axis.
            The keyword should be 'qbx' or 'qsx'.
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            This is the last step performed in determining the box
            dimensions, so all other limitations will be taken into
            account first.
            Default value is 0.
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.
        range_qdy_px : (min, max)
            Set the box pixel range along the vertical axis of the
            detector (qdy). This is a half open range [min, max).
        range_qdx_px : (min, max)
            Set the box pixel range along the horizontal axis of the
            detector (qdx). This is a half open range [min, max).
        **kwargs
            Ranges for any of the q-component attributes of this class
            can be provided as keyword arguments. For example,
            providing qsy=(-0.01, 0.01) would select a box that
            contains pixels within qsy values >= -0.01 and <= 0.01.
            Accepted q components include:
                qb
                qby
                qbx
                qbz
                qs
                qsy
                qsx
                qsz
        """

        invalid_kwargs = [x for x in kwargs.keys()
                          if x not in ACCEPTED_Q_KEYWORDS]
        if len(invalid_kwargs) > 0:
            raise KeyError(
                "The following keyword arguments are not allowed for this"
                f" function: {invalid_kwargs}. Please see the relevant "
                "doc strings for more details."
            )

        # dimensions along y or axis 0
        if width_qdy_px is not None:
            if center_qdy is not None:
                center_qdy = np.unravel_index(
                    np.nanargmin(np.abs(
                        getattr(self, center_qdy[0].lower())
                        - center_qdy[1])))[0]
            min_y, max_y = self._get_box_dims_size_y(
                size_qdy_px=width_qdy_px,
                center=center_qdy,
                shift_box_qdy_px=shift_box_qdy_px
            )
        elif range_qdy_px is not None:
            min_y, max_y = range_qdy_px
        else:
            (min_y, max_y), _ = self.get_box_dims_qrange(**kwargs)

        # dimensions along x or axis 1
        if width_qdx_px is not None:
            if center_qdx is not None:
                center_qdx = np.unravel_index(np.nanargmin(np.abs(
                    getattr(self, center_qdx[0].lower())
                    - center_qdx[1])))[1]
            min_x, max_x = self._get_box_dims_size_x(
                size_qdx_px=width_qdx_px,
                center=center_qdx,
                shift_box_qdx_px=shift_box_qdx_px
            )
        elif range_qdx_px is not None:
            min_x, max_x = range_qdx_px
        else:
            (min_x, max_x), _ = self.get_box_dims_qrange(**kwargs)

        return (min_y, max_y), (min_x, max_x)

    def integrate_box(
        self,
        mode: str = 'sum',
        axis: str | int = None,
        show_plot=True,
        subtract_background_offset: int | list[int] = None,
        width_qdy_px: int = None,
        width_qdx_px: int = None,
        range_qdy_px: tuple = None,
        range_qdx_px: tuple = None,
        center_qdy: tuple = None,
        center_qdx: tuple = None,
        shift_box_qdy_px: int = 0,
        shift_box_qdx_px: int = 0,
        plotting_kwargs={},
        **kwargs
    ) -> Tuple[ReducedData1D, Figure | None]:
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
        show_plot : bool, optional
            If set to True, the scattering image overlaid with the
            integration box boundaries will be shown in a first figure
            and the one-dimensional data will be shown in a second figure.
            Default value is True.
        subtract_background_offset: int, list[int], optional
            If set to a number of pixels greater than or equal to the
            width of the region of interest to be integrated over, a
            background subtraction will be performed by subtracting the
            intensity of an integrated box offset by the set number of
            pixels. If a single integer is provided, only one offset
            box will be used in the subtraction. If multiple are
            provided, the average signal from multiple integrated offset
            boxes will be used in the subtraction.
            Note that the integration mode for these boxes will align
            with the selected mode for this integration function.
        plotting_kwargs : dict, optional
            Keyword arguments passed to the matplotlib plotting helper
            used to generate the integration figure.

        Parameters for Box Refinement
        -----------------------------
        The following keyword arguments are specific to defining the
        box limits of the region of interest for integration. We caution
        the user to consider which keyword arguments to select as not
        all should be used simultaneously. Please see documentation
        for the data2d.get_box_dims() method for more details.

        width_qdy_px : int
            Set the box width along the vertical axis of the
            detector (qdy). It will be centered at the beam center
            unless otherwise set.
        width_qdx_px : int
            Set the box width along the horizontal axis of the
            detector (qdx). It will be centered at the beam center
            unless otherwise set.
        range_qdy_px : (min, max)
            Set the box pixel range along the vertical axis of the
            detector (qdy). This is a half open range [min, max).
        range_qdx_px : (min, max)
            Set the box pixel range along the horizontal axis of the
            detector (qdx). This is a half open range [min, max).
        center_qdy : (keyword, value)
            Center the horizontal positioning of the box at another
            value other than qdy=0.
            This assumes that qby and qsy align with the horizontal
            image axis.
            The keyword should be 'qby' or 'qsy'.
        center_qdx : (keyword, value)
            Center the vertical positioning of the box at another
            value other than qdx=0.
            This assumes that qbx and qsx align with the vertical
            image axis.
            The keyword should be 'qbx' or 'qsx'.
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            This is the last step performed in determining the box
            dimensions, so all other limitations will be taken into
            account first.
            Default value is 0.
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.
        **kwargs
            Ranges for any of the q-component attributes of this class
            can be provided as keyword arguments. For example,
            providing qsy=(-0.01, 0.01) would select a box that
            contains pixels within qsy values >= -0.01 and <= 0.01.
            Accepted q components include:
                qb
                qby
                qbx
                qbz
                qs
                qsy
                qsx
                qsz

        Returns
        -------
        ReducedData1D
            One-dimensional I vs. q data extracted from the integration.
        Figure | None
            Matplotlib figure generated for the integration when
            show_plot is True, otherwise None.

        """

        limits_qdy_px, limits_qdx_px = self.get_box_dims(
            width_qdy_px=width_qdy_px,
            width_qdx_px=width_qdx_px,
            range_qdy_px=range_qdy_px,
            range_qdx_px=range_qdx_px,
            center_qdy=center_qdy,
            center_qdx=center_qdx,
            shift_box_qdy_px=shift_box_qdy_px,
            shift_box_qdx_px=shift_box_qdx_px,
            **kwargs
        )

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

        q_rois = {}
        q_keys = ['qb', 'qby', 'qbx', 'qbz', 'qs', 'qsy', 'qsx', 'qsz', 'qsr', 'qbr']
        for key in q_keys:
            q_roi = getattr(self, key)[
                limits_qdy_px[0]: limits_qdy_px[1],
                limits_qdx_px[0]: limits_qdx_px[1]
            ]
            q_rois[key] = q_roi

        # extract scattering vector for this integration
        q = np.nanmean(q_rois['qbx' if axis == 0 else 'qby'], axis=axis)
        q_axis = 'qbx' if axis == 0 else 'qby'

        # extract background intensity
        if subtract_background_offset is not None:
            backgrounds = []
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

                    b_slice, _ = self.integrate_box(
                        mode=mode,
                        axis=axis,
                        show_plot=False,
                        range_qdy_px=limits_qdy_px_sub,
                        range_qdx_px=limits_qdx_px_sub,
                    )

                    backgrounds.append(b_slice)

            # even if some points are masked in some background offsets
            # we will use the background points
            background_i_avg = np.nanmean(
                np.array([b_slice.Iq for b_slice in backgrounds]),
                axis=0)
            integrated_i -= background_i_avg

        else:
            background_i_avg = None
            backgrounds = None

        # create instance of ReducedData1D to hold integration metadata
        integrated_q_slice = ReducedData1D(
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
            wavelength_nm=self.metadata['wavelength_nm'],
            sample_phi_deg=self.sample_phi_deg,
            sample_chi_deg=self.sample_chi_deg,
            sample_omega_deg=self.sample_omega_deg,
            **{key+'_roi': value for key, value in q_rois.items()},
            **{key: np.nanmean(value, axis=axis)
               for key, value in q_rois.items() if key != q_axis},
        )

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
            width_qdy_px=None,
            width_qdx_px=None,
            range_qdy_px=None,
            range_qdx_px=None,
            center_qdy=None,
            center_qdx=None,
            shift_box_qdy_px=0,
            shift_box_qdx_px=0,
            exclude_qdy=None,
            exclude_qdx=None,
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
        width_qdy_px : int, optional
            Set the box width along the vertical axis of the
            detector (qdy). It will be centered at the beam center
            unless otherwise set.
        width_qdx_px : int, optional
            Set the box width along the horizontal axis of the
            detector (qdx). It will be centered at the beam center
            unless otherwise set.
        range_qdy_px : (min, max), optional
            Set the box pixel range along the vertical axis of the
            detector (qdy). This is a half open range [min, max).
        range_qdx_px : (min, max), optional
            Set the box pixel range along the horizontal axis of the
            detector (qdx). This is a half open range [min, max).
        center_qdy : (keyword, value), optional
            Center the horizontal positioning of the box at another
            value other than qdy=0.
            This assumes that qby and qsy align with the horizontal
            image axis.
            The keyword should be 'qby' or 'qsy'.
        center_qdx : (keyword, value), optional
            Center the vertical positioning of the box at another
            value other than qdx=0.
            This assumes that qbx and qsx align with the vertical
            image axis.
            The keyword should be 'qbx' or 'qsx'.
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
     exclude_qdy : tuple[float, float], optional
         Exclude an inclusive region of q from the calculation.
         This is generally used to exclude the immediate region
         around the beamstop.
     exclude_qdx : tuple[float, float], optional
         Exclude an inclusive region of q from the calculation.
         This is generally used to exclude the immediate region
         around the beamstop.
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
        plotting_kwargs : dict, optional
            Keyword arguments passed to the matplotlib plotting helper
            used to generate the peak-finding figure.

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
        Figure | None
            Matplotlib figure generated for the peak-finding result
            when show_plot is True, otherwise None.
        """
        limits_qdy_px, limits_qdx_px = self.get_box_dims(
            width_qdy_px=width_qdy_px,
            width_qdx_px=width_qdx_px,
            range_qdy_px=range_qdy_px,
            range_qdx_px=range_qdx_px,
            center_qdy=center_qdy,
            center_qdx=center_qdx,
            shift_box_qdy_px=shift_box_qdy_px,
            shift_box_qdx_px=shift_box_qdx_px,
        )

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

        if self.qby_1d is not None and self.qbx_1d is not None:
            peaks_q = np.ones_like(peaks).astype(np.float64)

            peaks_q[:, 0] = np.interp(
                peaks[:, 0],
                np.arange(0, len(self.qby_1d)),
                self.qby_1d)

            peaks_q[:, 1] = np.interp(
                peaks[:, 1],
                np.arange(0, len(self.qbx_1d)),
                self.qbx_1d)

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
            width_qdy_px=None,
            width_qdx_px=None,
            range_qdy_px=None,
            range_qdx_px=None,
            center_qdy=None,
            center_qdx=None,
            shift_box_qdy_px=0,
            shift_box_qdx_px=0,
            exclude_q=None,
            peak_axis=None,
            integration_mode='sum',
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
        to determine the initial peak position. It is possible to use
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
         exclude_q : tuple[float, float], optional
             Exclude an inclusive region of q along the peak axis from
             the calculation. This is generally used to exclude the immediate
             region around the beamstop.
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

        Parameters for Box Refinement
        -----------------------------
        The following keyword arguments are specific to defining the
        box limits of the region of interest for integration. We caution
        the user to consider which keyword arguments to select as not
        all should be used simultaneously. Please see documentation
        for the data2d.get_box_dims() method for more details.

        width_qdy_px : int, optional
            Set the box width along the vertical axis of the
            detector (qdy). It will be centered at the beam center
            unless otherwise set.
        width_qdx_px : int, optional
            Set the box width along the horizontal axis of the
            detector (qdx). It will be centered at the beam center
            unless otherwise set.
        range_qdy_px : (min, max), optional
            Set the box pixel range along the vertical axis of the
            detector (qdy). This is a half open range [min, max).
        range_qdx_px : (min, max), optional
            Set the box pixel range along the horizontal axis of the
            detector (qdx). This is a half open range [min, max).
        center_qdy : (keyword, value), optional
            Center the horizontal positioning of the box at another
            value other than qdy=0.
            This assumes that qby and qsy align with the horizontal
            image axis.
            The keyword should be 'qby' or 'qsy'.
        center_qdx : (keyword, value), optional
            Center the vertical positioning of the box at another
            value other than qdx=0.
            This assumes that qbx and qsx align with the vertical
            image axis.
            The keyword should be 'qbx' or 'qsx'.
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            This is the last step performed in determining the box
            dimensions, so all other limitations will be taken into
            account first.
            Default value is 0.
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.

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
        Figure | None
            Matplotlib figure generated for the peak-finding result
            when show_plot is True, otherwise None.
        """

        limits_qdy_px, limits_qdx_px = self.get_box_dims(
            width_qdy_px=width_qdy_px,
            width_qdx_px=width_qdx_px,
            range_qdy_px=range_qdy_px,
            range_qdx_px=range_qdx_px,
            center_qdy=center_qdy,
            center_qdx=center_qdx,
            shift_box_qdy_px=shift_box_qdy_px,
            shift_box_qdx_px=shift_box_qdx_px,
        )

        min0, max0 = limits_qdy_px
        min1, max1 = limits_qdx_px

        if peak_axis is None:
            if (max0 - min0) > (max1 - min1):
                peak_axis = 0
                check_exclude = self.qby[min0:max0, int((max1-min1)/2)]
            else:
                peak_axis = 1
                check_exclude = self.qbx[int((max0-min0)/2), min1:max1]
        elif peak_axis == 'qdx':
            peak_axis = 1
            check_exclude = self.qbx[int((max0-min0)/2), min1:max1]
        elif peak_axis == 'qdy':
            peak_axis = 0
            check_exclude = self.qby[min0:max0, int((max1-min1)/2)]
        elif peak_axis == 1:
            check_exclude = self.qbx[int((max0-min0)/2), min1:max1]
        elif peak_axis == 0:
            check_exclude = self.qby[min0:max0, int((max1-min1)/2)]

        if exclude_q is not None:
            if isinstance(exclude_q, tuple):
                exclude_q = [exclude_q]
            exclude_q = np.array(exclude_q)
            exclude_px = []
            for zone in exclude_q:
                zone_px = np.where((check_exclude >= min(zone))
                                   & (check_exclude <= max(zone)))[0]
                exclude_px.append((min(zone_px), max(zone_px)))
        else:
            exclude_px = None

        peaks = find_peaks_2D_one_axis(
            self.image[min0:max0, min1:max1],
            peak_axis=peak_axis,
            integration_mode=integration_mode,
            log_scale=log_scale,
            refinement_size=refinement_size,
            mask=self.mask[min0:max0, min1:max1],
            algorithm=algorithm,
            exclude_ranges=exclude_px,
            **kwargs)

        if len(peaks.shape) < 2:
            peaks = np.empty((0, 2))
        peaks[:, 0] = peaks[:, 0] + min0
        peaks[:, 1] = peaks[:, 1] + min1

        if self.qby_1d is not None and self.qbx_1d is not None:
            peaks_q = np.ones_like(peaks).astype(np.float64)

            peaks_q[:, 0] = np.interp(
                peaks[:, 0],
                np.arange(0, len(self.qby_1d)),
                self.qby_1d)

            peaks_q[:, 1] = np.interp(
                peaks[:, 1],
                np.arange(0, len(self.qbx_1d)),
                self.qbx_1d)

            # moved exclusion check to the tools module
            # can only exclude q range along the peak axis
            # if exclude_q is not None:
            #     if isinstance(exclude_q, tuple):
            #         exclude_q = [exclude_q]
            #     exclude_q = np.array(exclude_q)
            #     for (ex_min, ex_max) in exclude_q:
            #         keep = (peaks_q[:, peak_axis] < ex_min) | (peaks_q[:, peak_axis] > ex_max)
            #         peaks = peaks[keep, :]
            #         peaks_q = peaks_q[keep, :]

        else:
            peaks_q = None

        if show_plot:
            fig = plotting.plot_data2d_find_peaks2d(
                self,
                peaks=peaks,
                limits_axis0=limits_qdy_px,
                limits_axis1=limits_qdx_px,
                zoom_plot=zoom_plot,
                exclude_ranges=exclude_px,
                exclude_axis=peak_axis,
                **kwargs
            )
        else:
            fig = None

        return peaks, peaks_q, fig

    def plot_data(
        self,
        log_scale=True,
        show_q=True,
        cmap='viridis',
        aspect='equal',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        **kwargs
    ):
        """
        Plot the scattering image with matplotlib.pyplot.imshow().

        Parameters
        ----------
        log_scale : bool
            If set to True, intensity values are plotted on a log scale.
            Default value is True.
        show_q : bool
            Shows the q components along x and y axes if available.
            If set to False, the pixel indices will be shown instead.
        cmap : str
            Matplotlib colormap name for the image intensity.
            Default is 'viridis'.
        aspect : str, float
            Define the aspect ratio of the pixels.
            'equal' : default, ensures that the pixels are square
            'auto' : changes the aspect ratio to fit within the plotting
                area of the figure
            float : manually set the aspct ratio of the pixel height vs width
        vmin : float
            Set the minimum value of the intensity color range.
        vmax : float
            Set the maximum value of the intensity color range.
        color_mask : str
            Set the color of the masked pixels. Default is 'transparent'.
            Other accepted strings are any of the matplotlib color names.
        color_inf : str
            Set the color of the pixels that have a value of inf or -inf.
            Accepted string are any of the matplotlib color names as well as
            'transparent'.
            Default is 'black'.
        color_nan : str
            Set the color of the pixels that have a value of nan.
            Accepted strings are any of the matplotlib color names as well as
            'transparent'.
            Default is 'red'.

        **kwargs
        --------
        Any of the keyword arguments for matplotlib.pyplot.imshow can be
        used. See the matplotlib documentation for more information.
        """
        fig = plotting.plot_data2d(
            self,
            log_scale=log_scale,
            show_q=show_q,
            cmap=cmap,
            aspect=aspect,
            vmin=vmin,
            vmax=vmax,
            color_mask=color_mask,
            color_inf=color_inf,
            color_nan=color_nan,
            **kwargs
        )

        return fig

    def find_beam_center_from_peaks(
            self,
            width_qdy_px=None,
            width_qdx_px=None,
            range_qdy_px=None,
            range_qdx_px=None,
            center_qdy=None,
            center_qdx=None,
            shift_box_qdy_px=0,
            shift_box_qdx_px=0,
            update=False,
            beam_center_guess=None,
            exclude_q=None,
            show_plot=True,
            zoom_plot=True,
            refinement_size=7,
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
        exclude_q : tuple[float, float], optional
            Exclude an inclusive region of q from the calculation.
            This is generally used to exclude the immediate region
            around the beamstop.
        ignore_peaks : list[int], optional
            Peak indices to ignore after peak finding when estimating
            the beam center.

        Parameters for Box Refinement
        -----------------------------
        The following keyword arguments are specific to defining the
        box limits of the region of interest for integration. We caution
        the user to consider which keyword arguments to select as not
        all should be used simultaneously. Please see documentation
        for the data2d.get_box_dims() method for more details.

        width_qdy_px : int, optional
            Set the box width along the vertical axis of the
            detector (qdy). It will be centered at the beam center
            unless otherwise set.
        width_qdx_px : int, optional
            Set the box width along the horizontal axis of the
            detector (qdx). It will be centered at the beam center
            unless otherwise set.
        range_qdy_px : (min, max), optional
            Set the box pixel range along the vertical axis of the
            detector (qdy). This is a half open range [min, max).
        range_qdx_px : (min, max), optional
            Set the box pixel range along the horizontal axis of the
            detector (qdx). This is a half open range [min, max).
        center_qdy : (keyword, value), optional
            Center the horizontal positioning of the box at another
            value other than qdy=0.
            This assumes that qby and qsy align with the horizontal
            image axis.
            The keyword should be 'qby' or 'qsy'.
        center_qdx : (keyword, value), optional
            Center the vertical positioning of the box at another
            value other than qdx=0.
            This assumes that qbx and qsx align with the vertical
            image axis.
            The keyword should be 'qbx' or 'qsx'.
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            This is the last step performed in determining the box
            dimensions, so all other limitations will be taken into
            account first.
            Default value is 0.
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.

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
        Figure | None
            Matplotlib figure generated for the beam-center result
            when show_plot is True, otherwise None.
        """

        if beam_center_guess is not None:
            self.update_metadata({'center_px': beam_center_guess},
                                 overwrite=True)
        box_dims = self.get_box_dims(
            width_qdy_px=width_qdy_px,
            width_qdx_px=width_qdx_px,
            range_qdy_px=range_qdy_px,
            range_qdx_px=range_qdx_px,
            center_qdy=center_qdy,
            center_qdx=center_qdx,
            shift_box_qdy_px=shift_box_qdy_px,
            shift_box_qdx_px=shift_box_qdx_px,
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
            range_qdy_px=box_dims[0],
            range_qdx_px=box_dims[1],
            peak_axis=peak_axis,
            exclude_q=exclude_q,
            show_plot=False,
            refinement_size=refinement_size,
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
                    center_qdx = float(np.nanmean(peaks[:, 1]))
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
            width_qdy_px=None,
            width_qdx_px=None,
            range_qdy_px=None,
            range_qdx_px=None,
            center_qdy=None,
            center_qdx=None,
            shift_box_qdy_px=0,
            shift_box_qdx_px=0,
            update=False,
            show_plot=True,
            peak_orders=None,
            exclude_q=None,
            zoom_plot=True,
            ignore_orders=[],
            refinement_size=7,
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
        exclude_q : tuple[float, float], optional
            Exclude an inclusive region of q from the calculation.
            This is generally used to exclude the immediate region
            around the beamstop.
        ignore_orders : list[int], optional
            Peak orders to ignore when estimating the sample detector
            distance.

        Parameters for Box Refinement
        -----------------------------
        The following keyword arguments are specific to defining the
        box limits of the region of interest for integration. We caution
        the user to consider which keyword arguments to select as not
        all should be used simultaneously. Please see documentation
        for the data2d.get_box_dims() method for more details.

        width_qdy_px : int, optional
            Set the box width along the vertical axis of the
            detector (qdy). It will be centered at the beam center
            unless otherwise set.
        width_qdx_px : int, optional
            Set the box width along the horizontal axis of the
            detector (qdx). It will be centered at the beam center
            unless otherwise set.
        range_qdy_px : (min, max), optional
            Set the box pixel range along the vertical axis of the
            detector (qdy). This is a half open range [min, max).
        range_qdx_px : (min, max), optional
            Set the box pixel range along the horizontal axis of the
            detector (qdx). This is a half open range [min, max).
        center_qdy : (keyword, value), optional
            Center the horizontal positioning of the box at another
            value other than qdy=0.
            This assumes that qby and qsy align with the horizontal
            image axis.
            The keyword should be 'qby' or 'qsy'.
        center_qdx : (keyword, value), optional
            Center the vertical positioning of the box at another
            value other than qdx=0.
            This assumes that qbx and qsx align with the vertical
            image axis.
            The keyword should be 'qbx' or 'qsx'.
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            This is the last step performed in determining the box
            dimensions, so all other limitations will be taken into
            account first.
            Default value is 0.
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.

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
        Figure | None
            Matplotlib figure generated for the SDD result when
            show_plot is True, otherwise None.
        """

        box_dims = self.get_box_dims(
            width_qdy_px=width_qdy_px,
            width_qdx_px=width_qdx_px,
            range_qdy_px=range_qdy_px,
            range_qdx_px=range_qdx_px,
            center_qdy=center_qdy,
            center_qdx=center_qdx,
            shift_box_qdy_px=shift_box_qdy_px,
            shift_box_qdx_px=shift_box_qdx_px,
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
            range_qdy_px=box_dims[0],
            range_qdx_px=box_dims[1],
            peak_axis=peak_axis,
            exclude_q=exclude_q,
            show_plot=False,
            refinement_size=refinement_size,
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
            # peak_orders = np.concatenate([
            #     np.flip(np.arange(0, len(low_peaks)))+1,
            #     np.arange(0, len(high_peaks))+1
            # ])
            peak_orders = np.round(peaks[:, peak_axis]/(2*np.pi/(pitch*10)),0).astype(int)

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

    def find_chi_from_peaks(
        self,
        width_qdy_px=None,
        width_qdx_px=None,
        range_qdy_px=None,
        range_qdx_px=None,
        center_qdy=None,
        center_qdx=None,
        shift_box_qdy_px=0,
        shift_box_qdx_px=0,
        show_plot=True,
        zoom_plot=True,
        exclude_q=None,
        **kwargs
    ):
        """
        Find the rotation angle, chi, of the sample coordinates x and y
        about the primary beam axis (qz). This function will look
        for the peaks that lie along the qsy=0 axis.

        For this method, the box dimensions are found internally for
        a box with specific widths along each axis centered around the
        starting beam center guess.

        This method should be performed on a scattering image at normal
        incidence (or as close as possible), i.e. sample_phi_deg +
        sample_phi_offset_deg = 0, otherwise the apparent angle of the
        points at either qsy=0 or qsx=0 will not be equal to chi but
        rather a combination of chi and omega (rotation about x-axis).

        Parameters
        ----------
        show_plot : bool
            If set to True, a first figure will display the scattering
            image overlaid with the integration box and markers on each
            detected peak while a second figure will show the 1D slice
            extracted from the integration and vertical lines at each
            peak position. The determiend beam center will be shown
            with dashed red lines.
        exclude_q : tuple[float, float], optional
            Exclude an inclusive region of q from the calculation.
            This is generally used to exclude the immediate region
            around the beamstop.

        Parameters for Box Refinement
        -----------------------------
        The following keyword arguments are specific to defining the
        box limits of the region of interest for integration. We caution
        the user to consider which keyword arguments to select as not
        all should be used simultaneously. Please see documentation
        for the data2d.get_box_dims() method for more details.

        width_qdy_px : int, optional
            Set the box width along the vertical axis of the
            detector (qdy). It will be centered at the beam center
            unless otherwise set.
        width_qdx_px : int, optional
            Set the box width along the horizontal axis of the
            detector (qdx). It will be centered at the beam center
            unless otherwise set.
        range_qdy_px : (min, max), optional
            Set the box pixel range along the vertical axis of the
            detector (qdy). This is a half open range [min, max).
        range_qdx_px : (min, max), optional
            Set the box pixel range along the horizontal axis of the
            detector (qdx). This is a half open range [min, max).
        center_qdy : (keyword, value), optional
            Center the horizontal positioning of the box at another
            value other than qdy=0.
            This assumes that qby and qsy align with the horizontal
            image axis.
            The keyword should be 'qby' or 'qsy'.
        center_qdx : (keyword, value), optional
            Center the vertical positioning of the box at another
            value other than qdx=0.
            This assumes that qbx and qsx align with the vertical
            image axis.
            The keyword should be 'qbx' or 'qsx'.
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            This is the last step performed in determining the box
            dimensions, so all other limitations will be taken into
            account first.
            Default value is 0.
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.

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
            Angle of clockwise rotation of the line formed by the
            located peaks about the primary beam path qbz.
            If the data analyzed is provided at normal incidence, i.e.,
            the sample rotation angle phi is as close to 0 as possible
            accounting for any offsets, this angle corresponds to
            the sample rotation angle chi.
        matplotlib.pyplot.figure | None
            If show_plot is set to True, the matplotlib pyplot figure
            generated is returned as the second object. If set to
            False, None is returned in its place.
        """

        box_dims = self.get_box_dims(
            width_qdy_px=width_qdy_px,
            width_qdx_px=width_qdx_px,
            range_qdy_px=range_qdy_px,
            range_qdx_px=range_qdx_px,
            center_qdy=center_qdy,
            center_qdx=center_qdx,
            shift_box_qdy_px=shift_box_qdy_px,
            shift_box_qdx_px=shift_box_qdx_px,
        )
        limits_qdy_px, limits_qdx_px = box_dims

        peak_axis = kwargs.pop('peak_axis', None)
        if peak_axis is None:
            peak_axis = 1
        peaks, _, _ = self.find_peaks2D_one_axis(
            range_qdy_px=limits_qdy_px,
            range_qdx_px=limits_qdx_px,
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
            rounded_center = np.round(
                np.array(self.metadata['center_px']), 0).astype(int)
            angle, slope, intercept = line_fit(
                [-1*self.qbx[y, x] for [y, x] in np.round(peaks[:], 0).astype(int)],
                [self.qby[y, x] for [y, x] in np.round(peaks[:], 0).astype(int)],
                force_intercept=(
                    self.qbx[rounded_center[0], rounded_center[1]],
                    self.qby[rounded_center[0], rounded_center[1]]))

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

        # angle is defined as positive is a clockwise rotation
        # this is because it is a counterclockwise rotation about the
        # beam based z-axis
        return -1*angle, fig

    def find_omega_from_peaks(
        self,
        width_qdy_px=None,
        width_qdx_px=None,
        range_qdy_px=None,
        range_qdx_px=None,
        center_qdy=None,
        center_qdx=None,
        shift_box_qdy_px=0,
        shift_box_qdx_px=0,
        show_plot=True,
        zoom_plot=True,
        exclude_q=None,
        **kwargs
    ):
        """
        Find the rotation angle, omega, about the positive x-axis in
        sample coordinate space.

        This method assumes that the sample rotation angle, chi, has
        already been determined and is loaded correctly in this
        Data2D instance's metadata.

        For this method, the box dimensions are found internally for
        a box with specific widths along each axis centered around the
        starting beam center guess.

        This method should be performed on a scattering image with a
        sample rotation angle, phi, far from normal incidence to most
        accurately determine omega. We also encourage the user to find
        omega at multiple phi angle to better understand uncertainty
        in the omega at different positions.

        Parameters
        ----------
        show_plot : bool
            If set to True, a first figure will display the scattering
            image overlaid with the integration box and markers on each
            detected peak while a second figure will show the 1D slice
            extracted from the integration and vertical lines at each
            peak position. The determiend beam center will be shown
            with dashed red lines.
        exclude_q : tuple[float, float], optional
             Exclude an inclusive region of q from the calculation.
             This is generally used to exclude the immediate region
             around the beamstop.

        Parameters for Box Refinement
        -----------------------------
        The following keyword arguments are specific to defining the
        box limits of the region of interest for integration. We caution
        the user to consider which keyword arguments to select as not
        all should be used simultaneously. Please see documentation
        for the data2d.get_box_dims() method for more details.

        width_qdy_px : int, optional
            Set the box width along the vertical axis of the
            detector (qdy). It will be centered at the beam center
            unless otherwise set.
        width_qdx_px : int, optional
            Set the box width along the horizontal axis of the
            detector (qdx). It will be centered at the beam center
            unless otherwise set.
        range_qdy_px : (min, max), optional
            Set the box pixel range along the vertical axis of the
            detector (qdy). This is a half open range [min, max).
        range_qdx_px : (min, max), optional
            Set the box pixel range along the horizontal axis of the
            detector (qdx). This is a half open range [min, max).
        center_qdy : (keyword, value), optional
            Center the horizontal positioning of the box at another
            value other than qdy=0.
            This assumes that qby and qsy align with the horizontal
            image axis.
            The keyword should be 'qby' or 'qsy'.
        center_qdx : (keyword, value), optional
            Center the vertical positioning of the box at another
            value other than qdx=0.
            This assumes that qbx and qsx align with the vertical
            image axis.
            The keyword should be 'qbx' or 'qsx'.
        shift_box_qdy_px : int, optional
            Number of pixels to shift the box by in the positive qdy
            direction. A negative value will shift the box in the
            negative qdy direction.
            This is the last step performed in determining the box
            dimensions, so all other limitations will be taken into
            account first.
            Default value is 0.
        shift_box_qdx_px : int, optional
            Number of pixels to shift the box by in the positive qdx
            direction. A negative value will shift the box in the
            negative qdx direction.
            Default value is 0.

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
            Angle of clockwise rotation of the line formed by the
            located peaks about the primary beam path qbz.
            If the data analyzed is provided far from normal incidence,
            i.e., the sample rotation angle phi is far from 0, and the
            sample rotation angle, chi, has been correctly provided in
            the metadata, this angle corresponds to the sample rotation
            angle omega.
        matplotlib.pyplot.figure | None
            If show_plot is set to True, the matplotlib pyplot figure
            generated is returned as the second object. If set to
            False, None is returned in its place.
        """

        box_dims = self.get_box_dims(
            width_qdy_px=width_qdy_px,
            width_qdx_px=width_qdx_px,
            range_qdy_px=range_qdy_px,
            range_qdx_px=range_qdx_px,
            center_qdy=center_qdy,
            center_qdx=center_qdx,
            shift_box_qdy_px=shift_box_qdy_px,
            shift_box_qdx_px=shift_box_qdx_px,
        )
        limits_qdy_px, limits_qdx_px = box_dims

        peak_axis = kwargs.pop('peak_axis', None)
        if peak_axis is None:
            peak_axis = 1
        peaks, _, _ = self.find_peaks2D_one_axis(
            range_qdy_px=limits_qdy_px,
            range_qdx_px=limits_qdx_px,
            peak_axis=peak_axis,
            exclude_q=exclude_q,
            show_plot=False,
            **kwargs
        )

        # determine whether the right number of peaks was found
        if peaks.shape[0] < 2:
            warnings.warn(
                "Insuffient peaks found to determine rotation angle.")
            angle, slope, intercept = np.nan, np.nan, np.nan
        else:
            rounded_center = np.round(
                np.array(self.metadata['center_px']), 0).astype(int)
            angle, slope, intercept = line_fit(
                [-1*self.qbx[y, x] for [y, x] in np.round(peaks[:], 0).astype(int)],
                [self.qby[y, x] for [y, x] in np.round(peaks[:], 0).astype(int)],
                force_intercept=(
                    self.qbx[rounded_center[0], rounded_center[1]],
                    self.qby[rounded_center[0], rounded_center[1]]))

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

        # angle is defined as positive is a clockwise rotation
        # this is because it is a counterclockwise rotation about the
        # beam based z-axis
        angle = -1*angle

        omega = self.calculate_omega(angle)

        return omega, fig

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

    def _get_metadata(self, keyword):
        """
        Return the metadata whether the keyword is found in the metadata
        or user_params dictionary.
        """
        if keyword in self.metadata.keys():
            return self.metadata[keyword]
        elif keyword in self.user_params.keys():
            return self.user_params[keyword]
        else:
            raise ValueError(
                f"Metadata or user_param not found for {keyword}."
            )

    def _lock_q_calculations(self):
        self._lock_q_calculations = True

    def _unlock_q_calculations(self):
        self._lock_q_calculations = False

    @property
    def sample_phi_deg(self):
        actual_sample_phi = self.metadata['sample_phi_deg']
        actual_sample_phi += self.metadata['sample_phi_offset_deg']
        return actual_sample_phi

    @property
    def sample_chi_deg(self):
        actual_sample_chi = self.metadata['sample_chi_deg']
        actual_sample_chi += self.metadata['sample_chi_offset_deg']
        return actual_sample_chi

    @property
    def sample_omega_deg(self):
        actual_sample_omega = self.metadata['sample_omega_deg']
        actual_sample_omega += self.metadata['sample_omega_offset_deg']
        return actual_sample_omega

    @property
    def detector_phi_deg(self):
        actual_detector_phi = self.metadata.get('detector_phi_deg', 0)
        actual_detector_phi -= self.metadata.get('detector_phi0_deg', 0)
        actual_detector_phi *= self.metadata.get('detector_phi_scale', 1)
        return actual_detector_phi

    @property
    def detector_y_mm(self):
        actual_detector_y = self.metadata.get('detector_y_mm', 0)
        actual_detector_y -= self.metadata.get('detector_y0_mm', 0)
        return actual_detector_y

    @property
    def energy_ev(self):
        return self.metadata.get('energy_ev')

    @property
    def wavelength_nm(self):
        return self.metadata.get('wavelength_nm')

    @property
    def exposure_time_s(self):
        return self.metadata.get('exposure_time_s')

    @property
    def sdd_cm(self):
        return self.metadata.get('sdd_cm')

    @property
    def pixel_size_um(self):
        return self.metadata.get('pixel_size_um')

    @property
    def data_directory(self):
        return self.metadata.get('data_directory')

    @property
    def filename(self):
        return self.metadata.get('filename')
