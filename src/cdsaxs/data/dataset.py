"""
This module includes container classes for datasets, i.e. many of 1D or
2D data instances.
"""

from __future__ import annotations
import warnings

import numpy as np
from tqdm import tqdm

from cdsaxs.data.data2d import Data2D
from cdsaxs.data.reduced_data1d import ReducedData1D
from cdsaxs.data.reduced_slice import (
    ReducedData1DSlice
)
import cdsaxs.plotting.plotting as plotting
from cdsaxs.sample import Sample


class Dataset():
    """
    A container class for a set of Data2D instances that make up a
    single CD-SAXS measurement for a sample.

    Attributes
    ----------
    datas : dict
        Dictionary containing the Data2D objects. The key for each
        instance is Data2D.name. Be cautious if the name attribute
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
    integrated_datasets : dict
        Dictionary of dictionaries containing integrated data. The key
        corresponds to an index (ordered by integration).
        For each inner ditionary, the keys align with the datas.keys() and
        the values are instances of IntegratedDataSlices. These
        dictionaries are produced by the cdsaxs integrators.

    reduced_datastets : dict

    reduced_slices : dict


    """

    def __init__(
            self,
            datas: list[Data2D] = None,
            name: str = None,
            sample: str = None
    ):
        if datas is not None:
            self.add_data(datas)
        else:
            self.datas = {}

        self.name = name
        self.sample = sample

    def add_data(self, datas: Data2D | list[Data2D]):
        """Add one or more Data2D instances to the dataset."""
        datas = [datas] if not isinstance(datas, list) else datas
        for data in datas:
            if data.name in self.datas.keys():
                raise ValueError(
                    "You do not have unique names for Data2D instances."
                )
            else:
                self.datas[data.name] = data

    def remove_data(self,
                    datas: Data2D | list[Data2D] | str | list[str]):
        """
        Remove one or more Data2D instances from the dataset.
        One or more isntances of Data2D can be provided or a list
        of keys to the datas dictionary.
        """
        if type(datas) is not list:
            datas = [datas]
        for data in datas:
            if isinstance(data, Data2D):
                name = data.name
            else:
                name = data
            try:
                del self.datas[name]
            except KeyError:
                warnings.warn(f"Could not delete {name} data as it was"
                            "not part of the dataset.")

    def assign_sample(self, sample: Sample):
        """
        Assign the measured sample with an instance of the Sample class.
        """
        # TODO: implement required sample checks
        self.sample = sample

    def update_all_metadata(self,
                            metadata: dict,
                            overwrite: bool = True,
                            keys: list = None,
                            verbose: bool = True):
        """
        Add or update metadata for all Data2D stored in this Dataset.
        Existing metadata parameters can be updated by keeping the
        overwrite argument to True.

        Parameters
        ----------
        metadata : dict
            Key : value pairs of accepted metadata (key) and their
            values. See Data2D class docstring for list of accepted
            keywords.
        overwrite : bool
            If set to True, any metadata provided to this method will
            overwrite the existing value in the instance if it already
            exists in self.metadata.
            Default value is True.
        keys : list
            List of keys to the datas dictionary to select which data
            the update should apply to.
        verbose : bool
            If set to True, a progress bar will appear as each data
            metadata is updated. This can be helpful when the q
            calcultation if being updated for relevant metadata
            changes. 
            Default value is True.
        """
        if keys is None:
            keys = list(self.datas.keys())

        if verbose:
            pbar = tqdm(range(len(keys)), desc="Updating datas: ",
                        position=0, leave=True)
        for key in keys:
            data = self.datas[key]
            data.update_metadata(metadata=metadata, overwrite=overwrite)
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()

    def filter_data_by_metadata(self, **filters):
        """
        Filter the data by any of the metadata or user_params.
        All filter criteria must be met to be returned from this method.

        Parameters
        ----------
        **filters
            The metadata filters are provided as keyword arguments.
            The argument name should match any of the keys in the
            metadata or user_params of the data.
            The value type of these keyword arguments will specify the
            criteria the data must meet.
            tuple : Data must fall within a closed range of (min, max).
                Any data where the metadata value is >= min and <= max
                will meet the criteria.
            int | float : Data must equal this value exactly.
            str : Data must equal this exactly.
            If multiple criteria for the same metadata must be met,
            a list of any of these values can be used. 
            list[tuple] : Data must fall within one of the ranges
                provided.
            list[int | float] : Data must equal one of the values in the
                list.
            list[str] : Data must equal one of the values in the list.

        Returns
        -------
        list
            List of data keys that meet the provided metadata criteria.
        """

        keys = list(self.datas.keys())

        for key, value in filters.items():
            if isinstance(value, tuple):
                keys = [x for x in keys
                        if self.datas[x]._get_metadata(key) <= value[1]
                        and self.datas[x]._get_metadata(key) >= value[0]]
            elif isinstance(value, (int, float, str)):
                keys = [x for x in keys
                        if self.datas[x]._get_metadata(key) == value]
            elif isinstance(value, list):
                keys_i = []
                for value_i in value:
                    if isinstance(value_i, tuple):
                        keys_i.extend([x for x in keys
                                       if self.datas[x]._get_metadata(key) <= value_i[1]
                                       and self.datas[x]._get_metadata(key) >= value_i[0]])
                    elif isinstance(value_i, (int, float, str)):
                        keys_i.extend([x for x in keys
                                      if self.datas[x]._get_metadata(key) == value_i])
                    else:
                        raise ValueError(
                            f"Didn't recognize filter for {key} of {value_i}."
                        )
                keys = list(set(keys_i))
            else:
                raise ValueError(
                    f"Didn't recognize filter for {key} of {value}."
                )

    def update_all_user_params(
            self, params: dict, overwrite: bool = True,
            keys: list = None):
        """
        Add key: value pairs to the user params for all Data2D.
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
        keys : list
            List of keys to the datas dictionary to select which data
            the update should apply to.
        """
        if keys is None:
            keys = list(self.datas.keys())
        for key in keys:
            data = self.datas[key]
            data.update_user_params(params=params, overwrite=overwrite)

    def normalize_all_data_by_metadata(
            self, normalize_by, keys=None):
        """
        Normalize all data by the selected metadata or user parameters.
        This will not reset any previous normalization. If a new
        series or normalizations is desired, please run reset normalization
        first or change reset_first to True.

        Parameters
        ----------
        normalize_by : list
            List of accepted metadata keywords or user parameter keys
            that should be used to normalize the data.
        keys : list
            A list of datas keys can be used to only apply the normalization
            to a subset of the data in datas.
        """
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.normalize_by_metadata(normalize_by)

    def scale_all_data_by_metadata(
            self, scale_by, keys=None):
        """
        Scale the image by the desired value.
        This does not undo any previous scalings unless reset_scale is
        called first or reset_first is set to True.

        Parameters
        ----------
        value : float or int
            Value by which to scale the data.
        reset_first : boolean
            If set to True, any previous scaling will be rest
            before applying the new requested scale.
            If left as False, the new parameters will be factored into
            the existing scaling factor.
        keys : list
            A list of datas keys can be used to only apply the scaling
            to a subset of the data in datas.
        """
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.scale_by_metadata(scale_by)

    def normalize_all_data(self, value, keys=None):
        """
        Scale the data by the recipricol of the specified value.or array
        of values that match the dimensions of the data image.
        """
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.normalize_data(value)

    def scale_all_data(self, value, keys=None):
        """
        Scale the data by the recipricol of the specified value.or array
        of values that match the dimensions of the data image.
        """
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.scale_data(value)

    def add_to_all_data(self, value, keys=None):
        """
        Scale the data by the recipricol of the specified value.or array
        of values that match the dimensions of the data image.
        """
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.add_to_data(value)

    def subtract_from_all_data(self, value, keys=None):
        """
        Scale the data by the recipricol of the specified value.or array
        of values that match the dimensions of the data image.
        """
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.subtract_from_data(value)

    def apply_footprint_correction(self, keys=None):
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.apply_footprint_correction()

    def apply_sample_size_correction(self, keys=None):
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.apply_sample_size_correction()

    def apply_substrate_absorption_correction(self, keys=None):
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.apply_substrate_absorption_correction()

    def reset_all_data_intensity(self, keys=None):
        """
        Reset all normalization, scaling, adding and subtracting.
        """
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.reset_intensity()

    def rotate_all_data_ccw(self, steps, keys=None):
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
        keys : list
            List of datas keys that identify which data this method
            should be applied to. If not provide, this method will be
            applied to all Data2D instances in datas.
        """
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.rotate_image_ccw(steps)

    def flip_all_data_horizontally(self, keys=None):
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.flip_horizontally()

    def flip_all_data_vertically(self, keys=None):
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.flip_vertically()

    def reset_all_data(self, keys=None):
        if keys is None:
            keys = list(self.datas.keys())

        for key in keys:
            data = self.datas[key]
            data.reset_image()

    def apply_rotation_correction_all_data(
            self, keys=None, angles={}, verbose=True, **kwargs):
        """
        Apply a rotation correction to all data images that aligns
        the qsy and qsx axes with the qby and qbx axes, respectively,
        based on the sample rotation angles chi and omega.

        Parameters
        ----------
        keys : list, optional
            List of datas keys that identify which data this method
            should be applied to. If not provide, this method will be
            applied to all Data2D instances in datas.
        angles : dict, optional
            The rotation angle can be set manually by providing the
            angles in a dictionary where the key corresponds to the
            Data2D key in dataset.datas and the value is the rotation
            angle.
            Not all angles have to be provided; the user can supply
            an angle to one or more of the datas and the others not
            provided will be calculated automatically from the sample
            rotation angles omega and chi.
            Note that the rotation angle should be positive for a
            counterclockwise rotation about the beam center position.
            This is opposite of the angle returned by 'find_chi_from_peaks'
            as that method is referring to chi which is defined as
            a counterclockwise rotation about the positive z axis which
            is a clockwise rotation of the scattering image to the
            user.
            Units are degrees.
        verbose : bool, optional
            If set to True, a progress bar will be displayed as the
            rotation is applied to the selected data.
            Set to False to hide progress bar.
            Default is True.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments accetped by the rotate_image function
            in Data2D can be passed through this method.
        """

        if keys is None:
            keys = list(self.datas.keys())

        if verbose:
            pbar = tqdm(range(len(keys)), desc="Rotating datas: ",
                        position=0, leave=True)
        for key in keys:
            data = self.datas[key]
            if key in angles.keys():
                rotation_angle = np.deg2rad(angles[key])
            else:
                phi = np.deg2rad(data.metadata['sample_phi_deg']
                                + data.metadata['sample_phi_offset_deg'])
                omega = np.deg2rad(data.metadata['sample_omega_deg']
                                + data.metadata['sample_omega_offset_deg'])
                chi = np.deg2rad(data.metadata['sample_chi_deg']
                                + data.metadata['sample_chi_offset_deg'])

                rotation_angle = np.rad2deg(np.atan(
                    np.tan(chi)*np.cos(phi) - np.sin(phi)*np.tan(omega)/np.cos(chi)
                ))

            data.rotate_image(rotation_angle, rotation_center=data.metadata['center_px_detector'], **kwargs)

            data.update_user_params({'rotation_correction_angle_deg': rotation_angle})

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

    def integrate_dataset(
        self,
        mode: str = 'sum',
        axis: str | int = None,
        subtract_background_offset: int | list[int] = None,
        keys=None,
        width_qdy_px: int | dict = None,
        width_qdx_px: int | dict = None,
        range_qdy_px: tuple | dict = None,
        range_qdx_px: tuple | dict = None,
        center_qdy: tuple | dict = None,
        center_qdx: tuple | dict = None,
        shift_box_qdy_px: int | dict = 0,
        shift_box_qdx_px: int | dict = 0,
        **kwargs
    ):
        """
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
        keys : str | list[str]
            Datas keys that select which Data2D instances the integration
            is applied to.
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

        Parameters for Box Refinement
        -----------------------------
        The following keyword arguments are specific to defining the
        box limits of the region of interest for integration. We caution
        the user to consider which keyword arguments to select as not
        all should be used simultaneously. In the case that the box is
        overdefined by the user, this method will prioritize the
        parameters in order of this list:

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
        **kwargs
            Any of the scattering vector attribute keywords can be
            used to define a range to set the box limits with fully
            closed ranges. For example:
                qby=(-0.03, 0.03)
            would determine box limits that encompass pixels with values
            >= -0.03 and <= 0.03 in qby.
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

        Returns
        -------
        IntegratedDataset
            Dataset that contains the integrated q-slices.
        """
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
        else:
            keys = list(self.datas.keys())

        qslices = []
        for key in keys:
            data = self.datas[key]
            qslice, _ = data.integrate_box(
                mode=mode,
                axis=axis,
                show_plot=False,
                # if a dictionary is provided we will use the data key
                # to select the correct keyword value otherwise it will
                # be the same for all data
                subtract_background_offset=subtract_background_offset,
                width_qdy_px=width_qdy_px if not isinstance(
                    width_qdy_px, dict) else width_qdy_px[key],
                width_qdx_px=width_qdx_px if not isinstance(
                    width_qdx_px, dict) else width_qdx_px[key],
                range_qdy_px=range_qdy_px if not isinstance(
                    range_qdy_px, dict) else range_qdy_px[key],
                range_qdx_px=range_qdx_px if not isinstance(
                    range_qdx_px, dict) else range_qdx_px[key],
                center_qdy=center_qdy if not isinstance(
                    center_qdy, dict) else center_qdy[key],
                center_qdx=center_qdx if not isinstance(
                    center_qdx, dict) else center_qdx[key],
                shift_box_qdy_px=shift_box_qdy_px if not isinstance(
                    shift_box_qdy_px, dict) else shift_box_qdy_px[key],
                shift_box_qdx_px=shift_box_qdx_px if not isinstance(
                    shift_box_qdx_px, dict) else shift_box_qdx_px[key],
            )
            qslices.append(qslice)

        dataset = ReducedDataset(qslices)
        return dataset


class ReducedDataset():

    def __init__(
            self,
            datas: ReducedData1D | list | ReducedDataset = None,
            name=None):
        """
        A set of 1D slices taken from detector images in a dataset.

        Parameters
        ----------
        data : ReducedData1D | list, optional
            A single QSlice instance or list of QSlice instances to
            initialize the qslices attribute of this class. If not
            provided, the qslices attribute will be an empty list and
            instances of QSlice can later be added.
        name : str
            Custom name of the integrated dataset.
        """

        self.name = name
        self.datas = []
        if datas is not None:
            self.add_data(datas=datas)

    def add_data(self, datas: ReducedData1D | list | ReducedDataset):
        """
        Add a single QSlice or list of QSlice instances to this dataset.
        You can also provide another instance of this class and the
        slices will get added to this instance.
        """
        if isinstance(datas, ReducedData1D):
            datas = [datas]
        elif isinstance(datas, ReducedDataset):
            datas = [datas]

        if isinstance(datas, list):
            for dat in datas:
                if isinstance(dat, ReducedData1D):
                    self.datas.append(dat)
                elif isinstance(dat, ReducedDataset):
                    self.datas.extend(dat.datas)
                else:
                    raise ValueError(
                        "Didn't recognize reduced data type "
                        f"{type(dat)}."
                    )
        else:
            raise ValueError(
                "Qslices should be a single ReducedData1D or ReducedDataset "
                "or a list of any combination of those types. Not: "
                f"{type(datas)}"
            )

    def plot_data(
            self,
            log_scale=True,
            cmap='viridis',
            vmin=None,
            vmax=None,
            filter_by_q={},
            filter_by_metadata={},
            interpolated_data=False,
            **kwargs
    ):

        fig = plotting.plot_reduced_dataset(
            self,
            log_scale=log_scale,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            filter_by_q=filter_by_q,
            filter_by_metadata=filter_by_metadata,
            interpolated_data=interpolated_data,
            **kwargs
        )

        return fig


class ReducedSlices():

    def __init__(
            self,
            slices: ReducedData1DSlice | list | ReducedSlices = None,
            name=None):
        """
        A set of 1D slices taken from detector images in a dataset.

        Parameters
        ----------
        data : ReducedData1D | list, optional
            A single QSlice instance or list of QSlice instances to
            initialize the qslices attribute of this class. If not
            provided, the qslices attribute will be an empty list and
            instances of QSlice can later be added.
        name : str
            Custom name of the integrated dataset.
        """

        self.name = name
        self.data = []
        if slices is not None:
            self.add_slices(slices=slices)

    def add_slices(self, slices: ReducedData1DSlice | list | ReducedSlices):
        """
        Add a single QSlice or list of QSlice instances to this dataset.
        You can also provide another instance of this class and the
        slices will get added to this instance.
        """
        if isinstance(slices, ReducedData1DSlice):
            slices = [slices]
        elif isinstance(slices, ReducedSlices):
            slices = [slices]

        if isinstance(slices, list):
            for slice_i in slices:
                if isinstance(slice_i, ReducedData1DSlice):
                    self.data.append(slice_i)
                elif isinstance(slice_i, ReducedSlices):
                    self.data.extend(slice_i.data)
                else:
                    raise ValueError(
                        "Didn't recognize reduced data type "
                        f"{type(slice_i)}."
                    )
        else:
            raise ValueError(
                "Slices should be a single ReducedData1DSlice or ReducedSlices"
                " or a list of any combination of those types. Not: "
                f"{type(slices)}"
            )

    def plot_data(self,
                  q_axis='qsz',
                  integrated_axis='qsx',
                  filter_by_q={},
                  log_scale=True,
                  offset_order=0,
                  offset_value=0,
                  ):

        fig = plotting.plot_reduced_slices(
            self,
            q_axis=q_axis,
            integrated_axis=integrated_axis,
            filter_by_q=filter_by_q,
            log_scale=log_scale,
            offset_order=offset_order,
            offset_value=offset_value)

        return fig

    def export_reduced_slices(
            self,
            filepath,
            filter_by_q={},
            q_axis='qsz',
            integrated_axis='qsx',
            orthogonal_axis ='qsy',
            header_axis = None,
            decimals=5):
        """
        Returns the slected reduced slices set currently stored in the
        dataset. The user must specify the index of the set of slices
        as well as the q_slice_axis. The number of decimal places the
        slice positions are rounded at can be changed with the
        decimals keyword argument.

        NOTE: currently only a q_slice_axis of 'qsx' is accepted or
        formatted appropriately in the output file.
        TODO: generalize this in the future.
        """

        if header_axis is None:
            header_axis = integrated_axis
        elif (header_axis != integrated_axis) and (header_axis != 'qsr'):
            warnings.warn(
                "Currently only qsr or the integrated axis can be used as the header "
                "for exporting reduced slices. The header has been set to "
                "the integrated axis."
            )
            header_axis = integrated_axis

        filtered_slices = self.data.copy()
        for key, value in filter_by_q.items():
            keep = []
            for data in filtered_slices:
                test = getattr(data, key)
                if np.nanmin(test) >= np.nanmin(value)\
                        and np.nanmax(test) <= np.nanmax(value):
                    keep.append(True)
                else:
                    keep.append(False)
            filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

        length = 0
        for r_slice in filtered_slices:
            length = np.max((length, len(getattr(r_slice, q_axis))))

        datas = []

        for r_slice in filtered_slices:
            q = getattr(r_slice, q_axis)
            Iq = getattr(r_slice, '_masked_Iq')
            q_int = getattr(r_slice, integrated_axis)
            q_offset = getattr(r_slice, orthogonal_axis)
            
            if header_axis == 'qsr':
                q_header = np.hypot(q_int, q_offset)
            else:
                q_header = q_int
            
            # sort by q
            sorted_indexes = np.argsort(q)
            q = q[sorted_indexes]
            Iq = Iq[sorted_indexes]

            select = (~np.isnan(Iq)) & (Iq > 0)

            new_q = np.hstack(
                ([r'$q_z (\AA^{-1})$'],
                    np.round(q, decimals=decimals).astype(str)[select],
                    [""]*(length-len(q[select])))
                    )

            new_Iq = np.hstack(
                ([f'qx = {np.round(q_header, decimals)}'],
                    np.round(Iq, decimals=decimals).astype(str)[select],
                    [""]*(length-len(Iq[select])))
                    )

            datas.append(new_q)
            datas.append(new_Iq)

        datas = np.array(datas).T
        np.savetxt(filepath, datas, delimiter=',', fmt='%s')
