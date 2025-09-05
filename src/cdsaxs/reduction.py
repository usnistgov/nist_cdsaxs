"""
Reduction tools for processing the cdsaxs Dataset class.
"""
from __future__ import annotations

import numpy as np

from cdsaxs.data1d import Data1D, ReducedData
from cdsaxs.dataset import Dataset
import cdsaxs.plotting as plotting
import cdsaxs.diffraction as diffraction


def integrate_dataset(
        dataset: Dataset,
        limits_qdy_px: list | tuple,
        limits_qdx_px: list | tuple,
        mode: str,
        axis: str | int,
        in_place: bool = True,
        box_angle_deg: float | dict = 0.0,
        rotation_sampling_mode: str = 'bicubic',
) -> None | dict:
    """
    Integrate a box defined by indexing limits for each data in
    the dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset containing datas on which the integration is performed.
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
    in_place: bool
        If set to True, the integrated dataset (dictionary) will be
        added to the dataset.integrated_datasets attribute. If set to
        False, the dictionary will be returned.
        Default value is True.
    box_angle_deg : float, dict
        Rotate the box by the set number of degrees clockwise
        about the center point. Rotating the box will maintain the
        size of the box.
        Providing a float value applies the same angle to all data
        images. Providing a dictionary using the same data keywords
        can be used to rotate each image at a unique angle.
        Units are in degrees.
        Default value is 0.
    rotation_sampling_mode : str
        Set the resampling method used when a box angle is provided.
        The box rotation works by rotating the image underneath then
        extracting the box for integration. Resampling of the
        image intensities can be performed with the 'nearest',
        'bilinear', or 'bicubic' methods in the PILLOW package.
        Default value is 'bicubic'.

    Returns
    -------
    dict
        Dictionary of integrated q slices, where each key is the name of
        the data image (dataset.datas.keys()) used and the value is
        of type IntegratedQSlice.
    """

    integrated_datas = {}
    for key, data in dataset.datas.items():
        integrated_q_slice = data.integrate_box(
            limits_qdy_px=limits_qdy_px,
            limits_qdx_px=limits_qdx_px,
            mode=mode,
            axis=axis,
            box_angle_deg=box_angle_deg[key] if type(box_angle_deg) is dict else box_angle_deg,
            rotation_sampling_mode=rotation_sampling_mode,
        )
        if key != integrated_q_slice.name:
            raise KeyError(
                "Something is wrong, the dataset data keys don't match"
                "the integrated slice names."
            )
        else:
            integrated_datas[key] = integrated_q_slice

    if in_place:
        if len(dataset.integrated_datasets.keys()) == 0:
            key = 0
        else:
            key = max(dataset.integrated_datasets.keys())+1
        dataset.integrated_datasets[key] = integrated_datas
        return None
    else:
        return integrated_datas


def integrate_dataset_box_of_size(
    dataset: Dataset,
    size_qdy_px: int,
    size_qdx_px: int,
    mode: str,
    axis: str | int,
    shift_box_qdy_px: int = 0,
    shift_box_qdx_px: int = 0,
    in_place=True,
    box_angle_deg: float | dict = 0.0,
    rotation_sampling_mode: str = 'bicubic',
    subtract_background=False,
    subtraction_offset=5
) -> None | dict:
    """
    Integrate a box of a specific size. By default this box is
    centered at the closest pixel to the beam center position (q=0),
    but it can be shifted in either qdy or qdx by a set number of
    pixels.

    Parameters
    ----------
    dataset : Dataset
        The dataset containing datas on which the integration is performed.
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
    in_place: bool
        If set to True, the integrated dataset (dictionary) will be
        added to the dataset.integrated_datasets attribute. If set to
        False, the dictionary will be returned.
        Default value is True.
    box_angle_deg : float, dict
        Rotate the box by the set number of degrees clockwise
        about the center point. Rotating the box will maintain the
        size of the box.
        Providing a float value applies the same angle to all data
        images. Providing a dictionary using the same data keywords
        can be used to rotate each image at a unique angle.
        Units are in degrees.
        Default value is 0.
    rotation_sampling_mode : str
        Set the resampling method used when a box angle is provided.
        The box rotation works by rotating the image underneath then
        extracting the box for integration. Resampling of the
        image intensities can be performed with the 'nearest',
        'bilinear', or 'bicubic' methods in the PILLOW package.
        Default value is 'bicubic'.
    subtract_background : bool, optional
        If set to true, will run a background subtraction on the integrated
        data based on the supplied integration box offset by a set number 
        of pixels.
        TODO: decide how to best approach this subtraction past this 
        initial implementation.
    subtraction_offset: int, optional
        The number of pixels to offset the integration box for calculating 
        the background intensity by. 

    Returns
    -------
    dict
        Dictionary of integrated q slices, where each key is the name of
        the data image (dataset.datas.keys()) used and the value is
        of type IntegratedQSlice.
    """
    integrated_datas = {}
    i = 0
    for key, data in dataset.datas.items():
        integrated_q_slice = data.integrate_box_of_size(
                size_qdy_px=size_qdy_px,
                size_qdx_px=size_qdx_px,
                mode=mode,
                axis=axis,
                shift_box_qdy_px=shift_box_qdy_px,
                shift_box_qdx_px=shift_box_qdx_px,
                show_plot=False,
                box_angle_deg=box_angle_deg[key] if type(box_angle_deg) is dict else box_angle_deg,
                rotation_sampling_mode=rotation_sampling_mode,
                subtract_background=subtract_background,
                subtraction_offset=subtraction_offset,
            )

        if key != integrated_q_slice.name:
            raise KeyError(
                "Something is wrong, the dataset data keys don't match"
                "the integrated slice names."
            )
        else:
            integrated_datas[key] = integrated_q_slice

        i += 1

    if in_place:
        if len(dataset.integrated_datasets.keys()) == 0:
            key = 0
        else:
            key = max(dataset.integrated_datasets.keys())+1
        dataset.integrated_datasets[key] = integrated_datas
    else:
        return integrated_datas


def integrate_dataset_box_of_q_range(
        dataset: Dataset,
        range_qdy: list | tuple,
        range_qdx: list | tuple,
        mode: str,
        axis: str | int,
        in_place=True,
        box_angle_deg: float | dict = 0.0,
        rotation_sampling_mode: str = 'bicubic',
        subtract_background=False,
        subtraction_offset=5
):
    """
    Integrate a box defined by scattering vector limits.

    Parameters
    ----------
    dataset : Dataset
        The dataset containing datas on which the integration is performed.
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
    in_place: bool
        If set to True, the integrated dataset (dictionary) will be
        added to the dataset.integrated_datasets attribute. If set to
        False, the dictionary will be returned.
        Default value is True.
    box_angle_deg : float, dict
        Rotate the box by the set number of degrees clockwise
        about the center point. Rotating the box will maintain the
        size of the box.
        Providing a float value applies the same angle to all data
        images. Providing a dictionary using the same data keywords
        can be used to rotate each image at a unique angle.
        Units are in degrees.
        Default value is 0.
    rotation_sampling_mode : str
        Set the resampling method used when a box angle is provided.
        The box rotation works by rotating the image underneath then
        extracting the box for integration. Resampling of the
        image intensities can be performed with the 'nearest',
        'bilinear', or 'bicubic' methods in the PILLOW package.
        Default value is 'bicubic'.
    subtract_background : bool, optional
        If set to true, will run a background subtraction on the integrated
        data based on the supplied integration box offset by a set number 
        of pixels.
        TODO: decide how to best approach this subtraction past this 
        initial implementation.
    subtraction_offset: int, optional
        The number of pixels to offset the integration box for calculating 
        the background intensity by. 

    Returns
    -------
    dict
        Dictionary of integrated q slices, where each key is the name of
        the data image (dataset.datas.keys()) used and the value is
        of type IntegratedQSlice.
    """

    integrated_datas = {}
    for key, data in dataset.datas.items():
        integrated_q_slice = data.integrate_box_of_q_range(
            range_qdy=range_qdy,
            range_qdx=range_qdx,
            mode=mode,
            axis=axis,
            box_angle_deg=box_angle_deg[key] if type(box_angle_deg) is dict else box_angle_deg,
            rotation_sampling_mode=rotation_sampling_mode,
            subtract_background=subtract_background,
            subtraction_offset=subtraction_offset
        )
        if key != integrated_q_slice.name:
            raise KeyError(
                "Something is wrong, the dataset data keys don't match"
                "the integrated slice names."
            )
        else:
            integrated_datas[key] = integrated_q_slice

    if in_place:
        if len(dataset.integrated_datasets.keys()) == 0:
            key = 0
        else:
            key = max(dataset.integrated_datasets.keys())+1
        dataset.integrated_datasets[key] = integrated_datas
    else:
        return integrated_datas


def create_reduced_QszQsx(
        dataset: Dataset,
        integrated_index: int = None,
        in_place: bool = True):
    """
    For each integrated q slice of each data image, a reduced dataset
    is created which is comprised of 'qsx', 'qsz' and 'Iq'.

    Currently this is only implemented for I vs. qdx slices and assumes
    that qdy = 0. TODO: expand this in the future to all axes.

    Parameters
    ----------
    dataset : Dataset
        The dataset containing datas on which the integration is performed.
    integrated_index : int
        Index of the integrated dataset of dataset to use in generating
        the reduced dataset. This is the key used in the
        dataset.integrated_datasets dictionary.
    in_place: bool
        If set to True, the reduced dataset (dictionary) will be
        added to the dataset.reduced_datasets attribute. If set to
        False, the dictionary will be returned.
        Default value is True.

    Returns
    -------
    dict
        Dictionary of dictionaries, where each key is the name of
        the data image (dataset.datas.keys()) used and the value is
        of type IntegratedQSlice.
    """
    if dataset.integrated_datasets is None:
        raise ValueError(
            "No integrated datasets to work with."
        )
    if integrated_index is None:
        integrated_index = max(dataset.integrated_datasets.keys())
    integrated_dataset = dataset.integrated_datasets[integrated_index]
    reduced_dataset = {}

    for key, data in dataset.datas.items():
        integrated_q_slice = integrated_dataset[key]

        if integrated_q_slice.q_axis == 'qdx':
            qsz, qsx, _, sample_phi_rad_corr = diffraction.qxz_to_qz_qx(
                integrated_q_slice.q,
                np.zeros(shape=integrated_q_slice.q.shape, dtype=np.float64),
                data.metadata['wavelength_nm'],
                data.metadata['sample_phi_deg'],
                samplethetaoffset=data.metadata['sample_phi_offset_deg']
            )

            reduced_dataset[key] = ReducedData(
                Iq=np.copy(integrated_q_slice.Iq),
                qsx=qsx,
                qsz=qsz,
                sample_phi_deg_corr=np.rad2deg(sample_phi_rad_corr),
                wavelength_nm=data.metadata['wavelength_nm']
            )

        elif integrated_q_slice.q_axis == 'qdy':
            # TODO: implement 2D CDSAXS
            raise ValueError(
                "Currently only integrated I vs. qdx data can be reduced."
            )

        else:
            raise ValueError(
                "Currently only integrated I vs. qdx data can be reduced."
            )

    if in_place:
        if dataset.reduced_datasets is None:
            dataset.reduced_datasets = {}
        dataset.reduced_datasets[integrated_index] = reduced_dataset


def slice_reduced_dataset(
    dataset: Dataset,
    reduced_index=None,
    q_values=[],
    q_widths=0.001,
    q_axis='qsx',
    find_peaks=False,
    peak_params={},
    in_place: bool = True,
    show_plot=True,
    interpolated_image=True,
    plot_marker_size=5,
):

    """
    Creates dictionary of qsz vs. qsx one-dimensional data (sample frame)
    built from one-dimensional integrated data of I vs. qdy (detector frame).
    """

    if not isinstance(q_widths, list):
        q_widths = [q_widths]*len(q_values)
    else:
        if len(q_widths) != len(q_values):
            raise ValueError(
                "q_widths must be float or list with same length as q_values."
            )

    if reduced_index is None:
        reduced_index = max(dataset.reduced_datasets.keys())

    q_ranges = [
        (val-width/2, val+width/2) for val, width in zip(q_values, q_widths)]

    slices = {}
    for i, q_range in enumerate(q_ranges):
        qsz = []
        Iq = []

        for data in dataset.reduced_datasets[reduced_index].values():
            selection = np.where((data.qsx >= q_range[0]) &
                                 (data.qsx <= q_range[1]) &
                                 (data.Iq >= 0))[0]
            if len(selection) > 0 and not np.isnan:
                append_qsz = np.nanmean(data.qsz[selection])
                append_iq = np.nanmean(data.Iq[selection])

                if not np.isnan(append_iq):
                    qsz.append(append_qsz)
                    Iq.append(append_iq)

        slices[q_values[i]] = Data1D(q=qsz, Iq=Iq, q_axis='qsz')

    if show_plot:

        fig = plotting.plot_reduced_dataset(dataset,
                                            index=reduced_index,
                                            log_scale=True,
                                            interpolated_image=interpolated_image,
                                            plot_marker_size=plot_marker_size)

        max_qsz = 0
        min_qsz = 0

        for data in dataset.reduced_datasets[reduced_index].values():
            max_qsz = max(max_qsz, np.nanmax(data.qsz))
            min_qsz = min(min_qsz, np.nanmin(data.qsz))

        for q, q_range in zip(q_values, q_ranges):
            fig.axes[0].vlines(q, min_qsz, max_qsz, color='red', linestyles='dashed', zorder=100)
            fig.axes[0].axvspan(q_range[0], q_range[1], facecolor='red', alpha=0.3, zorder=100) 
    else:
        fig = None

    if in_place:
        if reduced_index not in dataset.reduced_slices.keys():
            dataset.reduced_slices[reduced_index] = {}
        dataset.reduced_slices[reduced_index][q_axis] = slices
        return fig
    else:
        return slices
