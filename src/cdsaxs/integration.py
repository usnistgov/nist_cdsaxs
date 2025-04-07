"""
Includes functions for different modes of integration and
integration boxes for DataQyQxz images as part of a Dataset.
"""

from cdsaxs.data import Dataset, IntegratedDataset
import cdsaxs.image_tools as imgtools


def integrate_dataset(
        dataset: Dataset,
        integration_metadata: dict
        ) -> IntegratedDataset:
    """
    Most general dataset integration function.

    Parameters
    ----------
    dataset : Dataset
    integration_metadata : dict
        Integration details for each DataQyQxz in the Dataset. The key
        is identical to the corresponding key in Dataset.datas. The
        value is another dictionary with the following metadata:
            y_lims : Iterable of min and max indices along y-direction.
                This is a half-open range of [min, max).
            xz_lims : Iterable of min and max indices along x-direction.
                This is a half-open range of [min, max).
            calc_mode : String that specifies whether integration should
                be performed as a 'sum' or 'mean'.
            integration_axis : Axis along which the integration is
                performed. Accepted axes are 'y' or 'xz'.
    """

    integrated_data = {}
    for key, data in dataset.datas.items():
        integrated_data[key] = imgtools.integrate_image(
            data, **integration_metadata[key])

    return IntegratedDataset(dataset, integrated_data, integration_metadata)


def integrate_dataset_general_box(
        dataset: Dataset,
        box_size: tuple[int],
        offset: tuple[int] = (0, 0),
        calc_mode: str = 'sum',
        integration_axis: str = 'y',
        ) -> IntegratedDataset:
    """
    Integrate each of the DataQyQxz instances in the Dataset using
    within a box of specified dimensions.

    The integration box will be centered at the beam center unless an
    offset is specified in one or more directions. The offset value is
    provided in number of pixels. A positive offset value will shift the
    box in the positive q direction.

    Parameters
    ----------
    dataset : Dataset
        Instance of cdsaxs.data.Dataset.
    box_size : tuple[int, int]
        Tuple of length two of the box dimensions in number of pixels.
        First position is the size in the q_y direction and second
        position is the size in the q_xz direction.
    offset : tuple[int, int]
        Tuple of length two of the offset in number of pixels. First
        position corresponds to an offset in the q_y direction and the
        second position corresponds to an offset in the q_xz direction.
        A positive offset shifts the box in the positive q direction.
        Default offset is (0, 0).
    calc_mode : str, optional
        Set whether the integration is a 'sum' or 'mean' along the
        specfied axis.
        Default value is 'sum'.
    integration_axis, str, optional
        Set whether integration should be performed along the 'y'
        or 'xz' direction.
        Default value is 'y'.

    Returns
    -------
    IntegratedDataset
        Instance of cdsaxs.data.IntegratedDataset that inclues all
        one-dimensional spectra of I vs. q_y or of I vs. q_xz produced
        from the integration.
    """

    integration_metadata = {}
    for key, data in dataset.datas.items():
        y_lims, xz_lims = imgtools.find_box_limits_from_box_size(
            data, y_size=box_size[0], xz_size=box_size[1],
            y_offset=offset[0], xz_offset=offset[1]
        )
        integration_metadata[key] = {
            'y_lims': y_lims,
            'xz_lims': xz_lims,
            'calc_mode': calc_mode,
            'integration_axis': integration_axis
        }

    return integrate_dataset(dataset, integration_metadata)


def integrate_dataset_q_range(
        dataset: Dataset,
        qy_range: tuple[float],
        qxz_range: tuple[float],
        calc_mode: str = 'sum',
        integration_axis: str = 'y',
        ) -> IntegratedDataset:
    """
    Integrate each of the DataQyQxz instances in the Dataset within a
    box defined by the qy and qxz ranges.

    Parameters
    ----------
    dataset : Dataset
        Instance of cdsaxs.data.Dataset.
    qy_range : tuple[float, float]
        Tuple of length two of the min and max qy values to include in
        the integration box. Pixels that satisfy qy_min <= q < qy_max
        will be included in the box.
    qxz_range : tuple[float, float]
        Tuple of length two of the min and max qxz values to include in
        the integration box. Pixels that satisfy qxz_min <= q < qxz_max
        will be included in the box.
    calc_mode : str, optional
        Set whether the integration is a 'sum' or 'mean' along the
        specfied axis.
        Default value is 'sum'.
    integration_axis, str, optional
        Set whether integration should be performed along the 'y'
        or 'xz' direction.
        Default value is 'y'.

    Returns
    -------
    IntegratedDataset
        Instance of cdsaxs.data.IntegratedDataset that inclues all
        one-dimensional spectra of I vs. q_y or of I vs. q_xz produced
        from the integration.
    """

    integration_metadata = {}
    for key, data in dataset.datas.items():
        y_lims, xz_lims = imgtools.find_box_limits_from_q_ranges(
            data, qy_range, qxz_range)
        integration_metadata[key] = {
            'y_lims': y_lims,
            'xz_lims': xz_lims,
            'calc_mode': calc_mode,
            'integration_axis': integration_axis
        }

    return integrate_dataset(dataset, integration_metadata)
