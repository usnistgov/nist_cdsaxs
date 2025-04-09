"""Simple image processing tools relevant to CD-SAXS data reduction."""

import numpy as np

from cdsaxs.data import DataQyQxz


def find_box_limits_from_box_size(
        data: DataQyQxz,
        y_size: int,
        xz_size: int,
        y_offset: int,
        xz_offset: int
        ):
    """
    Find the index limits in y and xz directions from a box with dimensions
    of y_size by xz_size offset from the beam center by y_offset and xz_offset.
    """
    beam_center_px = data.metadata['center_px']

    y_min = beam_center_px[0] - int(y_size/2) - y_offset
    y_max = y_min + y_size

    xz_min = beam_center_px[1] - int(xz_size/2) - xz_offset
    xz_max = xz_min + xz_size

    # restrict indices to the image
    y_min = max(y_min, 0)
    xz_min = max(xz_min, 0)
    image_shape = data.imgdata.shape
    y_max = min(y_max, image_shape[0]-1)
    xz_max = min(xz_max, image_shape[1]-1)

    return (y_min, y_max), (xz_min, xz_max)


def find_box_limits_from_q_ranges(
        data: DataQyQxz,
        qy_range: list[float],
        qxz_range: list[float]
        ):
    """
    Find the index limits in y and xz directions from the q limits specified
    in either direction.
    """

    y_indices = np.where((data.qys >= qy_range[0])
                         & (data.qys < qy_range[1]))[0]
    y_lims = (np.min(y_indices), np.max(y_indices)+1)

    xz_indices = np.where((data.qxzs >= qxz_range[0])
                          & (data.qxzs < qxz_range[1]))[0]
    xz_lims = (np.min(xz_indices), np.max(xz_indices)+1)

    return y_lims, xz_lims


def integrate_image(
        data: DataQyQxz,
        y_lims,
        xz_lims,
        calc_mode,
        integration_axis
        ):
    """
    Simple integration of a single image once the box limits are known.
    """
    if integration_axis not in ['y', 'xz']:
        raise ValueError(
            f"Integration axis {integration_axis} is not recognized.")
    axis_index = 0 if integration_axis == 'y' else 1

    integrated_q = data.qxzs[xz_lims[0]: xz_lims[1]]\
        if integration_axis == 'y'\
        else data.qys[y_lims[0]: y_lims[1]]

    if calc_mode == 'sum':
        integrated_i = np.nansum(
            data.imgdata[y_lims[0]: y_lims[1], xz_lims[0]: xz_lims[1]],
            axis=axis_index)
    elif calc_mode == 'mean':
        integrated_i = np.nanmean(
            data.imgdata[y_lims[0]: y_lims[1], xz_lims[0]: xz_lims[1]]
        )
    else:
        raise ValueError(
            f"Integration calculation mode of {calc_mode} is not recognized."
        )

    return integrated_q, integrated_i
