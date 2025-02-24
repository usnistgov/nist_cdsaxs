# -*- coding: utf-8 -*-
"""
This is part of the CDSAXS Data Processing GUI.
This module contains various utility functions.
"""

from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import *

import numpy as np
from scipy import interpolate
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D


def expand(wave, min_value, max_value, num_points_between_min_max):
    """Expands a section of two-column array
    Args:
        wave: first column: index, second column: value
        min_value, max_value: float
        num_points_between_min_max: int
    """
    min_index = find_closest_index(wave[:, 0], min_value)
    max_index = find_closest_index(wave[:, 0], max_value) + 1
    num_points_total = num_points_between_min_max + min_index + wave.shape[0] - max_index
    max_index_new = num_points_between_min_max + min_index
    new_wave = np.empty((num_points_total, 2))
    new_wave[0:min_index, :] = wave[0:min_index, :]
    new_wave[max_index_new:, :] = wave[max_index:, :]
    new_wave[min_index:max_index_new, 0] = np.linspace(min_value, max_value, num_points_between_min_max)
    new_wave[min_index:max_index_new, 1] = np.interp(new_wave[min_index:max_index_new, 0], wave[:, 0], wave[:, 1])
    return new_wave


def find_closest_index(a_list, value):
    """Finds index of point in list with a value closest to input value"""
    return np.nanargmin(np.abs(np.asarray(a_list) - value))


def find_interp_index(a_list, value):
    """Finds floating point index in list with interpolated value equaling input value"""
    # np.interp arguments: interp x, [x data], [y data]
    return np.interp(value, a_list, range(len(a_list)))


def find_closest_indices(an_axis, min_extent, max_extent):
    """
    Args:
        an_axis: 1D monotonic, data units at the center of each pixel
        min_extent
        max_extent: data units
    Returns:
        slice_start, slice_end: indices of an_axis that will return all pixels whose centers fall inclusively within extents
        note that slice_end is the last pixel + 1
    """
    min_axis = np.asarray(an_axis, dtype=np.float) - min_extent
    min_axis[min_axis < 0] = np.nan
    slice_start = np.nanargmin(min_axis)
    max_axis = np.asarray(an_axis, dtype=np.float) - max_extent
    max_axis[max_axis > 0] = np.nan
    slice_end = np.nanargmin(np.abs(max_axis)) + 1
    return slice_start, slice_end


def find_closest_index_2D(xs, ys, x, y):
    """
    Args:
        xs, ys: coordinates in 2D separated into two lists
        x, y: coordinates of one point in 2D
    Returns index of point in xs, ys that is closest to (x, y)
    """
    return np.nanargmin(np.sqrt((np.asarray(xs) - x) ** 2 + (np.asarray(ys) - y) ** 2))


def find_interp_value(a_list, index):
    """Finds interpolated value at a given floating point index"""
    return np.interp(index, (np.floor(index), np.ceil(index)),
                     (a_list[int(np.floor(index))], a_list[int(np.ceil(index))]))


def bilinear_arbitrary_axes(I_listoflists, x_listoflists, y_listoflists, new_x_axis, new_y_axis, interp_y_first=False):
    """Bilinear interpolation of 2D data (structured on at least one axis) to new axes
    Unlike griddata, does not generate convex hull with excess interpolated data

    Args:
        I_listoflists: 2D array or list of lists, rows: old y axis, cols: old x axes, values: I at each point
        x_listoflists: 2D array or list of lists, rows: old y axis, cols: old x axes, values: old x at each point
        y_listoflists: 2D array or list of lists, rows: old y axis, cols: old x axes, values: old y at each point
        new_x_axis: 1D array or list
        new_y_axis: 1D array or list
        interp_y_first: bool, will interpolate y axis before x axis
    """
    if interp_y_first:
        x_listoflists, y_listoflists = y_listoflists, x_listoflists
        new_x_axis, new_y_axis = new_y_axis, new_x_axis
    I_listoflists = [i for i in I_listoflists if len(i) != 0]  # drop empty rows
    x_listoflists = [i for i in x_listoflists if len(i) != 0]
    y_listoflists = [i for i in y_listoflists if len(i) != 0]
    I_array_new_axes = np.empty((len(new_y_axis), len(new_x_axis)))
    I_array_newx_oldy = np.empty((len(I_listoflists), len(new_x_axis)))
    y_array_newx_oldy = np.empty((len(I_listoflists), len(new_x_axis)))
    I_fns = [interpolate.interp1d(x_listoflists[i], I_listoflists[i], bounds_error=False) for i in range(len(I_listoflists))]
    y_fns = [interpolate.interp1d(x_listoflists[i], y_listoflists[i], bounds_error=False) for i in range(len(I_listoflists))]
    # linear interpolation on newx, oldy for each oldy
    for row in range(len(I_listoflists)):
        I_array_newx_oldy[row, :] = I_fns[row](new_x_axis)
        y_array_newx_oldy[row, :] = y_fns[row](new_x_axis)
    # linear interpolation on newx, newy
    for col in range(len(new_x_axis)):
        I_fn = interpolate.interp1d(y_array_newx_oldy[:, col], I_array_newx_oldy[:, col], bounds_error=False)
        I_array_new_axes[:, col] = I_fn(new_y_axis)
    if interp_y_first:
        I_array_new_axes = np.transpose(I_array_new_axes)
    return I_array_new_axes


def make_masked(data, rect_dims, rowlist, collist):
    """Make sheared rectangle from 5 parameters, replaces points not in shape with np.nan

    Args:
        data: 2D array
        rect_dims: processing.RectDims ['left', 'bottom', 'width', 'height', 'angle_deg', 'threshold', 'width_peaks']
        (angle_deg: 0 is vertical box, positive is counter clockwise)
        rowlist: centers of pixels in data coordinates, vertical axis
        collist: centers of pixels in data coordinates, horizontal axis

    Returns:
        rect: mpl.patches.Rectangle object on original data
        data_masked: 2D data cropped and with points not in rect or edge replaced by np.nan
        rowlist_masked, collist_masked: 1D centers of pixels of cropped data in data coordinates
    """
    left_q = find_interp_value(collist, rect_dims.left)
    bottom_q = find_interp_value(rowlist, rect_dims.bottom)
    right_q = find_interp_value(collist, rect_dims.left + rect_dims.width)
    top_q = find_interp_value(rowlist, rect_dims.bottom + rect_dims.height)
    rect = Rectangle((left_q, bottom_q), right_q - left_q, top_q - bottom_q, color='red', alpha=0.2)
    # translate to origin, shear, translate back to original position
    transform = Affine2D().translate(-left_q, -bottom_q).skew_deg(-rect_dims.angle_deg, 0).translate(left_q, bottom_q)
    rect.set_transform(transform)
    rect_extents = rect.get_extents()
    bottom_px, top_px = find_closest_indices(rowlist, rect_extents.ymin, rect_extents.ymax)
    rowlist_masked = rowlist[bottom_px: top_px].copy()
    left_px, right_px = find_closest_indices(collist, rect_extents.xmin, rect_extents.xmax)
    collist_masked = collist[left_px: right_px].copy()
    # crops data to extents of rectangle
    # cropped arrays refer to uncropped ones using extra memory, copy them so garbage collector can delete uncropped
    data_masked = data[bottom_px: top_px, left_px: right_px].copy()
    # finds points within or on rectangle (due to radius=)
    A, B = np.meshgrid(collist_masked, rowlist_masked)
    coords = np.hstack((A.reshape((-1, 1)), B.reshape((-1, 1))))
    path = rect.get_path().transformed(rect.get_transform())
    keep = path.contains_points(coords, radius=0.000001).reshape(len(rowlist_masked), len(collist_masked))
    data_masked[~keep] = np.nan
    return rect, data_masked, rowlist_masked, collist_masked


def to1D(data, rowlist, collist, integrate_axis, subtract_value=None, reduce_fn=np.mean, absI=False, removenan=True):
    """Flattens 2D data to 1D, ignoring NaNs (make sure reduce_fn is NaN compatible if data has NaNs)

    Args:
        data: 2D array
        rowlist: centers of pixels in data coordinates, vertical axis
        collist: centers of pixels in data coordinates, horizontal axis
        integrate_axis: 0 for reducing along rows (reducing each col), 1 for reducing along columns (reducing each row)
        subtract_value: float to subtract from data, e.g. minimum of data
        reduce_fn: function that flattens data: mean or sum
        absI: whether to take absolute value after flattening (assuming negative values are noise, this is to improve log plots)
        removenan: whether to remove nan values after flattening

    Returns:
        I: 1D array of intensities
        ylist: list of midpoints of non-flattened axis
        x_midpt: midpoint of flattened axis
        xylist: list of sqrt(x**2+y**2) of midpoints
    """
    if subtract_value is not None:
        data = data - subtract_value
    I = reduce_fn(data, axis=integrate_axis)
    if absI:
        I = abs(I)
    ylist = rowlist if integrate_axis == 1 else collist
    temp0, temp1 = np.meshgrid(collist, rowlist)
    xlisttiled = temp0 if integrate_axis == 1 else temp1
    xlisttiled[np.isnan(data)] = np.nan
    xmedians = np.nanmedian(xlisttiled, axis=integrate_axis)
    if removenan:
        ylist = ylist[~np.isnan(I)]
        xmedians = xmedians[~np.isnan(I)]
        I = I[~np.isnan(I)]
    xylist = np.sign(ylist) * np.sqrt(xmedians ** 2 + ylist ** 2)
    return I, ylist, xylist


def make_rects(data, rowlist, collist, integrate_axis, bg_q=0, rect_midpt_q=None, rect_width_q=None):
    """Make background-subtracted 2D arrays and rectangle objects
    Different from make_masked in that rectangle can't be tilted and extends throughout all data in one axis but can handle backgrounds

    Args:
        data: array containing non-background and background (if any)
        rowlist: centers of pixels in data coordinates, vertical axis
        collist: centers of pixels in data coordinates, horizontal axis
        integrate_axis: 0 for reducing along rows (reducing each col), 1 for reducing along columns (reducing each row)
        bg_q: Width of the background slices on both sides of non-background to use as background. Defaults to 0.
        rect_midpt_q
        rect_width_q: Defines the subset of data (inclusive) containing non-background.
            Defaults to None to use entire data array minus bg_q.

    Returns:
        data_cropped: 2D array containing background-subtracted data
        rowlist_cropped, collist_cropped: 1D arrays
        rect, rect_bg: mpl.patches.Rectangle objects containing nonbackground data, nonbackground data
    """
    if bg_q < 1e-7:  # GUI doubleSpinBox won't give exactly 0
        bg_q = 0
    if rect_width_q is not None:
        rect_width_q = rect_width_q + 2 * bg_q  # defines the subset of data containing non-background and background

    if rect_midpt_q is None or rect_width_q is None:
        left_q_bg = collist[0] - 0.5 * (collist[1] - collist[0])
        right_q_bg = collist[-1] + 0.5 * (collist[-1] - collist[-2])
        bottom_q_bg = rowlist[0] - 0.5 * (rowlist[1] - rowlist[0])
        top_q_bg = rowlist[-1] + 0.5 * (rowlist[-1] - rowlist[-2])
    elif integrate_axis == 1:
        left_q_bg = rect_midpt_q - rect_width_q / 2
        right_q_bg = rect_midpt_q + rect_width_q / 2
        bottom_q_bg = rowlist[0] - 0.5 * (rowlist[1] - rowlist[0])
        top_q_bg = rowlist[-1] + 0.5 * (rowlist[-1] - rowlist[-2])
    elif integrate_axis == 0:
        left_q_bg = collist[0] - 0.5 * (collist[1] - collist[0])
        right_q_bg = collist[-1] + 0.5 * (collist[-1] - collist[-2])
        bottom_q_bg = rect_midpt_q - rect_width_q / 2
        top_q_bg = rect_midpt_q + rect_width_q / 2
    else:
        raise ValueError

    if integrate_axis == 1:
        left_q = left_q_bg + bg_q
        right_q = right_q_bg - bg_q
        bottom_q = bottom_q_bg
        top_q = top_q_bg
    elif integrate_axis == 0:
        left_q = left_q_bg
        right_q = right_q_bg
        bottom_q = bottom_q_bg + bg_q
        top_q = top_q_bg - bg_q
    else:
        raise ValueError

    rect = Rectangle((left_q, bottom_q), right_q - left_q, top_q - bottom_q, fill=True, color='red', alpha=0.2)
    bottom_px, top_px = find_closest_indices(rowlist, bottom_q, top_q)
    rowlist_cropped = rowlist[bottom_px: top_px]
    left_px, right_px = find_closest_indices(collist, left_q, right_q)
    collist_cropped = collist[left_px: right_px]
    data_cropped = data[bottom_px: top_px, left_px: right_px].copy()

    if bg_q != 0:
        rect_bg = Rectangle((left_q_bg, bottom_q_bg), right_q_bg - left_q_bg,
                            top_q_bg - bottom_q_bg, fill=True, color='orange', alpha=0.1)
        if integrate_axis == 1:
            left_px_bg, right_px_bg = find_closest_indices(collist, left_q_bg, right_q_bg)
            bg_left = data[bottom_px: top_px, left_px_bg: left_px]
            bg_right = data[bottom_px: top_px, right_px: right_q_bg + 1]
        elif integrate_axis == 0:
            bottom_px_bg, top_px_bg = find_closest_indices(rowlist, bottom_q_bg, top_q_bg)
            bg_left = data[bottom_px_bg: bottom_px, left_px: right_px]
            bg_right = data[top_px: top_px_bg + 1, left_px: right_px]
        else:
            raise ValueError
        data_cropped -= (np.nan_to_num(np.nanmean(bg_left, axis=integrate_axis, keepdims=True)) +
                         np.nan_to_num(np.nanmean(bg_right, axis=integrate_axis, keepdims=True))) / 2
    else:
        rect_bg = None

    return data_cropped, rowlist_cropped, collist_cropped, rect, rect_bg
