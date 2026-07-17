"""General tools for the code."""
import inspect
import warnings

import os
import csv
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from sklearn.linear_model import LinearRegression
from skimage import transform
from .calculators import gaussian


def default_mask(data):

    """
    Generate a boolean mask for invalid numeric values.

    Parameters
    ----------
    data : array-like
        Input numeric data.

    Returns
    -------
    mask : ndarray
        Boolean mask that is True where data contains nan, inf, or -inf.
    """
    data = np.array(data)
    mask = np.isnan(data)
    mask += np.isinf(data)
    mask += np.isneginf(data)

    return mask


def find_gaussian_peakloc(x, y, p0=None):
    """
    Find the peak location of a one-dimensional spectra using a Gaussian
    fit.

    Parameters
    ----------
    x : ndarray, float
        One-dimensional array of independent variable.
    y : ndarray, float
        One-dimensional array of dependent variable.
    p0 : list, optional
        Initial guess of the mean, std_dev, scale, and offset parameters
        of the Gaussian function.

    Returns
    -------
    peak_x : float
        Peak location based on the Gaussian fit.
    popt : list
        Optimized parameters mean, std_dev, scale, and offset from the
        Gaussian fit.
    """

    mask = default_mask(y)
    x_fit = np.array(x)[~mask]
    y_fit = np.array(y)[~mask]

    order_by_x = np.argsort(x_fit)
    x_fit = x_fit[order_by_x]
    y_fit = y_fit[order_by_x]

    if x_fit.shape[0] < 4:
        raise TypeError(
            "More than 4 data points need to be provided."
        )

    if p0 is None:
        try:
            offset = np.min(y_fit)
            amplitude = np.max(y_fit) - offset
            half_max = amplitude/2
            above_half = np.where(y_fit > half_max)[0]
            try:
                stdev = (x_fit[np.max(above_half)] - x_fit[np.min(above_half)])/2.355
            except:
                stdev = 1.0
            scale = amplitude * np.sqrt(2*np.pi)*stdev
            p0 = [x[np.argmax(y_fit)], stdev, scale, offset]
        except:
            p0 = None

    popt, _ = curve_fit(
        gaussian,
        x_fit, y_fit,
        p0=p0,
    )
    peak_x = popt[0]

    return float(peak_x), popt


def line_fit(x, y, force_intercept=None):
    """
    Fit a line to x and y data and return angle, slope, and intercept.

    The angle is defined counterclockwise from the x-axis. If the fit is
    constrained with force_intercept, the returned line passes through that
    point. In the case of a vertical line, slope and intercept are returned
    as nan.

    Parameters
    ----------
    x : array-like
        x coordinates of the data points.
    y : array-like
        y coordinates of the data points.
    force_intercept : tuple | None, optional
        Point (x, y) through which the fitted line must pass.

    Returns
    -------
    angle : float
        Line angle in degrees.
    slope : float
        Fitted slope.
    intercept : float
        Fitted intercept.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    if force_intercept is not None:
        x = x - force_intercept[0]
        y = y - force_intercept[1]

    if len(x) == 1:
        warnings.warn(
            "Only one point was provided for a line fit. Two or more" \
            "are required. Assuming a horizontal line."
        )
    try:
        model = LinearRegression(
            fit_intercept=False if force_intercept is not None else True)
        model.fit(x, y)
        slope = float(model.coef_[0][0])
        intercept = np.asarray(model.intercept_).item() # convert to scalar

        if force_intercept is not None:
            intercept = force_intercept[1] - slope * force_intercept[0]
        angle = np.rad2deg(np.arctan(slope))
    except ValueError:
        # vertical line
        angle = 90
        slope = np.nan
        intercept = np.nan

    return angle, slope, intercept


def gaussian_refine_peak_2D(image):

    """
    Refine a two-dimensional peak position with one Gaussian fit per axis.

    Parameters
    ----------
    image : ndarray
        Two-dimensional image region containing a single dominant peak.

    Returns
    -------
    a_opt : float
        Refined peak position along axis 0.
    b_opt : float
        Refined peak position along axis 1.
    """

    try:
        a_opt, _ = find_gaussian_peakloc(
            np.arange(0, image.shape[0]),
            np.nansum(image, axis=1))
    except RuntimeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 0;"
            "assuming peak is at the pixel with the highest value.")
        a_opt = int(np.nanargmax(np.nansum(image, axis=1)))
    except TypeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 0;"
            "assuming peak is at the pixel with the highest value.")
        a_opt = int(np.nanargmax(np.nansum(image, axis=1)))

    try:
        b_opt, _ = find_gaussian_peakloc(
            np.arange(0, image.shape[1]),
            np.nansum(image, axis=0))
    except RuntimeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 1;"
            "assuming peak is at the pixel with the highest value.")
        b_opt = int(np.nanargmax(np.nansum(image, axis=0)))
    except TypeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 1;"
            "assuming peak is at the pixel with the highest value.")
        b_opt = int(np.nanargmax(np.nansum(image, axis=0)))

    return float(a_opt), float(b_opt)


def rotate_image(image,
                 degrees, rotation_center,
                 resampling_mode="bilinear",
                 fill_mode="constant",
                 fill_constant=np.nan,
                 log_scale=False,
                 **kwargs):
    """

    Rotates an image by a specified number of degrees counterclockwise
    about the rotation center.

    Parameters
    ----------
    image : ndarray
        Two-dimensional image for rotation.
    rotation_center : list
        Center of rotation. Indices should be provided as [row, column]
        keeping in mind that numpy index orders rows from top to
        bottom and columns from left to right.
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
    log_scale : bool, optional
        Rotate the log-scale of your image. This could help resolve
        some artifacts caused by certain rotation sampling algorithms
        but you will lose any pixels that are negative (turned to nan).
        Deafult value is False.

    Other Parameters
    ----------------
    **kwargs
        Other keyword arguments for skimage.transform.rotate are
        accepted. These include:
            resize
            clip
            preserve_range

    Returns
    -------
    image : ndarray
        Two-dimensional rotated image of same dimensions as 'image'.
    """

    sklearn_resampling_modes = {
        "nearest_neighbor": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5,
    }

    resampling_mode = ''.join(filter(str.isalpha, resampling_mode.lower()))
    resampling_order = sklearn_resampling_modes[resampling_mode]

    fill_mode = fill_mode.lower()

    image = np.array(image)
    if log_scale:
        image = np.log10(image)
    image[default_mask(image)] = np.nan

    image = transform.rotate(
        image,
        angle=degrees,
        center=(rotation_center[1], rotation_center[0]),  # needs (col, row)
        order=resampling_order,
        mode=fill_mode,
        cval=fill_constant,
        **kwargs
    )

    if log_scale:
        image = np.power(10, image)

    return image


def rotate_image_pillow(
        image, degrees, rotation_center, resampling_mode="bilinear",
        log_scale=False, fillcolor=-9999):
    """

    Rotates an image by a specified number of degrees counterclockwise
    about the rotation center.

    Parameters
    ----------
    image : ndarray
        Two-dimensional image for rotation.
    rotation_center : list
        Center of rotation. Indices should be provided as [row, column]
        keeping in mind that numpy index orders rows from top to
        bottom and columns from left to right.
    resampling_mode : str, optional
        Set the resampling method used during the rotation.
        The box rotation works by rotating the image underneath then
        extracting the box for integration. Resampling of the
        image intensities can be performed with the 'nearest',
        'bilinear', or 'bicubic' methods in the PILLOW package.
        We encourage the user to look into the rotation sampling modes
        as this may result in 'features' in your data due to sharp
        log-scale peaks.
        Default value is 'bilinear'.
    log_scale : bool, optional
        Rotate the log-scale of your image. This could help resolve
        some artifacts caused by certain rotation sampling algorithms
        but you will lose any pixels that are negative (turned to nan).
        Deafult value is False.
    fillcolor : float, optional
        A temporary value used to fill pixels that are outside of the
        original image after rotation. These pixels will
        be replaced with NAN after the rotation is complete. A float
        is used to comply with the keyword argument requirements of
        the pillow package rotate() function used to perform the
        rotation.
        Default value is -9999.

    Returns
    -------
    image : ndarray
        Two-dimensional rotated image of same dimensions as 'image'.
    """

    if resampling_mode == 'nearest':
        resample = Image.Resampling.NEAREST
    elif resampling_mode == 'bilinear':
        resample = Image.Resampling.BILINEAR
    else:
        resample = Image.Resampling.BICUBIC

    image = np.array(image)
    if log_scale:
        image = np.log10(image)
    image[default_mask(image)] = np.nan
    # convert to PILLOW Image for the rotation
    image = Image.fromarray(image)
    image = image.rotate(
        degrees,
        resample=resample,
        # pillow calls for (x, y) of beam center
        center=(rotation_center[1], rotation_center[0]),
        fillcolor=fillcolor)
    image = np.array(image)
    image[image == fillcolor] = np.nan
    if log_scale:
        image = np.power(10, image)

    return image


def find_peaks_1D(data, log_scale=True, refinement_size=7, mask=None,
                  algorithm='scikit', refinement=True, **kwargs):
    """
    Find peaks across one-dimensional data using scikit-image.feature
    peak_local_max() function. The peak location is then further refined
    with a Gaussian fit along the single axis. Refinement is required for
    more accurate peak positions as the peak_local_max() only returns
    the positions to the nearest pixel.

    Parameters
    ----------
    data : NDArray, list
        One-dimensional data as a numpy array or list.
    log_scale : bool, optional
        If set to True, the data will be passed to the peak finding
        algorithm on a log sale of intensity. If set to False, the image
        will be sent to the peak finding algorithm with its original
        values.
        Default value is True.
    refinement_size : int
        Define the pixel range centered on the peaks in which to peform
        Gaussian refinement.
        Default value is 7. Minimum value is 4.
    mask : NDArray, list
        One-dimensional boolean array or list of points to mask during
        the peak finding operation.
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
    coordinates : NDArray
        An n x 2 array of peak coordinate positions will be returned for
        n number of peaks found.
    coordinates_px : NDArray
        An n x 2 array of peak coordinate positions rounded to the
        nearest pixels will be returned for n number of peaks found.
    """
    if algorithm not in ['scikit', 'scipy']:
        raise ValueError(
            f"The algorithm {algorithm} is not recognized. Use either "
            "'scikit' or 'scipy'."
        )

    # check the threshold_abs
    if algorithm == 'scikit':
        value = kwargs.get("threshold_abs")
        if value is None:
            kwargs["threshold_abs"] = 0

    data_fed = np.array(data)
    if mask is not None:
        data_fed[mask] = np.nan
    if log_scale:
        data_fed = np.log10(data_fed)
        data_fed[np.isneginf(data_fed)] = np.nan

    if algorithm == 'scikit':
        accepted_kwargs = [
            param.name for param in inspect.signature(
                peak_local_max).parameters.values()
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY)
            and param.default is not inspect.Parameter.empty
        ]
        coordinates_px = peak_local_max(
            data_fed,
            **{x: y for x, y in kwargs.items() if x in accepted_kwargs})
        coordinates_px = coordinates_px.tolist()
        coordinates_px = [x[0] for x in coordinates_px]
    elif algorithm == 'scipy':
        accepted_kwargs = [
            param.name for param in inspect.signature(
                find_peaks).parameters.values()
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY)
            and param.default is not inspect.Parameter.empty
        ]
        coordinates_px, _ = find_peaks(
            data_fed,
            **{x: y for x, y in kwargs.items() if x in accepted_kwargs})

    coordinates = []

    refinement_size = max(refinement_size, 4)
    if data_fed.shape[0] < 4:
        warnings.warn(
            "Data does not have enough points for refinement."
            "Using pixel location."
            )
        return np.array([x[0] for x in coordinates_px])

    for x in coordinates_px:
        x_min = max(0, x - int(refinement_size/2))
        x_max = min(x + (refinement_size - int(refinement_size/2)),
                    data_fed.shape[0])

        if x_max-x_min < 4:
            while x_max < data_fed.shape[0] and x_max-x_min < 4:
                x_max += 1
            while x_min > 0 and x_max-x_min < 4:
                x_min -= 1

        data_refine = data_fed[x_min:x_max]

        try:
            x_opt, _ = find_gaussian_peakloc(
                np.arange(0, data_refine.shape[0]),
                data_refine)
        except RuntimeError:
            warnings.warn(
                "Could not fit Gaussian to the peak location;"
                "assuming peak is at the pixel with the highest value.")
            x_opt = int(np.nanargmax(data_refine))

        x_opt += x_min

        coordinates.append(x_opt)

    coordinates = [x for x in coordinates if x < len(data) and x >= 0]

    return np.array(coordinates)


def find_peaks_2D(image, log_scale=True, refinement_size=7, mask=None,
                  **kwargs):
    """
    Find peaks across a two-dimensional image using scikit-image.feature
    peak_local_max() function and then further refined with local
    Gaussian fits across the two axes. Refinement is required for more
    accurate peak positions as the peak_local_max() only returns the
    positions to the nearest pixel.

    Parameters
    ----------
    image : NDArray
        Two-dimensional image as a numpy array.
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
    mask : NDArray
        Two-dimensional boolean array of pixels to mask during the
        peak finding operation.

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
    coordinates : NDArray
        An n x 2 array of peak coordinate positions will be returned for
        n number of peaks found.
    """
    # check the threshold_abs
    value = kwargs.pop("threshold_abs", 0)
    kwargs["threshold_abs"] = value

    image_fed = np.copy(image)
    if mask is not None:
        image_fed[mask] = np.nan
    if log_scale:
        image_fed = np.log10(image_fed)
        image_fed[np.isneginf(image_fed)] = np.nan


    accepted_kwargs = [
        param.name for param in inspect.signature(peak_local_max).parameters.values()
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY)
        and param.default is not inspect.Parameter.empty
    ]
    coordinates_px = peak_local_max(
        image_fed,
        **{x: y for x, y in kwargs.items() if x in accepted_kwargs})
    coordinates_px = coordinates_px.tolist()

    coordinates = []

    refinement_size = max(refinement_size, 4)
    if min(image.shape) < 4:
        warnings.warn(
            "Image is not large enough for refinement. Using pixel location."
            )
        return np.array(coordinates_px)

    for (y, x) in coordinates_px:
        y_min = max(0, y - int(refinement_size/2))
        y_max = min(y + (refinement_size - int(refinement_size/2)),
                    image.shape[0])

        x_min = max(0, x - int(refinement_size/2))
        x_max = min(x + (refinement_size - int(refinement_size/2)),
                    image.shape[1])

        if y_max-y_min < 4:
            while y_max < image.shape[0] and y_max-y_min < 4:
                y_max += 1
            while y_min > 0 and y_max-y_min < 4:
                y_min -= 1
        if x_max-x_min < 4:
            while x_max < image.shape[1] and x_max-x_min < 4:
                x_max += 1
            while x_min > 0 and x_max-x_min < 4:
                x_min -= 1

        image_refine = image_fed[y_min:y_max, x_min:x_max]
        y_opt, x_opt = gaussian_refine_peak_2D(image_refine)
        y_opt += y_min
        x_opt += x_min

        coordinates.append([y_opt, x_opt])

    coordinates = [(y, x) for (y, x) in coordinates
                   if y < image.shape[0] and y >= 0
                   and x < image.shape[1] and x >= 0]

    return np.array(coordinates)


def find_peaks_2D_one_axis(
        image, peak_axis, integration_mode='sum', log_scale=True,
        refinement_size=7, mask=None, algorithm='scikit', **kwargs):
    """
    Find peaks along one axis of a two-dimensional image using the
    scikit-image.feature peak_local_max() function. The peaks are
    further refined with local Gaussian fits across the two axes at the
    peak locations. Refinement is required for more accurate peak
    positions as the peak_local_max() only returns the positions to the
    nearest pixel.

    The old version of this function used scipy.signal find_peaks()
    to determine the initial peak positions. It is possible to use this
    algorithm by switching the 'algorithm' keyword argument to 'scipy'.

    This function differs from find_peaks_2D() in that it only allows
    for the primary peaks to be found along a single axis. For example,
    if axis 1 is chosen as the peak axis, the image provided will be
    integrated along axis 0 (summed or averaged) to find the primary
    peak location along axis 1. Then the peak location in axis 0 will
    be determined as the highest intensity pixel at each peak location
    along axis 1. This is then refined by the Gaussian fits.

    Parameters
    ----------
    image : NDArray
        Two-dimensional image as a numpy array.
    peak_axis : int
        The axis along which the peaks should be found. If axis 0 is
        selected, the image will be integrated along axis 1. This means
        peaks found will likely be vertical in your image. If axis 1 is
        selected, the image will be integrated along axis 0. This means
        peaks found will likely be horizontal in your image.
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
    coordinates : NDArray
        An n x 2 array of peak coordinate positions will be returned for
        n number of peaks found.
    """
    image_fed = np.copy(image)
    if mask is not None:
        image_fed[mask] = np.nan
    drop_if_any_nan = np.isnan(image_fed).any(axis=1-peak_axis)
    if integration_mode == 'sum':
        image_fed = np.nansum(image_fed, axis=1-peak_axis)
    elif integration_mode == 'mean':
        image_fed = np.nanmean(image_fed, axis=1-peak_axis)
    else:
        raise ValueError(
            f"Integration mode {integration_mode} not recognized."
            "Use 'mean' or 'sum'."
        )
    image_fed[drop_if_any_nan] = np.nan

    refinement_size = max(refinement_size, 4)
    coordinates_peak_axis = find_peaks_1D(image_fed, log_scale=log_scale,
                                          refinement_size=refinement_size,
                                          algorithm=algorithm,
                                          **kwargs)
    coordinates_peak_axis = np.round(
        np.array(coordinates_peak_axis), 0).astype(int)

    if peak_axis == 1:
        peaks_other = np.argmax(image[:, coordinates_peak_axis], axis=0)
        coordinates_px = [
            (y, x) for x, y in zip(coordinates_peak_axis, peaks_other)]
    else:
        peaks_other = np.argmax(image[coordinates_peak_axis, :], axis=1)
        coordinates_px = [
            (y, x) for y, x in zip(coordinates_peak_axis, peaks_other)]

    coordinates_px = []
    for a in coordinates_peak_axis:
        a = int(np.round(a, 0))
        if peak_axis == 1:
            coordinates_px.append([int(np.argmax(image[:, a])), a])
        elif peak_axis == 0:
            coordinates_px.append([a, int(np.argmax(image[a, :]))])

    if min(image.shape) < 4:
        warnings.warn(
            "Image is not large enough for refinement. Using pixel location."
            )
        return np.array(coordinates_px)

    coordinates = []
    if log_scale:
        image = np.log10(image)
        image[np.isneginf(image)] = np.nan
    else:
        image = np.array(image)

    for (y, x) in coordinates_px:
        y_min = max(0, y - int(refinement_size/2))
        y_max = min(y + (refinement_size - int(refinement_size/2)),
                    image.shape[0])

        x_min = max(0, x - int(refinement_size/2))
        x_max = min(x + (refinement_size - int(refinement_size/2)),
                    image.shape[1])

        if y_max-y_min < 4:
            while y_max < image.shape[0] and y_max-y_min < 4:
                y_max += 1
            while y_min > 0 and y_max-y_min < 4:
                y_min -= 1
        if x_max-x_min < 4:
            while x_max < image.shape[1] and x_max-x_min < 4:
                x_max += 1
            while x_min > 0 and x_max-x_min < 4:
                x_min -= 1

        image_refine = image[y_min:y_max, x_min:x_max]
        y_opt, x_opt = gaussian_refine_peak_2D(image_refine)
        y_opt += y_min
        x_opt += x_min

        coordinates.append([y_opt, x_opt])

    coordinates = [(y, x) for (y, x) in coordinates
                   if y < image.shape[0] and y >= 0
                   and x < image.shape[1] and x >= 0]

    return np.array(coordinates)


def find_maximum_rectangular_roi(data):
    """
    Find the largest rectangular region of valid pixels in a mask.

    The input should be a boolean array, or an array of 1 and 0 values,
    where True or 1 marks pixels that belong to the candidate region of
    interest.

    Parameters
    ----------
    data : ndarray
        Two-dimensional boolean or binary array describing valid pixels.

    Returns
    -------
    row_bounds : tuple
        Row bounds of the maximum rectangular region as (min_row, max_row).
    col_bounds : tuple
        Column bounds of the maximum rectangular region as (min_col, max_col).
    height : ndarray
        Height map used internally for the rectangle search.
    width : ndarray
        Width map used internally for the rectangle search.
    area : ndarray
        Area map used internally for the rectangle search.
    """

    data = data.astype(int)

    height = np.flipud(np.cumsum(np.flipud(data), axis=0))*data
    where_ones_start = np.where((data[1:, :] - data[:-1, :]) == 1, 1, 0)
    adjustments = height[1:, :] * where_ones_start
    adjustments = np.flipud(np.maximum.accumulate(np.flipud(adjustments), axis=0))
    height[:-1, :] = height[:-1, :] - adjustments
    height[height < 0] = 0

    width = np.fliplr(np.cumsum(np.fliplr(data), axis=1))*data
    where_ones_start = np.where((data[:, 1:] - data[:, :-1]) == 1, 1, 0)
    adjustments = width[:, 1:] * where_ones_start
    adjustments = np.fliplr(np.maximum.accumulate(np.fliplr(adjustments), axis=1))
    width[:, :-1] = width[:, :-1] - adjustments
    width[width < 0] = 0

    area = height * width

    found_it = False
    counter = 0
    while not found_it and counter <= 1e5:
        min0, min1 = np.unravel_index(np.argmax(area), area.shape)

        # take the test area assuming first row and column of
        # continuous ones sets the boundaries
        test_area = data[min0:min0+height[min0, min1], min1:min1+width[min0, min1]]

        # there could still be zeros anywhere else in test area
        # find the area for different size boxes here after excluding
        # those pixels
        areas = []
        px_x = np.tile(np.arange(0, test_area.shape[1]), test_area.shape[0])+1
        px_y = np.tile(np.arange(0, test_area.shape[0]).reshape(-1, 1), test_area.shape[1]).reshape(-1)+1
        for x, y in zip(px_x, px_y):
            if np.min(test_area[:y, :x])==0:
                areas.append(0)
            else:
                areas.append(x*y)
        x, y = px_x[np.argmax(areas)], px_y[np.argmax(areas)]
        actual_area = x*y

        # new box that's actually the biggest without zeros
        test_area = test_area[:y, :x]
        area[min0, min1] = actual_area
        new_max = np.max(area)
        if new_max == actual_area or new_max == 1:
            found_it = True

        # ideally it will find the answer quick but set a break point
        # just in case for this while loop
        counter += 1

    max0 = min0 + test_area.shape[0]
    max1 = min1 + test_area.shape[1]

    return (min0, max0), (min1, max1), height, width, area

def export_metadata(dataset, timestamp):
    """
    Export the metadata of a dataset

    Parameters
    ----------
    dataset : Dataset
        Dataset containing the metadata to export.
    
    timestamp: str
        Timestamp for the export directory.

    Returns
    -------
    None
    """
    try:
        with open(f"./export/{timestamp}/metadata_{timestamp}.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            
            # get keys
            keyslist = []
            for keys in dataset.datas.keys():
                keyslist.append(keys)
            keyslist.sort(key=lambda x: float(x))  # Sort keys numerically

            headers = ["Key"] + keyslist
            writer.writerow(headers)

            writer.writerow(["Metadata:"])

            # write metadata rows
            for key in dataset.datas[keyslist[0]].metadata.keys():
                row = [key]

                for k in keyslist:
                    row.append(dataset.datas[k].metadata.get(key))
                writer.writerow(row)
                
            writer.writerow(["User Parameters:"])

            # write user param rows
            for key in dataset.datas[keyslist[0]].user_params.keys():
                row = [key]

                for k in keyslist:
                    row.append(dataset.datas[k].user_params.get(key))
                writer.writerow(row)
            
        print(f"Metadata exported to ./export/{timestamp}/metadata_{timestamp}.csv successfully.")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def export_data_transformations(dataset, timestamp):
    """
    Export the data transformations of a dataset

    Parameters
    ----------
    dataset : Dataset
        Dataset containing the data transformations to export.
    
    timestamp: str
        Timestamp for the export directory.

    Returns
    -------
    None
    """
    try:
        with open(f"./export/{timestamp}/data_transformations_{timestamp}.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(["Transformation Type:", "Value", "Key"])

            # get keys
            keyslist = []
            for key in dataset.datas.keys():
                writer.writerow([key])

                for transform_tuple in dataset.datas[key].data_transformations:

                    list_of_transforms = []
                    for transform in transform_tuple:
                        list_of_transforms.append(transform)
                    writer.writerow(list_of_transforms)

        print(f"Data transformations exported to ./export/{timestamp}/data_transformations_{timestamp}.csv successfully.")
    except Exception as e:
        print(f"Error writing CSV: {e}")