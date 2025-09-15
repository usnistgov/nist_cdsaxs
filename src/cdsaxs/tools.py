"""General tools for the code."""

import warnings

import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
from skimage.feature import peak_local_max

from cdsaxs.calculators import gaussian


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
    float
        Peak location based on the Gaussian fit.
    list
        Optimized parameters mean, std_dev, scale, and offset from the
        Gaussian fit.
    """
    popt, _ = curve_fit(
        gaussian,
        x, y,
        p0=p0 if p0 is not None else [
            x[np.nanargmax(y)], 1, np.max(y), 0
        ]
    )
    peak_x = popt[0]
    peak_index = np.argmin(np.abs(x-peak_x))

    return float(peak_x), int(peak_index), popt


def line_fit(x, y):
    """
    Fit a line to the x, y data and return angle, slope, intercept.
    The angle is defined counterclockwise from the x-axis.
    In the case of a vertical line, slope and intercept are returned as nan.
    """
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)

    if len(x) == 1:
        warnings.warn(
            "Only one point was provided for a line fit. Two or more" \
            "are required. Assuming a horizontal line."
        )
    try:
        fit = linregress(x, y)
        angle = np.rad2deg(np.arctan(fit.slope))
        slope, intercept = (fit.slope, fit.intercept)
    except ValueError:
        # vertical line
        angle = 90
        slope = np.nan
        intercept = np.nan

    return angle, slope, intercept


def gaussian_refine_peak_2D(image):

    """
    Refine a peak in 2D with two Gaussian fits, one along either axis.
    """

    try:
        a_opt, _, _ = find_gaussian_peakloc(
            np.arange(0, image.shape[0]),
            np.nansum(image, axis=1))
    except RuntimeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 0;"
            "assuming peak is at the pixel with the highest value.")
        a_opt = int(np.nanargmax(np.nansum(image, axis=1)))

    try:
        b_opt, _, _ = find_gaussian_peakloc(
            np.arange(0, image.shape[1]),
            np.nansum(image, axis=0))
    except RuntimeError:
        warnings.warn(
            "Could not fit Gaussian to the peak location along axis 1;"
            "assuming peak is at the pixel with the highest value.")
        b_opt = int(np.nanargmax(np.nansum(image, axis=0)))

    return float(a_opt), float(b_opt)


def find_peaks_2D_legacy(
        image, integrated_slice, integrated_axis,
        peak_params, peak_find_scale='linear',
        opt_width=(7, 7)):
    """
    Performs a 2-dimensional peak finding algorithm that is optimized
    with gaussian fits along both dimensions of the image.

    If a gaussian fit fails to the data, try increasing the optimization
    box width. Otherwise, the function will assume the peak location is
    at the maximum value pixel.

    Returns list of peak coordinates.
    """

    if peak_find_scale == 'linear':
        peaks, _ = find_peaks(integrated_slice, **peak_params)
    elif peak_find_scale == 'log':
        peaks, _ = find_peaks(
            np.log10(integrated_slice), **peak_params)
    else:
        raise ValueError(
            f"The peak_find_scale {peak_find_scale} is not recognized."
        )

    if integrated_axis == 0:
        peaks_other = np.argmax(image[:, peaks], axis=0)
        peaks_px = [
            (y, x) for x, y in zip(peaks, peaks_other)]
    else:
        peaks_other = np.argmax(image[peaks, :], axis=1)
        peaks_px = [
            (y, x) for y, x in zip(peaks, peaks_other)]

    peaks_px_opt = []
    for (a, b) in peaks_px:
        a_min = max(0, a - int(opt_width[0]/2))
        a_max = min(a + (opt_width[0] - int(opt_width[0]/2)), image.shape[0])
        b_min = max(0, b - int(opt_width[1]/2))
        b_max = min(b + (opt_width[1] - int(opt_width[1]/2)), image.shape[1])

        image_box = image[a_min:a_max, b_min:b_max]
        if peak_find_scale == 'log':
            image_box = np.log10(image_box)

        a_opt, b_opt = gaussian_refine_peak_2D(image_box)
        a_opt += a_min
        b_opt += b_min

        peaks_px_opt.append((a_opt, b_opt))

    return peaks_px_opt


def rotate_image(image,
                 degrees, rotation_center, resampling_mode="bicubic",
                 fillcolor=-9999):
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
    rotation_sampling_mode : str
        Set the resampling method used during the rotation.
        The box rotation works by rotating the image underneath then
        extracting the box for integration. Resampling of the
        image intensities can be performed with the 'nearest',
        'bilinear', or 'bicubic' methods in the PILLOW package.
        Default value is 'bicubic'.
    fillcolor : float
        A temporary value used to fill pixels that are outside of the
        original image after rotation. These pixels will
        be replaced with NAN after the rotation is complete. A float
        is used to comply with the keyword argument requirements of
        the pillow package rotate() function used to perform the
        rotation.
        Default value is -9999.

    Returns
    -------
    ndarray
        Two-dimensional rotated image of same dimensions as 'image'.
    """

    if resampling_mode == 'nearest':
        resample = Image.Resampling.NEAREST
    elif resampling_mode == 'bilinear':
        resample = Image.Resampling.BILINEAR
    else:
        resample = Image.Resampling.BICUBIC

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

    return image


def find_peaks_1D(data, log_scale=True, refinement_size=7,
                  algorithm='scikit', **kwargs):
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
    if log_scale:
        data_fed = np.log10(data_fed)

    if algorithm == 'scikit':
        coordinates_px = peak_local_max(data_fed, **kwargs)
        coordinates_px = coordinates_px.tolist()
        coordinates_px = [x[0] for x in coordinates_px]
    elif algorithm == 'scipy':
        coordinates_px, _ = find_peaks(data_fed, **kwargs)

    coordinates = []

    refinement_size = max(refinement_size, 4)
    if data_fed.shape[0] < 4:
        warnings.warn(
            "Data does not have enough points for refinement."
            "Using pixel location."
            )
        return [x[0] for x in coordinates_px]

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
            x_opt, _, _ = find_gaussian_peakloc(
                np.arange(0, data_refine.shape[0]),
                data_refine)
        except RuntimeError:
            warnings.warn(
                "Could not fit Gaussian to the peak location;"
                "assuming peak is at the pixel with the highest value.")
            x_opt = int(np.nanargmax(data_refine))
        x_opt += x_min

        coordinates.append(x_opt)

    return coordinates


def find_peaks_2D(image, log_scale=True, refinement_size=7, **kwargs):
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
        An n x 2 array of peak coordinate positions will be returned for
        n number of peaks found.
    NDArray
        An n x 2 array of peak coordinate positions rounded to the
        nearest pixels will be returned for n number of peaks found.
    """
    # check the threshold_abs
    value = kwargs.get("threshold_abs")
    if value is None:
        kwargs["threhold_abs"] = 0

    image_fed = np.copy(image)
    if log_scale:
        image_fed = np.log10(image_fed)

    coordinates_px = peak_local_max(image_fed, **kwargs)
    coordinates_px = coordinates_px.tolist()

    coordinates = []

    refinement_size = max(refinement_size, 4)
    if min(image.shape) < 4:
        warnings.warn(
            "Image is not large enough for refinement. Using pixel location."
            )
        return coordinates_px

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

    return coordinates


def find_peaks_2D_one_axis(
        image, peak_axis, integration_mode='sum', log_scale=True,
        refinement_size=7, algorithm='scikit', **kwargs):
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
    NDArray
        An n x 2 array of peak coordinate positions will be returned for
        n number of peaks found.
    """
    # check the threshold_abs
    image_fed = np.copy(image)
    if integration_mode == 'sum':
        image_fed = np.nansum(image_fed, axis=1-peak_axis)
    elif integration_mode == 'mean':
        image_fed = np.nanmean(image_fed, axis=1-peak_axis)
    else:
        raise ValueError(
            f"Integration mode {integration_mode} not recognized."
            "Use 'mean' or 'sum'."
        )

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
        return coordinates_px

    coordinates = []
    if log_scale:
        image = np.log10(image)
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

    return coordinates
