"""General tools for the code."""

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

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
            x[y == np.max(y)][0], 1, np.max(y), 0
        ]
    )
    peak_x = popt[0]
    peak_index = np.argmin(np.abs(x-peak_x))

    return peak_x, peak_index, popt


def line_fit(x, y):
    """
    Fit a line to the x, y data and return angle, slope, intercept.
    The angle is defined counterclockwise from the x-axis.
    In the case of a vertical line, slope and intercept are returned as nan.
    """
    x = np.array(x).reshape(-1)
    y = np.array(x).reshape(-1)

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


def gaussian_find_peaks_2D(image, integrated_slice, integrated_axis,
                           peak_params, peak_find_scale='linear',
                           opt_width=(5, 5)):
    """
    Performs a 2-dimensional peak finding algorithm that is optimized
    with gaussian fits along both dimensions of the image.

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
        a_min = a - int(opt_width[0]/2)
        a_max = a + (opt_width[0] - int(opt_width[0]/2))
        b_min = b - int(opt_width[1]/2)
        b_max = b + (opt_width[1] - int(opt_width[1]/2))

        image_box = image[a_min:a_max, b_min:b_max]
        if peak_find_scale == 'log':
            image_box = np.log10(image_box)

        a_opt, _, _ = find_gaussian_peakloc(
            np.arange(a_min, a_max),
            np.sum(image_box, axis=1))
        b_opt, _, _ = find_gaussian_peakloc(
            np.arange(b_min, b_max),
            np.sum(image_box, axis=0))

        peaks_px_opt.append((a_opt, b_opt))

    return peaks_px_opt
