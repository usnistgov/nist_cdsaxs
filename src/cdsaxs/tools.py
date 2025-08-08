"""General tools for the code."""

import numpy as np
from scipy.optimize import curve_fit

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
    peak_x = popt[1]
    peak_index = np.argmin(np.abs(x-peak_x))

    return peak_x, peak_index, popt
