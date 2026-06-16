"""
Helpful plotting tools for cdsaxs.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
from scipy.spatial import KDTree
from . import _plotting_tools as plotting_tools


def create_even_axis_ticks(data, num=6, includes_zero=True):
    """
    Produces evenly spaced indices and corresponding q values
    for scattering image axes.

    The scattering vector array, q, must be sorted. It doesn't have to
    be increasing or decreasing as long as it's in order.

    If you want to include q=0 as one of the enforced tick marks,
    includes_zero should be set to True.

    TODO: figure out what happens if includes_zero=True and 0 is not in q
    """
    data = np.array(data).reshape(-1)
    sort_data = np.argsort(data)
    data = data[sort_data]

    if not includes_zero:
        ticks_index = np.arange(0, len(data))[sort_data]
        ticks_index = np.linspace(ticks_index, num=num)
        ticks_data = np.interp(ticks_index, np.arange(0, len(data)), data)

    else:
        spacing_exp = np.ceil(
            np.log10((np.nanmax(data)-np.nanmin(data))/(num-1)))
        spacing = 10**spacing_exp

        start = np.ceil(np.nanmin(data)/spacing)*spacing
        stop = np.floor(np.nanmax(data)/spacing)*spacing + spacing

        ticks_data = np.round(np.arange(start, stop, spacing),
                              int(np.abs(min(0, spacing_exp))))

        while len(ticks_data) < (num-1):
            spacing /= 2
            spacing_exp = np.floor(np.log10(spacing))

            start = np.ceil(np.nanmin(data)/spacing)*spacing
            stop = np.floor(np.nanmax(data)/spacing)*spacing + spacing
            ticks_data = np.round(np.arange(start, stop, spacing),
                                  int(np.abs(min(0, spacing_exp))))

        # confirm the last point from np.arange is correct with floating point
        # calculations (see numpy documentation)
        if ticks_data[-1] > np.nanmax(data):
            ticks_data = ticks_data[:-1]

        ticks_index = np.interp(
            ticks_data,
            data,
            np.arange(0, len(data))[sort_data])

        # make sure that the ticks are suffiently spaced in pixels
        if np.min(np.abs(np.diff(ticks_index))) < 10:
            ticks_index = [ticks_index[0], ticks_index[-1]]
            ticks_data = [ticks_data[0], ticks_data[-1]]

        return ticks_index, ticks_data


def generate_formatted_axis_label(q_axis):
    """
    Generate formatted axis label with units based on the axis string.
    This only does anything with q axes currently; everythign else it
    just returns back to you.

    """

    if q_axis[0] == 'q':

        units = r"(\mathring{\text{A}}^{-1})$"

        label = r"$q"
        if len(q_axis)==1:
            label = label + r"\thinspace" + units

        else:
            subscript = r"_{" + q_axis[1]
            if len(q_axis) > 2:
                for var in q_axis[2:]:
                    subscript += f",{var}"
            subscript += r"}\thinspace"

            label = label + subscript + units

    elif q_axis == 'Iq':

        label = 'I(q)'
    else:
        label = q_axis

    return label


def plot_data1d_add_data(
        fig,
        x,
        y,
        **kwargs
):
    fig = plt.figure(fig)
    plt.errorbar(x, y, **kwargs)
    return fig


def plot_data2d_add_roi(
        fig,
        limits_axis0,
        limits_axis1,
        color='red',
        label=None,
        fmt='-',
        **kwargs
):
    xmin, xmax = limits_axis1
    ymin, ymax = limits_axis0
    xmin -= 0.5
    xmax -= 0.5
    ymin -= 0.5
    ymax -= 0.5
    x = [xmin, xmin, xmax, xmax, xmin]
    y = [ymin, ymax, ymax, ymin, ymin]

    fig = plt.figure(fig)
    plt.errorbar(x, y, color=color, label=label, fmt=fmt, **kwargs, zorder=1000)

    plt.legend(bbox_to_anchor=(1.75, 1), loc='upper left', bbox_transform=fig.axes[1].transAxes)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    return fig


def plot_data2d_add_points(
        fig,
        x,
        y,
        color='red',
        label=None,
        fmt='o',
        **kwargs
):
    fig = plt.figure(fig)
    plt.errorbar(x, y, color=color, label=label, fmt=fmt, **kwargs, zorder=1000)

    plt.legend(bbox_to_anchor=(1.75, 1), loc='upper left', bbox_transform=fig.axes[1].transAxes)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    return fig


def plotly_update_axes(fig, x_axis=None, y_axis=None,
                       x_range=None, y_range=None,
                       x_data=None, y_data=None):

    fig.update_xaxes(ticks='outside')
    fig.update_yaxes(ticks='outside')

    if x_axis is not None:
        fig.update_xaxes(
            title=plotting_tools.generate_formatted_axis_label(x_axis))
    if x_range is not None:
        fig.update_xaxes(
            {'range': x_range}
        )

    if y_axis is not None:
        fig.update_yaxes(
            title=plotting_tools.generate_formatted_axis_label(y_axis))
    if y_range is not None:
        fig.update_yaxes(
            {'range': y_range}
        )

    if x_data is not None:
        ticks, labels = plotting_tools.create_even_q_ticks(x_data)
        fig.update_xaxes(tickvals=ticks, ticktext=labels)

    if y_data is not None:
        ticks, labels = plotting_tools.create_even_q_ticks(y_data)
        fig.update_yaxes(tickvals=ticks, ticktext=labels)

    return fig


def plotly_update_layout(fig, title=None, width=None,
                         xscale=None, yscale=None):

    if title is not None:
        fig.update_layout({'title': title})

    if width is not None:
        fig.update_layout(width=width)

    if xscale is not None:
        fig.update_layout(
            xaxis_type=xscale
        )

    if yscale is not None:
        fig.update_layout(
            yaxis_type=yscale
        )

    return fig


def plotly_update_colorbar_ticks(fig, colorbar_ticks, colorbar_labels,
                                 type='image', title=None):

    if type == 'scatter':
        fig.update_traces(
            marker_colorbar_tickvals=colorbar_ticks,
            marker_colorbar_ticktext=colorbar_labels,
            selector=dict(type='scatter')
        )
    elif type == 'image':
        fig.update_layout(
            coloraxis_colorbar={
                'tickvals': colorbar_ticks,
                'ticktext': colorbar_labels,
            })
        fig.update_layout(
            coloraxis_colorbar={
                'ticks': 'outside'
            }
        )

    if type == 'image' and title is not None:

        fig.update_layout(
            coloraxis_colorbar={
                'title': {'text': title, 'side': 'right'}
            }
        )

    return fig


def get_vmin_vmax(data, log_scale, mask=None):
    """
    Determine the vmin and vmax value for either 'linear' or 'log' scale
    of the provided data.
    """
    data = np.array(data)
    if mask is not None:
        data[mask] = np.nan

    if log_scale:
        vmin = np.log10(np.max([np.nanmin(data[data > 0]), 0.1]))
        vmax = np.log10(np.nanmax(data))

    else:
        vmin = 0
        vmax = np.nanmax(data)

    return vmin, vmax


def prepare_data_for_plotting(image, log_scale, vmin=0, mask=None):
    """
    Prepare the image for plotting. If log_scale is True, this will
    apply the log and set any pixels that were less than or equal to 0
    as -999 so that they can be flagged and set to the color black. Any
    masked points will be set as np.nan so that they show as transparent.

    For linear systems, the masked points will be set as np.nan.
    """

    plotting_image = np.copy(image)
    if mask is not None:
        plotting_image[mask] = np.nan
    if log_scale:
        mask_negative = plotting_image <= 0
        with np.errstate(divide='ignore', invalid='ignore'):
            plotting_image = np.log10(plotting_image)
            set_to_vmin = (plotting_image > 0) & (plotting_image <= vmin)
        plotting_image[mask_negative] = -999
        plotting_image[set_to_vmin] = vmin

    return plotting_image


def generate_interpolated_reduced_data(
        qsx,
        qsz,
        Iq,
        *,
        grid_size=1000,
        method: str = "cubic",
        distance_factor: float = 5,
        verbose: bool = False,
):
    """
    Parts of this function were generated by gpt-oss-120b.

    Interpolate scattered intensity data onto a regular (grid_size x grid_size) mesh,
    but **do not** let the interpolation spill over large empty regions.

    Parameters
    ----------
    qsx, qsz : 1-D ``np.ndarray``
        X- and Z-coordinates of the measured points.
    Iq : 1-D ``np.ndarray``
        Measured intensity.  Points where ``np.isnan(Iq)`` are ignored.
    grid_size : int, optional
        Number of points per axis in the output grid (default 1000 → 1M grid points).
    method : {"linear", "cubic", "nearest"}, optional
        Interpolation method passed to ``scipy.interpolate.griddata``.
        The original code used ``cubic``.
    distance_factor : float, optional
        Multiplier applied to a robust estimate of the typical nearest-neighbour spacing.
        Grid points farther away than ``distance_factor * spacing`` are set to ``np.nan``,
        effectively cutting off interpolation across gaps.
    verbose : bool, optional
        If True, prints a short summary of the masking statistics.

    Returns
    -------
    grid_x, grid_z, grid_Iq : ``np.ndarray``
        Regular mesh coordinates (shape ``(grid_size, grid_size)``) and the interpolated
        intensity.  Points that lie in a “whitespace” region are ``np.nan``.
    """

    # remove NaN entries
    remove_nan = np.isnan(Iq)
    qsx = qsx[~remove_nan]
    qsz = qsz[~remove_nan]
    Iq = Iq[~remove_nan]

    # create grid
    grid_x, grid_z = np.meshgrid(
        np.linspace(np.min(qsx), np.max(qsx), grid_size),
        np.linspace(np.min(qsz), np.max(qsz), grid_size),
    )

    # interpolate over everything
    raw_Iq = interpolate.griddata((qsx, qsz), Iq, (grid_x, grid_z), method=method)

    # build KD tree for measured points
    tree = KDTree(np.column_stack([qsx, qsz]))

    # get k-th nearest neighbors from all points,
    # use median of this data for typical spacing
    dists, _ = tree.query(np.column_stack([qsx, qsz]), k=2)
    typical_spacing = np.median(dists[:, 1])

    # find cutoff for masking
    cutoff = distance_factor * typical_spacing

    if verbose:
        print(f"[interpolator] typical nearest-neighbour spacing = {typical_spacing:.3g}")
        print(f"[interpolator] using cut-off = {cutoff:.3g}  (factor = {distance_factor})")

    # get array of each point on grid
    grid_points = np.column_stack([grid_x.ravel(), grid_z.ravel()])
    # distance for each point to its closest measured point
    min_dist, _ = tree.query(grid_points, k=1)
    mask_inside = min_dist.reshape(grid_x.shape) <= cutoff

    # apply mask, set points outside to NaN
    grid_Iq = np.where(mask_inside, raw_Iq, np.nan)

    # optional diagnostics
    if verbose:
        total = grid_x.size
        kept = np.count_nonzero(mask_inside)
        print(f"[interpolator] kept {kept}/{total} grid points "
              f"({kept/total:.1%}) within the admissible radius.")

    return grid_x, grid_z, grid_Iq

def new_interp_func(I_listoflists, x_listoflists, y_listoflists, qsx, qsz, grid_size, verbose):

    """
    From the old python gui

    Bilinear interpolation of 2D data (structured on at least one axis) to new axes
    Unlike griddata, does not generate convex hull with excess interpolated data

    Old gui took in 2D arrays for Qsx, Qsz, Iq. New interp function took in 1D array of these values.
    Current replacement is just changing the processing from .extend() to .append().

    Creates x, y axis from linspace of the 1d array.

    Args:
        I_listoflists: 2D array or list of lists, rows: old y axis, cols: old x axes, values: I at each point
        x_listoflists: 2D array or list of lists, rows: old y axis, cols: old x axes, values: old x at each point
        y_listoflists: 2D array or list of lists, rows: old y axis, cols: old x axes, values: old y at each point
        qsx: 1D array or list of x values
        qsz: 1D array or list of z values
        grid_size: size of grid
    """
    if verbose:
        print("Creating new axis from qsx, qsz")
    new_x_axis = np.linspace(np.min(qsx), np.max(qsx), grid_size)
    new_y_axis = np.linspace(np.min(qsz), np.max(qsz), grid_size)
    if verbose:
        print("Created new axis from qsx, qsz")

    if verbose:
        print("Dropping empty rows")
    I_listoflists = [i for i in I_listoflists if len(i) != 0]  # drop empty rows
    x_listoflists = [i for i in x_listoflists if len(i) != 0]
    y_listoflists = [i for i in y_listoflists if len(i) != 0]
    if verbose:
        print("Dropped empty rows")

    if verbose:
        print("Creating new helper axes")
    I_array_new_axes = np.empty((len(new_y_axis), len(new_x_axis)))
    I_array_newx_oldy = np.empty((len(I_listoflists), len(new_x_axis)))
    y_array_newx_oldy = np.empty((len(I_listoflists), len(new_x_axis)))
    if verbose:
        print("Created new helper axes")

    if verbose:
        print("Creating Iq interpolation function")
    I_fns = [interpolate.interp1d(x_listoflists[i], I_listoflists[i], bounds_error=False) for i in range(len(I_listoflists))]
    if verbose:
        print("Created Iq interpolation function")
        print("Creating y interpolation function")
    y_fns = [interpolate.interp1d(x_listoflists[i], y_listoflists[i], bounds_error=False) for i in range(len(I_listoflists))]
    if verbose:
        print("Created Iq interpolation function")
    
    if verbose:
        print("Interpolating on newx, oldy for each oldy")
    # linear interpolation on newx, oldy for each oldy
    for row in range(len(I_listoflists)):
        I_array_newx_oldy[row, :] = I_fns[row](new_x_axis)
        y_array_newx_oldy[row, :] = y_fns[row](new_x_axis)

    if verbose:
        print("Success")
        print("Interpolating on newx, newy")
    # linear interpolation on newx, newy
    for col in range(len(new_x_axis)):
        I_fn = interpolate.interp1d(y_array_newx_oldy[:, col], I_array_newx_oldy[:, col], bounds_error=False)
        I_array_new_axes[:, col] = I_fn(new_y_axis)

    if verbose:
        print("Success")

    # return_i_arr = []
    # for row in range(len(I_array_new_axes)):
    #     return_i_arr.extend(I_array_new_axes[row])

    if verbose:
        print("Meshing x and y axis")
    grid_x, grid_z = np.meshgrid(
        new_x_axis,
        new_y_axis,
    )
    if verbose:
        print("Success")
        
    return grid_x, grid_z, I_array_new_axes