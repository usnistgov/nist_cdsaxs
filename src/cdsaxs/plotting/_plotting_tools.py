"""
Helpful plotting tools for cdsaxs.
"""

import numpy as np
from numpy.typing import NDArray
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go

import cdsaxs.plotting._plotting_tools as plotting_tools
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cdsaxs.diffraction as diffraction


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


def plot2D_interactive(
        image: NDArray,
        mask=None,
        axis0=None,
        axis1=None,
        axis0_type=None,
        axis1_type=None,
        title=None,
        log_scale=True,
        vmin=None,
        vmax=None,
        cmap='viridis',
        showcolorbar=True,
        width=750,
        aspect='equal'):
    # TODO axis not rendering in vs code notebook - KNOWN ISSUE VSCODE/PLOTLY

    vmin, vmax = get_vmin_vmax(image, log_scale=log_scale, mask=mask)
    plot_image = prepare_data_for_plotly_colorbar(
        image, log_scale=log_scale, mask=mask, vmin=vmin)

    # setup custom colorscale
    if not log_scale:
        custom_colorscale = cmap
        global_min = vmin
    else:
        standard_scale = plotly.colors.sample_colorscale(
            cmap, sample_points=list(np.linspace(0, 1, 101))
        )
        custom_colorscale = [[0, 'black']]  # -10 points will be black
        global_min = vmin-1e-15

        for val, color in zip(np.linspace(0, 1, 101), standard_scale):
            scale_val = vmin + val * (vmax - vmin)
            normalized_val = (scale_val - global_min) / (vmax - global_min)
            custom_colorscale.append([normalized_val, color])

    # plot the image keeping the aspect ratio of equal for square pixels
    fig = px.imshow(plot_image, zmin=global_min, zmax=vmax,
                    color_continuous_scale=custom_colorscale, aspect=aspect)

    fig = plotly_update_axes(
        fig,
        x_axis=axis1_type, y_axis=axis0_type,
        x_data=axis1, y_data=axis0
    )

    if log_scale and showcolorbar:
        colorbar_ticks = list(np.arange(
            vmin, np.ceil(vmax) if vmax % 1 > 0 else np.ceil(vmax)+1, step=1))
        colorbar_labels = [10**x for x in colorbar_ticks]
        colorbar_labels = [f"{x:.{0}e}" for x in colorbar_labels]
        fig = plotly_update_colorbar_ticks(
            fig, colorbar_ticks=colorbar_ticks,
            colorbar_labels=colorbar_labels, type='image')

    fig.update_layout(
        width=width
    )
    if showcolorbar:
        fig.update_layout(
            coloraxis_colorbar={
                'title': {'text': 'Intensity', 'side': 'right'},
                'ticks': 'outside',
            })
    else:
        fig.update_coloraxes(showscale=False)

    if title:
        fig.update_layout({'title': title})

    fig.update_layout(legend=dict(x=1.5, y=1, yanchor="top"))

    return fig


def plot2D_add_ROI_interactive(
    fig,
    rois,
    roi_colors=None,
    roi_line_style=None,
    name=None,
    showlegend=False
):
    """
    rois : list
        List of rectangular regions of interest. Each ROI in the list
        is defined by a tuple of two tuples that represent the limits
        along each axis: 
        ((min0, max0), (min1, max1))
    """

    for i, ((min0, max0), (min1, max1)) in enumerate(rois):
        min0 -= 0.5
        max0 -= 0.5
        min1 -= 0.5
        max1 -= 0.5
        x = [min1, min1, max1, max1, min1]
        y = [min0, max0, max0, min0, min0]
        if roi_colors is not None:
            color = roi_colors[i]
        else:
            color = 'red'
        if roi_line_style is not None:
            linestyle = roi_line_style[i]
        else:
            linestyle = 'solid'
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', line=dict(color=color, dash=linestyle),
            name=name, showlegend=showlegend
        ))

        return fig


def plot2D_add_points_interactive(
    fig,
    points,
    point_colors=None,
    name=None,
    showlegend=False,
):
    """
    points : list
        List of tuples that contain x and y traces to add as scatter
        points to a 2D image plot. The x and y can either be one value
        as a single point or an iterable of points that will all get
        plotted as the same trace.
        (x, y) or [([x1, x2], [y1, y2]), (x3, y3)]
    """

    for i, (x, y) in points:
        if point_colors is not None:
            color = point_colors[i]
        else:
            color = None
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(color=color) if color is not None else {},
            name=name, showlegend=showlegend
        ))

    return fig


def plot1D_interactive(x, y, error_y=None, mask=None,
           axis_x_type=None, axis_y_type=None,
           axis_x_lims=None, axis_y_lims=None,
           color=None, title=None, showlegend=False, name=None,
           linestyle=None, width=500, log_scale=True):

    if mask is not None:
        y[mask] = np.nan
        if error_y is not None:
            error_y[mask] = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode=linestyle if linestyle is not None else "lines+markers",
        error_y={} if error_y is None else dict(
            type='data',
            array=error_y,
            visible=True,
        ),
        line=dict(color=color) if color is not None else {},
        name=name,
        showlegend=showlegend,
        legendrank=200
    ))

    if axis_x_type is not None:
        fig.update_xaxes(
            title=plotting_tools.generate_formatted_axis_label(
                axis_x_type
            ),
            ticks='outside'
        )
    else:
        fig.update_xaxes(
            title='x',
            ticks='outside'
        )

    if axis_y_type is not None:
        fig.update_yaxes(
            title=plotting_tools.generate_formatted_axis_label(
                axis_y_type
            ),
            ticks='outside'
        )
    else:
        fig.update_yaxes(
            title='y',
            ticks='outside'
        )

    fig.update_layout(
        width=width,
    )

    if log_scale:
        fig.update_layout(
            yaxis_type="log"
        )

    if axis_x_lims is not None:
        fig.update_xaxes(
            {'range': (axis_x_lims[0], axis_x_lims[1])}
        )
    if axis_y_lims is not None:
        if log_scale:
            fig.update_yaxes(
                {'range': (np.log10(axis_y_lims[0]), np.log10(axis_y_lims[1]))}
            )
        else:
            fig.update_yaxes(
                {'range': (axis_y_lims[0], axis_y_lims[1])}
            )
    else:
        if log_scale:
            fig.update_yaxes(
                {'range': (0, np.log10(np.nanmax(y)*10))}
            )
        else:
            fig.update_yaxes(
                {'range': (0, np.nanmax(y)*1.05)}
            )

    if title:
        fig.update_layout({'title': title})

    return fig


def plot1D_add_trace_interactive(fig, x, y, error_y=None, mask=None,
                     showlegend=False, name=None, linestyle=None):

    if mask is not None:
        y[mask] = np.nan
        if error_y is not None:
            error_y[mask] = None

    # add trace behind
    fig_data = [go.Scatter(
        x=x,
        y=y,
        mode=linestyle if linestyle is not None else "lines+markers",
        error_y={} if error_y is None else dict(
            type='data',
            array=error_y,
            visible=True,
        ),
        name=name,
        showlegend=showlegend,
        legendrank=2000
    ),] + list(fig.data)
    fig.data = []
    for trace in fig_data:
        fig.add_trace(trace)

    return fig


def generate_interpolated_integrated_data(
        qd,
        y,
        Iq,
        grid_size=1000,
):

    grid_x, grid_y = np.meshgrid(
        np.linspace(np.min(qd), np.max(qd), grid_size),
        np.linspace(np.min(y), np.max(y), grid_size)
    )
    grid_Iq = griddata((qd, y), Iq, (grid_x, grid_y), method='cubic')

    return grid_x, grid_y, grid_Iq


def generate_interpolated_reduced_data(
        qsx,
        qsz,
        Iq,
        wavelength_nm,
        sample_phi_deg_range,
        grid_size=1000,
):

    remove_nan = np.isnan(Iq)
    qsx = qsx[~remove_nan]
    qsz = qsz[~remove_nan]
    Iq = Iq[~remove_nan]

    grid_x, grid_z = np.meshgrid(
        np.linspace(np.min(qsx), np.max(qsx), grid_size),
        np.linspace(np.min(qsz), np.max(qsz), grid_size)
    )
    grid_Iq = griddata((qsx, qsz), Iq, (grid_x, grid_z), method='cubic')

    sample_phi_grid = np.array(
        diffraction.qx_qz_to_sample_theta(wavelength_nm, grid_x, grid_z))
    sample_phi_grid = np.rad2deg(sample_phi_grid[0, :, :])

    filter_out = (
            sample_phi_grid > np.max(sample_phi_deg_range)
        ) | (
            sample_phi_grid < np.min(sample_phi_deg_range)
        )

    grid_Iq[filter_out] = np.nan

    return grid_x, grid_z, grid_Iq, filter_out

