"""
Helpful plotting tools for cdsaxs.
"""

import numpy as np
from numpy.typing import NDArray
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go

import cdsaxs._plotting_tools as plotting_tools
from scipy.interpolate import griddata

import cdsaxs.diffraction as diffraction



def create_even_q_ticks(q, num=6, includes_zero=True):
    """
    Produces evenly spaced indices and corresponding q values
    for scattering image axes.

    The scattering vector array, q, must be sorted. It doesn't have to
    be increasing or decreasing as long as it's in order.

    If you want to include q=0 as one of the enforced tick marks,
    includes_zero should be set to True.

    TODO: figure out what happens if includes_zero=True and 0 is not in q
    """
    if not includes_zero:
        ticks_index = np.arange(0, len(q))
        ticks_index = np.linspace(ticks_index, num=num)
        ticks_q = q[ticks_index]
        return ticks_index, ticks_q

    else:
        spacing_exp = np.ceil(np.log10((np.nanmax(q)-np.nanmin(q))/(num-1)))
        spacing = 10**spacing_exp

        start = np.ceil(np.nanmin(q)/spacing)*spacing
        stop = np.floor(np.nanmax(q)/spacing)*spacing + spacing

        ticks_q = np.round(np.arange(start, stop, spacing),
                           int(np.abs(min(0, spacing_exp))))

        while len(ticks_q) < (num-1):
            spacing /= 2
            spacing_exp = np.floor(np.log10(spacing))

            start = np.ceil(np.nanmin(q)/spacing)*spacing
            stop = np.floor(np.nanmax(q)/spacing)*spacing
            ticks_q = np.round(np.arange(start, stop+spacing, spacing),
                               int(np.abs(min(0, spacing_exp))))

        # confirm the last point from np.arange is correct with floating point
        # calculations (see numpy documentation)
        if ticks_q[-1] > np.nanmax(q):
            ticks_q = ticks_q[:-1]

        ticks_interp = []
        ticks_index = np.arange(0, len(q))
        for val in ticks_q:
            if q[0] > q[-1]:
                ticks_interp.append(
                    np.interp(val, np.flip(q), np.flip(ticks_index)))
            else:
                ticks_interp.append(np.interp(val, q, ticks_index))

        # make sure that the ticks are suffiently spaced in pixels
        if np.min(np.abs(np.diff(ticks_interp))) < 10:
            ticks_interp = [ticks_interp[0], ticks_interp[-1]]
            ticks_q = [ticks_q[0], ticks_q[-1]]

        return ticks_interp, ticks_q


def generate_axis_label_units(q_axis):
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

    plot_image = np.copy(image)

    if mask is not None:
        plot_image[mask] = np.nan

    if not log_scale:
        vmin = 0 if vmin is None else vmin
        vmax = np.nanmax(plot_image) if vmax is None else vmax
        custom_colorscale = cmap
        overall_min = vmin
    else:
        vmin = np.log10(np.max([np.nanmin(plot_image[plot_image > 0]), 0.1]))\
            if vmin is None else vmin
        vmax = np.log10(np.nanmax(plot_image)) if vmax is None else vmax
        with np.errstate(divide='ignore', invalid='ignore'):
            plot_image_log = np.log10(plot_image)
        plot_image_log[plot_image <= 0] = -10  # will make these points black
        plot_image_log[np.isnan(plot_image)] = None  # will be transparent
        plot_image = plot_image_log

        # modify the colorscale so that points lower than the range
        # show up as black

        viridis_scale = plotly.colors.sample_colorscale(
            cmap, samplepoints=list(np.linspace(0, 1, 101)))
        custom_colorscale = []
        custom_colorscale.append([0, 'black'])

        overall_min = vmin-1e-15

        for val, color in zip(np.linspace(0, 1, 101), viridis_scale):
            scaled_val = vmin + val * (vmax - vmin)
            normalized_val = (scaled_val - overall_min) / (vmax - overall_min)
            custom_colorscale.append([normalized_val, color])

    # plot the image keeping the aspect ratio of equal for square pixels
    fig = px.imshow(plot_image, zmin=overall_min, zmax=vmax,
                    color_continuous_scale=custom_colorscale, aspect=aspect)

    fig.update_yaxes(
        title=plotting_tools.generate_axis_label_units(axis0_type)
        if axis0_type is not None else "",
        ticks='outside'
    )
    if axis0 is not None:
        ticks, labels = plotting_tools.create_even_q_ticks(axis0)
        fig.update_yaxes(tickvals=ticks, ticktext=labels)

    fig.update_xaxes(
        title=plotting_tools.generate_axis_label_units(axis1_type)
        if axis1_type is not None else "",
        ticks='outside'
    )
    if axis1 is not None:
        ticks, labels = plotting_tools.create_even_q_ticks(axis1)
        fig.update_xaxes(tickvals=ticks, ticktext=labels)

    if log_scale and showcolorbar:
        colorbar_ticks = list(np.arange(
            vmin, np.ceil(vmax) if vmax % 1 > 0 else np.ceil(vmax)+1, step=1))
        colorbar_labels = [10**x for x in colorbar_ticks]
        colorbar_labels = [f"{x:.{0}e}" for x in colorbar_labels]
        fig.update_layout(
            coloraxis_colorbar={
                'tickvals': colorbar_ticks,
                'ticktext': colorbar_labels,
            })
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
            title=plotting_tools.generate_axis_label_units(
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
            title=plotting_tools.generate_axis_label_units(
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


def generate_interpolated_reduced_data(
        qsx,
        qsz,
        Iq,
        wavelength_nm,
        sample_phi_deg_range,
        grid_size=1000,
):
    
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

    return grid_x, grid_z, grid_Iq

