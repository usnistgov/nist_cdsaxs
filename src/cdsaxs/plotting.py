"""
Plotting functions for cdsaxs data classes.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.gridspec as gridspec
import numpy as np
from numpy.typing import NDArray
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as ipw
from ipywidgets import widgets, interact, VBox


import cdsaxs._plotting_tools as plotting_tools


def plot2D(image: NDArray, axis0=None, axis1=None,
           axis0_type=None, axis1_type=None, title=None,
           log_scale=True):

    plot_image = np.copy(image)
    if log_scale:
        with np.errstate(divide='ignore', invalid='ignore'):
            plot_image = np.log10(plot_image)
        vmin = np.nanmin(plot_image[plot_image > -np.inf])
        vmax = np.nanmax(plot_image)
        # set all pixels that were 0 counts to one order of magnitude lower on color scale
        # the pixels that were nan will all show as white
        plot_image[np.isneginf(plot_image)] = vmin-1
        plot_image[np.isnan(plot_image)] = None
    else:
        vmin = 0
        vmax = np.nanmax(plot_image)

    fig = px.imshow(plot_image, zmin=vmin, zmax=vmax, color_continuous_scale='viridis', aspect='equal')

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

    if log_scale:
        colorbar_ticks = list(np.arange(vmin, np.ceil(vmax), step=1))
        colorbar_labels = [10**x for x in colorbar_ticks]
        colorbar_labels = [f"{x:.{0}e}" for x in colorbar_labels]
        fig.update_layout(
            coloraxis_colorbar={
                'tickvals': colorbar_ticks,
                'ticktext': colorbar_labels,
            })
    fig.update_layout(
        width=500,
        coloraxis_colorbar={
            'title': {'text': 'Intensity', 'side': 'right'},
            'ticks': 'outside',
        })

    return fig


def plot_QdyQdx_integration(data, integrated_q_slice, log_scale=True):

    fig = plot2D(data.image, axis0=data.qdy, axis1=data.qdx,
                 axis0_type='qdy', axis1_type='qdx', log_scale=True)

    # box limits, lines get drawn in the middle of pixels so offset
    # half open range by 0.5 pixels
    xmin, xmax = integrated_q_slice.limits_axis1
    xmin -= 0.5
    xmax -= 0.5
    ymin, ymax = integrated_q_slice.limits_axis0
    ymin -= 0.5
    ymax -= 0.5
    x = [xmin, xmin, xmax, xmax, xmin]
    y = [ymin, ymax, ymax, ymin, ymin]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', line=dict(color='red')
    ))
    
    # integrated 1D data
    fig_slice = go.Figure(data=go.Scatter(
        x=integrated_q_slice.q,
        y=integrated_q_slice.I,
        mode='lines+markers',
        error_y=dict(
            type='data',
            array=integrated_q_slice.dI,
            visible=True
        )
    ))

    fig_slice.update_xaxes(
        title=plotting_tools.generate_axis_label_units(
            integrated_q_slice.q_axis
        ),
        ticks='outside'
    )

    fig_slice.update_yaxes(
        title='Total Intensity' if integrated_q_slice.mode == 'sum'
        else 'Average Intensity' if integrated_q_slice.mode == 'mean'
        else 'Intensity',
        ticks='outside'
    )

    fig_slice.update_layout(
        width=500,
    )

    if log_scale:
        fig_slice.update_layout(
            yaxis_type="log"
        )
    else:
        fig_slice.update_yaxes(
            {'range': (0, np.nanmax(integrated_q_slice.I)*1.05)}
        )

    return fig, fig_slice

