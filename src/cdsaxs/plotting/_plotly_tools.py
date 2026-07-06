import numpy as np
from numpy.typing import NDArray
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go

from . import _plotting_tools as plotting_tools

def plot_scattering_image(
    image: NDArray,
    mask=None,
    axis0_vals=None,
    axis1_vals=None,
    axis0_type=None,
    axis1_type=None,
    title=None,
    log_scale=True,
    vmin=None,
    vmax=None,
    cmap='viridis',
    width=500,
    aspect='equal'
):
    """
    Create a Plotly figure for a 2D scattering image.

    Parameters
    ----------
    image : NDArray
        Two-dimensional image array to display.
    mask : NDArray, optional
        Two-dimensional boolean mask for image, where True marks pixels
        excluded from the plotted intensity map.
    axis0_vals : array-like, optional
        Coordinate values corresponding to axis 0 of image.
    axis1_vals : array-like, optional
        Coordinate values corresponding to axis 1 of image.
    axis0_type : str, optional
        Axis label or q-axis name for the y axis.
    axis1_type : str, optional
        Axis label or q-axis name for the x axis.
    title : str, optional
        Figure title.
    log_scale : bool, optional
        If True, display intensity on a log10 scale.
    vmin : float, optional
        Lower bound of the displayed color range.
    vmax : float, optional
        Upper bound of the displayed color range.
    cmap : str, optional
        Plotly continuous color scale name.
    width : int, optional
        Figure width in pixels.
    aspect : str, optional
        Aspect handling passed to plotly.express.imshow.

    Returns
    -------
    fig : go.Figure
        Plotly figure containing the scattering image.
    """
    #########################
    # determine color range #
    #########################
    if vmin is not None and vmax is None:
        _, vmax = plotting_tools.get_vmin_vmax(
            image, log_scale=log_scale, mask=mask)
    elif vmin is None and vmax is not None:
        vmin, _ = plotting_tools.get_vmin_vmax(
            image, log_scale=log_scale, mask=mask)
    elif vmin is None and vmax is None:
        vmin, vmax = plotting_tools.get_vmin_vmax(
            image, log_scale=log_scale, mask=mask)

    plotting_image = plotting_tools.prepare_data_for_plotting(
        image=image,
        log_scale=log_scale,
        vmin=vmin,
        mask=mask
    )

    #############
    # plot data #
    #############
    fig = px.imshow(
        plotting_image,
        zmin=vmin,
        zmax=vmax,
        color_continuous_scale=cmap,
        aspect=aspect
    )

    ###############
    # format axes #
    ###############
    if axis0_type is not None:
        fig.update_yaxes(
            title=plotting_tools.generate_formatted_axis_label(axis0_type))
    if axis0_vals is not None:
        ticks, labels = plotting_tools.create_even_axis_ticks(axis0_vals)
        fig.update_yaxes(
            tickvals=ticks, ticktext=labels
        )

    if axis1_type is not None:
        fig.update_xaxes(
            title=plotting_tools.generate_formatted_axis_label(axis1_type))
    if axis1_vals is not None:
        ticks, labels = plotting_tools.create_even_axis_ticks(axis1_vals)
        fig.update_xaxes(
            tickvals=ticks, ticktext=labels
        )

    ###################
    # format colorbar #
    ###################
    if log_scale:
        colorbar_ticks = list(np.arange(
            vmin, np.ceil(vmax) if vmax % 1 > 0 else vmax+1, step=1
        ))
        colorbar_labels = [10**x for x in colorbar_ticks]
        colorbar_labels = [f"{x:.{0}e}" for x in colorbar_labels]
        fig.update_layout(
            coloraxis_colorbar={
                'tickvals': colorbar_ticks,
                'ticktext': colorbar_labels
            }
        )
    fig.update_layout(
        coloraxis_colorbar={
            'title': {'text': 'Intensity', 'side': 'right'}
        }
    )

    #################
    # format figure #
    #################
    if title is not None:
        fig.update_layout(title_text=title)
    fig.update_layout(width=width)

    return fig


def add_roi_to_scattering_image(
        fig,
        limits_axis0,
        limits_axis1,
        color='red',
        linestyle='solid',
        label=None,
        showlegend=False,
):
    """
    Add a rectangular region-of-interest outline to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to update.
    limits_axis0 : tuple[int, int]
        Index limits [min, max) for the ROI along axis 0.
    limits_axis1 : tuple[int, int]
        Index limits [min, max) for the ROI along axis 1.
    color : str, optional
        Line color for the ROI outline.
    linestyle : str, optional
        Plotly dash style for the ROI outline.
    label : str, optional
        Legend label for the ROI trace.
    showlegend : bool, optional
        If True, show the ROI trace in the legend.

    Returns
    -------
    fig : go.Figure
        Updated Plotly figure with the ROI outline added.
    """

    (min0, max0) = limits_axis0
    (min1, max1) = limits_axis1
    # subtract -0.5 so the lines show in between pixels rather than on them
    min0 -= 0.5
    max0 -= 0.5
    min1 -= 0.5
    max1 -= 0.5

    x = [min1, min1, max1, max1, min1]
    y = [min0, max0, max0, min0, min0]

    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', line=dict(color=color, dash=linestyle),
        name=label, showlegend=showlegend
    ))

    return fig
