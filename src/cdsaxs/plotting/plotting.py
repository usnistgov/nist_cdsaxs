"""
Plotting functions for cdsaxs data classes.
"""
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np

from . import _plotting_tools as plotting_tools
from ._plotting_kwargs import (
    ERRORBAR_KWARGS,
    SCATTER_KWARGS,
    IMSHOW_KWARGS
)


def plot_image(
        image,
        fig=None,
        mask=None,
        log_scale=True,
        title=None,
        axis0_vals=None,
        axis0_type=None,
        axis1_vals=None,
        axis1_type=None,
        cmap='viridis',
        aspect='equal',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        **kwargs
        ):
    """
    Plot image data using matplotlib.pyplot.imshow().

    Parameters
    ----------
    image : NDArray
        Two dimensional array of intensities; image to be displayed.
    fig : matplotlib.figure
        Pass a figure instance to add to an existing plot rather than
        creating a new one with this function.
    mask : NDArray
        Two dimensional boolean array where pixels set to True are
        masked.
    log_scale : bool
        If set to True, intensity values are plotted on a log scale.
        Default value is True.
    axis0_vals : list, NDArray
        One dimensional list of values that cooresponds to axis 0 of
        the image.
    axis0_type : str
        Defines the y-axis q-component or the label for the y-axis.
    axis1_vals : list, NDArray
        One dimensional list of values that corresponds to axis 1 of
        the image.
    axis1_type : str
        Defines the x-axis q-component or the label for the x-axis.
    cmap : str
        Matplotlib colormap name for the image intensity.
        Default is 'viridis'.
    aspect : str, float
        Define the aspect ratio of the pixels. 
        'equal' : default, ensures that the pixels are square
        'auto' : changes the aspect ratio to fit within the plotting
            area of the figure
        float : manually set the aspct ratio of the pixel height vs width
    vmin : float
        Set the minimum value of the intensity color range.
    vmax : float
        Set the maximum value of the intensity color range.
    color_mask : str
        Set the color of the masked pixels. Default is 'transparent'.
        Other accepted strings are any of the matplotlib color names.
    color_inf : str
        Set the color of the pixels that have a value of inf or -inf.
        Accepted string are any of the matplotlib color names as well as
        'transparent'.
        Default is 'black'.
    color_nan : str
        Set the color of the pixels that have a value of nan.
        Accepted strings are any of the matplotlib color names as well as
        'transparent'.
        Default is 'red'.

    **kwargs
    --------
    Any of the keyword arguments for matplotlib.pyplot.imshow can be
    used. See the matplotlib documentation for more information.
    """

    # determine colorbar range
    # if log_scale make sure that vmin is 0.1 at a minimum
    if vmin is None:
        vmin = np.max(
            [np.nanmin(image[image > 0]), 0.1]
            ) if log_scale else 0
    if vmax is None:
        vmax = np.nanmax(image)

    # set masked points to nan in the image to be plotted
    plotting_image = np.array(image)
    plotting_image[mask] = np.nan

    if mask is None:
        mask = np.ones_like(plotting_image).astype(bool)
    # mask out other pixels that are either infinity or nan and
    # somehow did not get included in the standard mask
    mask_inf = ((np.isinf(plotting_image)
                 | np.isneginf(plotting_image))
                & ~mask)
    mask_nan = (np.isnan(plotting_image)
                & ~mask)

    # additional masked points also set to nan in the image for plotting
    plotting_image[mask_inf] = np.nan
    plotting_image[mask_nan] = np.nan

    # create a second image where masked, inf, and nan pixels
    # correspond to the custom colorbar of color_*
    mask_image = np.ones_like(plotting_image)*np.nan
    mask_image[mask] = 0
    mask_image[mask_inf] = 0.5
    mask_image[mask_nan] = 1

    custom_colors = [
        color_mask if color_mask != 'transparent' else (1, 1, 1, 0),
        color_inf if color_inf != 'transparent' else (1, 1, 1, 0),
        color_nan if color_nan != 'transparent' else (1, 1, 1, 0),
    ]

    # if log_scale any pixels less than or equal to 0 don't fit into
    # the colorbar range so instead we make them the same color as
    # inf pixels and make them nan in the plotting image
    if log_scale:
        mask_less_than_equal_0 = ((plotting_image <= 0) &
                                  ~np.isnan(plotting_image))
        plotting_image[mask_less_than_equal_0] = np.nan
        mask_image[mask_less_than_equal_0] = 0.5

    # plotting data
    fig = plt.figure(fig)

    # plot the data image
    norm = mpl_colors.LogNorm(vmin=vmin, vmax=vmax) if log_scale\
        else mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = getattr(plt.cm, cmap)
    cmap.set_bad((0, 0, 0, 0))  # all nan's are transparent in plotting image
    im = plt.imshow(plotting_image, cmap=cmap,
                    # alpha=image_alpha,
                    aspect=aspect, zorder=1,
                    norm=norm,
                    **{x: y for x, y in kwargs.items() if x in IMSHOW_KWARGS}
                    )

    # plot the mask/inf/nan image
    custom_cmap = mpl_colors.ListedColormap(custom_colors)
    custom_cmap.set_bad((0, 0, 0, 0))  # points not masked are transparent
    im_masks = plt.imshow(
        mask_image,
        cmap=custom_cmap,
        aspect=aspect,
        vmin=0, vmax=1,
        interpolation=None,
        zorder=-1
    )

    # colorbars
    cbar_masks = plt.colorbar(im_masks, pad=0.125)
    cbar_masks.set_ticks([1/6, 0.5, 5/6])
    cbar_masks.set_ticklabels(["mask", "+/- inf", "nan"], rotation=90)
    cbar_masks.ax.tick_params(size=0)

    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("Intensity",
                       rotation=90, ha='right', va='center', labelpad=10)

    # format the overall plot and axes
    max_figure_chars = 30
    if title is not None and len(title) > max_figure_chars:
        i = max_figure_chars
        while i < len(title):
            title = title[:i] + '\n' + title[i:]
            i = i + max_figure_chars + 2
    plt.title(f"{title}", wrap=True)
    # if hasattr(fig.canvas, 'header_visible'):
    #     fig.canvas.header_visible = False

    if axis1_vals is not None:
        ticks, labels = plotting_tools.create_even_axis_ticks(axis1_vals)
        title = plotting_tools.generate_formatted_axis_label(axis1_type)
        plt.xticks(ticks, labels)
        plt.xlabel(title)

    if axis0_vals is not None:
        ticks, labels = plotting_tools.create_even_axis_ticks(axis0_vals)
        title = plotting_tools.generate_formatted_axis_label(axis0_type)
        plt.yticks(ticks, labels)
        plt.ylabel(title)

    return fig


def plot_image_add_roi(
        limits_axis0,
        limits_axis1,
        fig,
        show_legend=True,
        zorder=1000,
        fmt='-',
        **kwargs
):
    """
    Add a region of interest outline onto a scattering image plot.

    Parameters
    ----------
    limits_axis0 : list
        Indices range to specify the ROI along axis 0, [min, max).
    limits_axis1 : list
        Indices range to specify the ROI along axis 1, [min, max).
    fig : matplotlib.figure
        The figure object with an image plot that the ROI should be
        added to.
    show_legend : bool
        If set to True, the legend will be shown.
    zorder : int
        Set the layering of different traces in the figure.
    fmt : str
        The line format for the region outline.
        Default is '-'.
        Use accepted formats for matplotlib.errorbar.
    """
    fig = plt.figure(fig)

    xmin, xmax = limits_axis1
    ymin, ymax = limits_axis0
    # adjust pixel indices so lines are drawn surrounding included pixels
    xmin -= 0.5
    xmax -= 0.5
    ymin -= 0.5
    ymax -= 0.5
    x = [xmin, xmin, xmax, xmax, xmin]
    y = [ymin, ymax, ymax, ymin, ymin]

    fig = plot_errorbar(
        x, y,
        zorder=zorder,
        show_legend=show_legend,
        fig=fig,
        fmt=fmt,
        **kwargs)

    if show_legend:
        plt.legend(
            bbox_to_anchor=(1, -0.2),
            loc='upper right',
            # bbox_transform=fig.axes[-1].transAxes
        )

    plt.tight_layout()

    return fig


def plot_errorbar(
        x,
        y,
        log_scale_y=False,
        log_scale_x=False,
        show_legend=True,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        xticks=None,
        xticks_labels=None,
        yticks=None,
        yticks_labels=None,
        fig=None,
        title=None,
        **kwargs
):
    """
    Plot the 1D data x, y using matplotlib.pyplot.errorbar().

    Parameters
    ----------
    x, y : array-like
        The data points to be plotted.
    log_scale_y : bool, optional
        Convert y-axis to a log scale by setting to true.
        Default value is False.
    log_scale_x : bool, optional
        Convert x-axis to a log scale by setting to true.
        Default value is False.
    show_legend : bool, optional
        Show the legend on the plot by setting to True.
        Default value is True.
    xlim : tuple, list, optional
        Set the plotting range along the x-axis, (xmin, xmax).
        Default is None.
    ylim : tuple, list, optional
        Set the plotting range along the y-axis, (ymin, ymax).
        Default is None.
    xlabel : str, optional
        Set the label on the x-axis.
        Default is None.
    ylabel : str, optional
        Set the label on the y-axis.
        Default is None.
    xticks : list, optional
        Specify the tick locations along the x-axis.
        Default is None.
    xticks_labels : list, optional
        Specify the labels for each tick location specified in xticks.
        Default is None.
    yticks : list, optional
        Specify the tick locations along the y-axis.
        Default is None.
    yticks_labels : list, optional
        Specify the labels for each tick location specified in yticks.
        Default is None.
    fig : matplotlib.figure, optional
        Pass along the matplotlib figure instance if the trace should
        be added to the figure rather than craeting a new one.
        Default is None.
    title : str, optional
        Set the plot title.
        Default is None.

    **kwargs
    --------
    Any additional keyword arguments accepted by
    matplotlib.pyplot.errorbar can be provided.
    """

    fig = plt.figure(fig)

    plt.errorbar(
        x, y,
        **{x: y for x, y in kwargs.items() if x in ERRORBAR_KWARGS})

    if log_scale_y:
        plt.yscale('log')
    if log_scale_x:
        plt.xscale('log')

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    if show_legend:
        plt.legend()

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title, wrap=True)

    if xticks is not None:
        plt.xticks(xticks, labels=xticks_labels)
    if yticks is not None:
        plt.yticks(yticks, labels=yticks_labels)

    plt.tight_layout()

    return fig


def plot_data2d(
        data2d,
        fig=None,
        log_scale=True,
        show_q=True,
        cmap='viridis',
        aspect='equal',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        **kwargs
        ):
    """
    Plot the 2D data in Data2D using matplotlib.pyplot.imshow().

    Parameters
    ----------
    data2d : Data2D
        Instance of Data2D that contains the scattering image for plotting.
    fig : matplotlib.figure
        Pass a figure instance to add to an existing plot rather than
        creating a new one with this function.
    log_scale : bool
        If set to True, intensity values are plotted on a log scale.
        Default value is True.
    show_q : bool
        Shows the q components along x and y axes if available.
        If set to False, the pixel indices will be shown instead.
    cmap : str
        Matplotlib colormap name for the image intensity.
        Default is 'viridis'.
    aspect : str, float
        Define the aspect ratio of the pixels. 
        'equal' : default, ensures that the pixels are square
        'auto' : changes the aspect ratio to fit within the plotting
            area of the figure
        float : manually set the aspct ratio of the pixel height vs width
    vmin : float
        Set the minimum value of the intensity color range.
    vmax : float
        Set the maximum value of the intensity color range.
    color_mask : str
        Set the color of the masked pixels. Default is 'transparent'.
        Other accepted strings are any of the matplotlib color names.
    color_inf : str
        Set the color of the pixels that have a value of inf or -inf.
        Accepted string are any of the matplotlib color names as well as
        'transparent'.
        Default is 'black'.
    color_nan : str
        Set the color of the pixels that have a value of nan.
        Accepted strings are any of the matplotlib color names as well as
        'transparent'.
        Default is 'red'.

    **kwargs
    --------
    Any of the keyword arguments for matplotlib.pyplot.imshow can be
    used. See the matplotlib documentation for more information.
    """

    fig = plot_image(
        data2d.image,
        mask=data2d.mask,
        log_scale=log_scale,
        title=f"Data2D: {data2d.name}",
        axis0_vals=data2d.qby_1d if show_q else None,
        axis0_type='qdy',
        axis1_vals=data2d.qbx_1d if show_q else None,
        axis1_type='qdx',
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        color_mask=color_mask,
        color_inf=color_inf,
        color_nan=color_nan,
        fig=fig,
        **kwargs
    )

    return fig


def plot_data1d(
        data1d,
        q_axis=None,
        log_scale_y=False,
        log_scale_x=False,
        show_legend=True,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel="Intensity",
        xticks=None,
        xticks_labels=None,
        yticks=None,
        yticks_labels=None,
        fig=None,
        title=None,
        **kwargs
):
    """
    Plot the 1D data x, y using matplotlib.pyplot.errorbar().

    Parameters
    ----------
    data1d : Data1D, list[Data1D]
        Instance(s) of Data1D that contains the I vs. q data to be plotted.
    q_axis : str
        Set the q component to be used along the x-axis of the plot.
        If left as the default, None, the primary q_axis of data1d
        will be used.
    log_scale_y : bool, optional
        Convert y-axis to a log scale by setting to true.
        Default value is False.
    log_scale_x : bool, optional
        Convert x-axis to a log scale by setting to true.
        Default value is False.
    show_legend : bool, optional
        Show the legend on the plot by setting to True.
        Default value is True.
    xlim : tuple, list, optional
        Set the plotting range along the x-axis, (xmin, xmax).
        Default is None.
    ylim : tuple, list, optional
        Set the plotting range along the y-axis, (ymin, ymax).
        Default is None.
    xlabel : str, optional
        Set the label on the x-axis.
        Default will be a formatted name of the selected q_axis.
    ylabel : str, optional
        Set the label on the y-axis.
        Default is None.
    xticks : list, optional
        Specify the tick locations along the x-axis.
        Default is None.
    xticks_labels : list, optional
        Specify the labels for each tick location specified in xticks.
        Default is None.
    yticks : list, optional
        Specify the tick locations along the y-axis.
        Default is None.
    yticks_labels : list, optional
        Specify the labels for each tick location specified in yticks.
        Default is None.
    fig : matplotlib.figure, optional
        Pass along the matplotlib figure instance if the trace should
        be added to the figure rather than craeting a new one.
        Default is None.
    title : str, optional
        Set the plot title.
        Default is None.

    **kwargs
    --------
    Any additional keyword arguments accepted by
    matplotlib.pyplot.errorbar can be provided.
    """
    fig = plt.figure(fig)

    if type(data1d) is not list:
        data1d = [data1d]

    for data in data1d:
        if q_axis is None:
            q_axis = data.q_axis

        if xlabel is None:
            xlabel = plotting_tools.generate_formatted_axis_label(q_axis)

        plot_errorbar(
            x=getattr(data, q_axis),
            y=data._masked_Iq,
            fig=fig,
            log_scale_y=log_scale_y,
            log_scale_x=log_scale_x,
            show_legend=show_legend,
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
            xticks_labels=xticks_labels,
            yticks=yticks,
            yticks_labels=yticks_labels,
            title=title,
            **kwargs
        )

    return fig


def plot_qslice(
        qslice,
        log_scale=True,
        cmap='viridis',
        vmin=None,
        vmax=None,
        xlim=None,
        ylim=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        aspect='auto',
        show_backgrounds=True,
        show_legend=True,
        color_slice='darkcyan',
        color_avg_background='grey',
        **kwargs):
    """
    Create the individual plots that show the integration of a scattering
    image into a one-dimensional slice.

    Parameters
    ----------
    qslice : ReducedData1D
        Instance of ReducedData1D that is created when Data2D is
        integrated along one of the axes.
    log_scale : bool
        If set to True, the data will be plotted on a log scale.
    cmap : str
        Colormap for intensity in image plots.
        Any matplotlib colormap name is accepted.
    vmin : float
        Minimum value of the colormap.
    vmax : float
        Maximum value of the colormap.
    color_mask : str
        Set the color of the masked pixels. Default is 'transparent'.
        Other accepted strings are any of the matplotlib color names.
    color_inf : str
        Set the color of the pixels that have a value of inf or -inf.
        Accepted string are any of the matplotlib color names as well as
        'transparent'.
        Default is 'black'.
    color_nan : str
        Set the color of the pixels that have a value of nan.
        Accepted strings are any of the matplotlib color names as well as
        'transparent'.
        Default is 'red'.
    xlim : tuple, list
        Range along the x-axis for the 1D data.
    ylim : tuple, list
        Range along the y-axis for the 1D data.
    show_backgrounds : bool
        If set to True, the background slice information will also
        be shown in the blots.
    show_legend : bool
        If set to True, the legends will be shown on the plots.
    color_integration_box : str
        Set the color of the region of interest outline.
    color_background_box : str
        Set the color of the region of interest outlines for the
        background slices.
    color_slice : str
        Set the color of the trace for the one dimensional slice.
    color_avg_background : str
        Set the color of the background traces. Currently only one
        color is accepted and all backgrounds will be displayed as
        this color.
    """

    # plot the image roi used in the integration
    fig_box = plot_image(
        qslice.image_roi,
        mask=qslice.image_mask,
        log_scale=log_scale,
        title="Integration Box",
        axis0_vals=qslice.data2d.qby_1d[
            qslice.limits_axis0[0]:qslice.limits_axis0[1]],
        axis0_type='qdy',
        axis1_vals=qslice.data2d.qbx_1d[
            qslice.limits_axis1[0]:qslice.limits_axis1[1]],
        axis1_type='qdx',
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        color_mask=color_mask,
        color_inf=color_inf,
        color_nan=color_nan,
        **kwargs
    )

    # plot the 1D slice from the integration with or without background
    fig_slice = plot_data1d(
        qslice,
        log_scale=log_scale,
        label="QSlice I(q)",
        zorder=1000,
        color=color_slice,
        fmt='o-',
        show_legend=show_legend,
        xlim=xlim,
        ylim=ylim,
        **kwargs
    )

    if show_backgrounds and qslice.background_Iq is not None:
        fig_slice = plot_errorbar(
            qslice.q,
            qslice.background_Iq,
            fig=fig_slice,
            label="Avg. Background",
            zorder=100,
            color=color_avg_background,
            show_legend=show_legend,
            fmt='o-',
            **kwargs
        )

    # plot all the individual backgrounds
    if show_backgrounds and qslice.background_Iq is not None:
        fig_background = plot_errorbar(
            qslice.q,
            qslice.background_Iq,
            label="Avg. Background",
            zorder=100,
            log_scale_y=log_scale,
            xlabel=plotting_tools.generate_formatted_axis_label(qslice.q_axis),
            ylabel="Intensity",
            color=color_avg_background,
            show_legend=show_legend,
            fmt='o-',
            xlim=xlim,
            ylim=ylim,
            **kwargs
        )
        for i, b_slice in enumerate(qslice.background_qslices):
            fig_background = plot_errorbar(
                b_slice.q,
                b_slice.Iq,
                fig=fig_background,
                show_legend=show_legend,
                label=f'Background {i}',
                zorder=10,
                fmt='o-',
                **kwargs
            )

    else:
        fig_background = None

    return fig_box, fig_slice, fig_background


def plot_data2d_integrate_box(
        qslice,
        log_scale=True,
        cmap='viridis',
        aspect='equal',
        aspect_box='auto',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        xlim=None,
        ylim=None,
        show_backgrounds=True,
        show_legend=True,
        color_integration_box='red',
        color_background_box='orange',
        color_slice='black',
        color_avg_background='grey',
        **kwargs
):
    """
    Create all the plots that show the integration of a scattering image
    into a one-dimensional slice.

    Parameters
    ----------
    qslice : ReducedData1D
        Instance of ReducedData1D that is created when Data2D is
        integrated along one of the axes.
    log_scale : bool
        If set to True, the data will be plotted on a log scale.
    cmap : str
        Colormap for intensity in image plots.
        Any matplotlib colormap name is accepted.
    aspect : str
        Set the aspect ratio of the pixels in the first imshow figure
        where the entire scattering images is displayed.
        Default is 'auto'. See matplotlib.pyplot.imshow for more
        information.
    aspect_box : str
        Set he aspect ratio of the pixels in the second imshow figure
        that only shows the region of interest from the scattering image.
        Default is 'auto'.
    vmin : float
        Minimum value of the colormap.
    vmax : float
        Maximum value of the colormap.
    color_mask : str
        Set the color of the masked pixels. Default is 'transparent'.
        Other accepted strings are any of the matplotlib color names.
    color_inf : str
        Set the color of the pixels that have a value of inf or -inf.
        Accepted string are any of the matplotlib color names as well as
        'transparent'.
        Default is 'black'.
    color_nan : str
        Set the color of the pixels that have a value of nan.
        Accepted strings are any of the matplotlib color names as well as
        'transparent'.
        Default is 'red'.
    xlim : tuple, list
        Range along the x-axis for the 1D data.
    ylim : tuple, list
        Range along the y-axis for the 1D data.
    show_backgrounds : bool
        If set to True, the background slice information will also
        be shown in the blots.
    show_legend : bool
        If set to True, the legends will be shown on the plots.
    color_integration_box : str
        Set the color of the region of interest outline.
    color_background_box : str
        Set the color of the region of interest outlines for the
        background slices.
    color_slice : str
        Set the color of the trace for the one dimensional slice.
    color_avg_background : str
        Set the color of the background traces. Currently only one
        color is accepted and all backgrounds will be displayed as
        this color.

    **kwargs
    --------
        Accepted keyword arguments to matplotlib.pyplot.errorbar
        function.
    """
    fig_image = plot_data2d(
        qslice.data2d,
        log_scale=log_scale,
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        color_mask=color_mask,
        color_inf=color_inf,
        color_nan=color_nan,)

    # add integration box
    fig_image = plot_image_add_roi(
        qslice.limits_axis0,
        qslice.limits_axis1,
        fig=fig_image,
        color=color_integration_box,
        show_legend=show_legend,
        label="Integration",
        zorder=1000,
        **kwargs
    )

    # add any existing background boxes
    if qslice.background_qslices is not None and show_backgrounds:
        for i, b_slice in enumerate(qslice.background_qslices):
            fig_image = plot_image_add_roi(
                b_slice.limits_axis0,
                b_slice.limits_axis1,
                fig=fig_image,
                color=color_background_box,
                label="Background" if i == 0 else None,
                show_legend=show_legend,
                **kwargs
            )

    fig_box, fig_slice, fig_backgrounds = plot_qslice(
        qslice=qslice,
        log_scale=log_scale,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xlim=xlim,
        ylim=ylim,
        color_mask=color_mask,
        color_inf=color_inf,
        color_nan=color_nan,
        aspect=aspect_box,
        show_backgrounds=show_backgrounds,
        show_legend=show_legend,
        color_slice=color_slice,
        color_avg_background=color_avg_background,
        **kwargs
    )

    return fig_image, fig_box, fig_slice, fig_backgrounds


def plot_data2d_find_peaks2d(
        data2d,
        peaks,
        limits_axis0,
        limits_axis1,
        log_scale=True,
        zoom_plot=True,
        cmap='viridis',
        aspect='equal',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        color_peaks='yellow',
        color_integration_box='yellow',
        **kwargs
):
    """
    Plot of a scattering image overlaid with a region of interest
    outline and points where a peak has been found.

    Parameters
    ----------
    data2d : Data2D
        Instance of Data2D that contains the scattering image for plotting.
    peaks : NDArray
        Two dimensional data of peak positions in y, x of the image.
        These are in units of pixels and so they can be flaot values
        and are not restricted to index integers.
    limits_axis0 : list
        Index range of the ROI along axis 0, [min, max).
    limits_axis1 : list
        Index range of the ROI along axis 1, [min, max).
    log_scale : bool
        If set to True, intensity values are plotted on a log scale.
        Default value is True.
    zoom_plot : bool
        if set to True, the scattering image will be 'zoomed in' on a
        region around the ROI so the peaks can be easily seen.
    cmap : str
        Matplotlib colormap name for the image intensity.
        Default is 'viridis'.
    aspect : str, float
        Define the aspect ratio of the pixels. 
        'equal' : default, ensures that the pixels are square
        'auto' : changes the aspect ratio to fit within the plotting
            area of the figure
        float : manually set the aspct ratio of the pixel height vs width
    vmin : float
        Set the minimum value of the intensity color range.
    vmax : float
        Set the maximum value of the intensity color range.
    color_mask : str
        Set the color of the masked pixels. Default is 'transparent'.
        Other accepted strings are any of the matplotlib color names.
    color_inf : str
        Set the color of the pixels that have a value of inf or -inf.
        Accepted string are any of the matplotlib color names as well as
        'transparent'.
        Default is 'black'.
    color_nan : str
        Set the color of the pixels that have a value of nan.
        Accepted strings are any of the matplotlib color names as well as
        'transparent'.
        Default is 'red'.
    color_peaks : str
        Color of the markers that show where the located peaks are on
        the scattering image.
        Accepted strings are any of the matplotlib color names.
        Default is 'yellow'.
    color_integration_box : str
        Color of the ROI outline that shows the search area for the
        peaks. Accepted strings are any of the matplotlib color names.
        Default value is 'yellow'.
    """

    fig = plot_data2d(
        data2d,
        log_scale=log_scale,
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        color_mask=color_mask,
        color_inf=color_inf,
        color_nan=color_nan,
        fig=None
    )

    fig = plot_image_add_roi(
        limits_axis0=limits_axis0,
        limits_axis1=limits_axis1,
        fig=fig,
        show_legend=False,
        color=color_integration_box,
        zorder=1000
    )

    if peaks.shape[0] > 0:
        fig = plot_errorbar(
            peaks[:, 1],
            peaks[:, 0],
            color=color_peaks,
            fmt='o',
            fig=fig,
            show_legend=False,
            **kwargs,
            zorder=1000
        )

    if zoom_plot:
        plt.xlim(
            max(limits_axis1[0]-20, 0),
            min(limits_axis1[1]+20, data2d.image.shape[1]))
        plt.ylim(
            min(limits_axis0[1]+20, data2d.image.shape[0]),
            max(limits_axis0[0]-20, 0))
    return fig


def plot_data2d_find_detector_rotation_correction(
        data2d,
        peaks,
        line,
        limits_axis0,
        limits_axis1,
        log_scale=True,
        zoom_plot=True,
        cmap='viridis',
        aspect='equal',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        color_peaks='yellow',
        color_integration_box='yellow',
        **kwargs
):
    """
    Plot the peaks located in the region of interest of a scattering
    image overlaid with a line fit to those peaks.

    Parameters
    ----------
    data2d : Data2D
        Instance of Data2D that contains the scattering image for plotting.
    peaks : NDArray
        Two dimensional data of peak positions in y, x of the image.
        These are in units of pixels and so they can be flaot values
        and are not restricted to index integers.
    line : list
        Angle, slope, and intercept of the fit line.
    limits_axis0 : list
        Index range of the ROI along axis 0, [min, max).
    limits_axis1 : list
        Index range of the ROI along axis 1, [min, max).
    log_scale : bool
        If set to True, intensity values are plotted on a log scale.
        Default value is True.
    zoom_plot : bool
        if set to True, the scattering image will be 'zoomed in' on a
        region around the ROI so the peaks can be easily seen.
    cmap : str
        Matplotlib colormap name for the image intensity.
        Default is 'viridis'.
    aspect : str, float
        Define the aspect ratio of the pixels. 
        'equal' : default, ensures that the pixels are square
        'auto' : changes the aspect ratio to fit within the plotting
            area of the figure
        float : manually set the aspct ratio of the pixel height vs width
    vmin : float
        Set the minimum value of the intensity color range.
    vmax : float
        Set the maximum value of the intensity color range.
    color_mask : str
        Set the color of the masked pixels. Default is 'transparent'.
        Other accepted strings are any of the matplotlib color names.
    color_inf : str
        Set the color of the pixels that have a value of inf or -inf.
        Accepted string are any of the matplotlib color names as well as
        'transparent'.
        Default is 'black'.
    color_nan : str
        Set the color of the pixels that have a value of nan.
        Accepted strings are any of the matplotlib color names as well as
        'transparent'.
        Default is 'red'.
    color_peaks : str
        Color of the markers that show where the located peaks are on
        the scattering image.
        Accepted strings are any of the matplotlib color names.
        Default is 'yellow'.
    color_integration_box : str
        Color of the ROI outline that shows the search area for the
        peaks. Accepted strings are any of the matplotlib color names.
        Default value is 'yellow'.
    """

    fig = plot_data2d(
        data2d,
        log_scale=log_scale,
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        color_mask=color_mask,
        color_inf=color_inf,
        color_nan=color_nan,
        fig=None
    )

    fig = plot_image_add_roi(
        limits_axis0=limits_axis0,
        limits_axis1=limits_axis1,
        fig=fig,
        show_legend=False,
        color=color_integration_box,
        zorder=1000
    )

    if peaks.shape[0] > 0:
        fig = plot_errorbar(
            peaks[:, 1],
            peaks[:, 0],
            color=color_peaks,
            fmt='o',
            fig=fig,
            show_legend=False,
            **kwargs,
            zorder=1000
        )

    angle, slope, intercept = line
    if not np.isnan(angle):
        if np.isnan(slope):  # vertical line
            q_range = peaks[:, 1]
            plot_line = peaks[:, 0]
        else:
            # negative x was used for line fit
            qbx_box = -1*data2d.qbx_1d[limits_axis1[0]: limits_axis1[1]]
            plot_line = slope*qbx_box + intercept
            qbx_index = np.arange(min(limits_axis1), max(limits_axis1), 1)
            qby_index = [np.round(
                np.argmin(np.abs(data2d.qby[:, row]-val)), 0).astype(int)
                for row, val in zip(qbx_index, plot_line)]
            plot_line = qby_index
            q_range = qbx_index
            # q_range = np.arange(min(limits_axis1), max(limits_axis1), 1)
            # plot_line = q_range*slope + intercept
        fig = plot_errorbar(
            q_range,
            plot_line,
            color=color_peaks,
            fmt=':',
            fig=fig,
            show_legend=False,
            **kwargs,
            zorder=1000
        )

    if zoom_plot:
        plt.xlim(
            max(limits_axis1[0]-20, 0),
            min(limits_axis1[1]+20, data2d.image.shape[1]))
        plt.ylim(
            min(limits_axis0[1]+20, data2d.image.shape[0]),
            max(limits_axis0[0]-20, 0))
    return fig


def plot_data2d_find_beam_center(
        data2d,
        peaks,
        limits_axis0,
        limits_axis1,
        log_scale=True,
        zoom_plot=True,
        cmap='viridis',
        aspect='equal',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        color_peaks='yellow',
        color_integration_box='yellow',
        color_beam_center='yellow',
        show_beam_center=True,
        **kwargs
):
    """
    Plot the peaks found in a region of interest of a scattering image
    along with the determined beam center position based on those peaks.

    Parameters
    ----------
    data2d : Data2D
        Instance of Data2D that contains the scattering image for plotting.
    peaks : NDArray
        Two dimensional data of peak positions in y, x of the image.
        These are in units of pixels and so they can be flaot values
        and are not restricted to index integers.
    limits_axis0 : list
        Index range of the ROI along axis 0, [min, max).
    limits_axis1 : list
        Index range of the ROI along axis 1, [min, max).
    log_scale : bool
        If set to True, intensity values are plotted on a log scale.
        Default value is True.
    zoom_plot : bool
        if set to True, the scattering image will be 'zoomed in' on a
        region around the ROI so the peaks can be easily seen.
    cmap : str
        Matplotlib colormap name for the image intensity.
        Default is 'viridis'.
    aspect : str, float
        Define the aspect ratio of the pixels. 
        'equal' : default, ensures that the pixels are square
        'auto' : changes the aspect ratio to fit within the plotting
            area of the figure
        float : manually set the aspct ratio of the pixel height vs width
    vmin : float
        Set the minimum value of the intensity color range.
    vmax : float
        Set the maximum value of the intensity color range.
    color_mask : str
        Set the color of the masked pixels. Default is 'transparent'.
        Other accepted strings are any of the matplotlib color names.
    color_inf : str
        Set the color of the pixels that have a value of inf or -inf.
        Accepted string are any of the matplotlib color names as well as
        'transparent'.
        Default is 'black'.
    color_nan : str
        Set the color of the pixels that have a value of nan.
        Accepted strings are any of the matplotlib color names as well as
        'transparent'.
        Default is 'red'.
    color_peaks : str
        Color of the markers that show where the located peaks are on
        the scattering image.
        Accepted strings are any of the matplotlib color names.
        Default is 'yellow'.
    color_integration_box : str
        Color of the ROI outline that shows the search area for the
        peaks. Accepted strings are any of the matplotlib color names.
        Default value is 'yellow'.
    color_beam_center : str
        Color of the beam center crosshairs on the image. Accepted
        strings are any of the matplotlib color names.
        Default value is 'yellow'.
    show_beam_center : bool
        If set to True, the beam center crosshairs are shown on the
        scattering image. Default is True.
    """
    fig = plot_data2d_find_peaks2d(
        data2d,
        peaks=peaks,
        limits_axis0=limits_axis0,
        limits_axis1=limits_axis1,
        log_scale=log_scale,
        zoom_plot=zoom_plot,
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        color_mask=color_mask,
        color_inf=color_inf,
        color_nan=color_nan,
        color_peaks=color_peaks,
        color_integration_box=color_integration_box,
        **kwargs
    )

    if show_beam_center:
        if type(show_beam_center) is not tuple:
            center = data2d.metadata['center_px']
        else:
            center = show_beam_center
        fig = plot_errorbar(
            [center[1], center[1]],
            [0-0.5, data2d.image.shape[0]-0.5],
            show_legend=False,
            fig=fig,
            color=color_beam_center,
            zorder=1000,
        )
        fig = plot_errorbar(
            [0-0.5, data2d.image.shape[1]-0.5],
            [center[0], center[0]],
            show_legend=False,
            fig=fig,
            color=color_beam_center,
            zorder=1000
        )


def plot_data2d_find_sdd(
        data2d,
        peaks,
        limits_axis0,
        limits_axis1,
        sdd_cm,
        log_scale=True,
        zoom_plot=True,
        cmap='viridis',
        aspect='equal',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        color_peaks='yellow',
        color_integration_box='yellow',
        **kwargs
):
    """
    Plot a scattering image and found peaks as part of the find_sdd
    method.

    Parameters
    ----------
    data2d : Data2D
        Instance of Data2D that contains the scattering image for plotting.
    peaks : NDArray
        Two dimensional data of peak positions in y, x of the image.
        These are in units of pixels and so they can be flaot values
        and are not restricted to index integers.
    limits_axis0 : list
        Index range of the ROI along axis 0, [min, max).
    limits_axis1 : list
        Index range of the ROI along axis 1, [min, max).
    log_scale : bool
        If set to True, intensity values are plotted on a log scale.
        Default value is True.
    zoom_plot : bool
        if set to True, the scattering image will be 'zoomed in' on a
        region around the ROI so the peaks can be easily seen.
    cmap : str
        Matplotlib colormap name for the image intensity.
        Default is 'viridis'.
    aspect : str, float
        Define the aspect ratio of the pixels. 
        'equal' : default, ensures that the pixels are square
        'auto' : changes the aspect ratio to fit within the plotting
            area of the figure
        float : manually set the aspct ratio of the pixel height vs width
    vmin : float
        Set the minimum value of the intensity color range.
    vmax : float
        Set the maximum value of the intensity color range.
    color_mask : str
        Set the color of the masked pixels. Default is 'transparent'.
        Other accepted strings are any of the matplotlib color names.
    color_inf : str
        Set the color of the pixels that have a value of inf or -inf.
        Accepted string are any of the matplotlib color names as well as
        'transparent'.
        Default is 'black'.
    color_nan : str
        Set the color of the pixels that have a value of nan.
        Accepted strings are any of the matplotlib color names as well as
        'transparent'.
        Default is 'red'.
    color_peaks : str
        Color of the markers that show where the located peaks are on
        the scattering image.
        Accepted strings are any of the matplotlib color names.
        Default is 'yellow'.
    color_integration_box : str
        Color of the ROI outline that shows the search area for the
        peaks. Accepted strings are any of the matplotlib color names.
        Defaulkt value is 'yellow'.
    """
    if type(sdd_cm) is tuple:
        title = f"Sample to Detector Distance:\n{sdd_cm[0]} cm +/- {sdd_cm[1]} cm"
    else:
        title = f"Sample to Detector Distance\n{sdd_cm[0]} cm"
    fig = plot_data2d_find_peaks2d(
        data2d,
        peaks=peaks,
        limits_axis0=limits_axis0,
        limits_axis1=limits_axis1,
        log_scale=log_scale,
        zoom_plot=zoom_plot,
        cmap=cmap,
        aspect=aspect,
        vmin=vmin,
        vmax=vmax,
        color_mask=color_mask,
        color_inf=color_inf,
        color_nan=color_nan,
        color_peaks=color_peaks,
        color_integration_box=color_integration_box,
        title=title,
        **kwargs
    )

    return fig


def plot_reduced_dataset(
        reduced_dataset,
        log_scale=True,
        cmap='viridis',
        vmin=None,
        vmax=None,
        levels=None,
        filter_by_q={},
        filter_by_metadata={},
        interpolated_data=False,
        grid_size=1000,
        method="cubic",
        distance_factor=5,
        verbose=False,
        **kwargs):
    """
    Plot the reduced dataset as 'qsz' vs 'qsx' using
    matplotlib.pyplot.imshow().

    Parameters
    ----------
    reduced_dataset : ReducedDataset
        Instance of ReducedDataset that contains the data to be plotted.
    log_scale : bool
        If set to True, intensities are plotted on a log scale.
    cmap : str
        Matplotlib colormap name used for the intensity color.
    vmin : float
        Minimum intensity value of the colormap.
    vmax : float
        Maximum intensity value of the colormap.
    filter_by_q : dict
        Key: value pairs of q-component: (min, max) to filter out
        the reduced data.
    filter_by_metadata : dict
        Key: value pairs of metadata key: (min, max) to filter out
        the reduced data.
    interpolated_data : bool
        If set to True, the data will be interpolated onto a grid for
        viewing.

    **kwargs
    --------
    Any keyword arguments for imshow can be passed through this function.
    """

    filtered_slices = reduced_dataset.datas.copy()

    for key, value in filter_by_metadata.items():
        keep = []
        for data in filtered_slices:
            try:
                test = getattr(data, key)
            except:
                keep.append(False)
                continue
            if isinstance(value, tuple):
                if np.nanmin(test) >= np.nanmin(value)\
                        and np.nanmax(test) <= np.nanmax(value):
                    keep.append(True)
                else:
                    keep.append(False)
            elif isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
                if value == test:
                    keep.append(True)
                else:
                    keep.append(False)
            elif isinstance(value, list):
                if test in value:
                    keep.append(True)
                else:
                    keep.append(False)
            else:
                # could not interpret filter
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    for q, qrange in filter_by_q.items():
        keep = []
        for data in filtered_slices:
            test = getattr(data, q)
            if np.nanmin(test) >= qrange[0]\
                    and np.nanmax(test) <= qrange[1]:
                keep.append(True)
            else:
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    q_xaxis = []
    q_yaxis = []
    Iqs = []
    # wavelengths = []
    # sample_phi_degs = []

    for data in filtered_slices:
        q_xaxis.extend(list(getattr(data, 'qsx')))
        q_yaxis.extend(list(getattr(data, 'qsz')))
        Iq = np.copy(data.Iq)
        Iq[data.mask] = np.nan
        Iqs.extend(list(Iq))
        # wavelengths.append(data.wavelength_nm)
        # sample_phi_degs.append(data.sample_phi_deg)

    q_xaxis = np.array(q_xaxis)
    q_yaxis = np.array(q_yaxis)
    Iqs = np.array(Iqs)

    # if len(list(set(wavelengths))) > 1:
    #     warnings.warn(
    #         "You are using multiple wavelengths in your"
    #         "reduction, is that correct? For now we are showing data"
    #         "under the assumption of a single wavelength!")

    if vmin is None:
        vmin = np.max(
            [np.nanmin(Iqs), 0.1]
            ) if log_scale else 0
    if vmax is None:
        vmax = np.nanmax(Iqs)

    fig = plt.figure()
    if log_scale:
        norm = mpl_colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)

    if interpolated_data:
        x_interp, y_interp, Iq_interp =\
            plotting_tools.generate_interpolated_reduced_data(
                qsx=q_xaxis,
                qsz=q_yaxis,
                Iq=Iqs,
                grid_size=grid_size,
                method=method,
                distance_factor=distance_factor,
                verbose=verbose
                # wavelength_nm=wavelengths[0],
                # sample_phi_deg_range=sample_phi_degs,
            )
        if log_scale and levels is None:
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 100)
        elif levels is not None:
            levels = levels
        else:
            levels = np.linspace(vmin, vmax, 100)
        data_plot = plt.contourf(
            x_interp, y_interp, Iq_interp,
            cmap=cmap, norm=norm,
            levels=levels,
            # locator=ticker.LogLocator(base=10, subs='all') if log_scale\
            # else ticker.MaxNLocator(),
            **kwargs
        )
    else:
        data_plot = plt.scatter(
            q_xaxis,
            q_yaxis,
            c=Iqs,
            cmap=cmap,
            norm=norm,
            **{x: y for x, y in kwargs.items() if x in SCATTER_KWARGS})
    cbar_ticks = np.power(10, np.arange(
        np.ceil(np.log10(vmin)), np.floor(np.log10(vmax))+1, 1))
    colorbar = plt.colorbar(data_plot, ticks=cbar_ticks)
    colorbar.set_label('Intensity')

    plt.title(reduced_dataset.name, wrap=True)
    plt.xlabel(plotting_tools.generate_formatted_axis_label('qsx'))
    plt.ylabel(plotting_tools.generate_formatted_axis_label('qsz'))

    plt.tight_layout()

    return fig


def plot_slice_reduced_dataset(
        reduced_dataset,
        q_bins,
        log_scale=True,
        cmap='viridis',
        vmin=None,
        vmax=None,
        interpolated_data=False,
        slice_color='red',
        slice_lw=2,
        **kwargs):
    """
    Plot the reduced dataset overlaid with the slice locations and
    their box width.
    
    Parameters
    ----------
    reduced_dataset : ReducedDataset
        Instance of ReducedDataset that contains the data to be plotted.
    q_bins : list[tuple]
        Each item in the list is a tuple that is (min_q, q, max_q) for
        each slice.
    log_scale : bool
        If set to True, intensities are plotted on a log scale.
    cmap : str
        Matplotlib colormap name used for the intensity color.
    vmin : float
        Minimum intensity value of the colormap.
    vmax : float
        Maximum intensity value of the colormap.
    interpolated_data : bool
        If set to True, the data will be interpolated onto a grid for
        viewing.
    slice_color : str
        Set the color of the slice overlaid on the scattering image.
    slice_lw : float
        Set the linewidth of the slice marker line.
    """

    fig = plot_reduced_dataset(
        reduced_dataset=reduced_dataset,
        log_scale=log_scale,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolated_data=interpolated_data,
        **kwargs
    )

    min_y, max_y = fig.axes[0].get_ylim()
    for (min_q, q, max_q) in q_bins:
        plt.axvspan(xmin=min_q, xmax=max_q,
                    facecolor=slice_color, alpha=0.3, zorder=1000)
        plt.vlines(q, min_y, max_y, color=slice_color,
                   linestyles='dashed', zorder=10000, lw=slice_lw)

    return fig


def plot_reduced_slices(
        reduced_slices,
        q_axis='qsz',
        integrated_axis='qsx',
        filter_by_q={},
        log_scale=True,
        offset_order=0,
        offset_value=0,
        **kwargs):
    """
    Plot the reduced slices using matplotlib.pyplot.errorbar().

    Parameters
    ----------
    reduced_slices : ReducedSlices
        Instance of ReducedSlices that contains the 1d data to be
        plotted.
    q_axis : str
        Primary q-axis to plot along the x-axis.
        Default is 'qsz'.
    integrated_axis : str
        Q-axis that was integrated to generate slices. This will
        provide the legend labels.
        Default value is 'qsx'.
    filter_by_q : dict
        Key: value pairs of q-component: (min, max) to filter which
        slices should be plotted from reduced_slices.
    log_scale : bool
        Plot the data on a log scale in y.
        Default is True.
    offset_order : int, float
        Offset the data by a set number of orders of magnitude.
    offset_value : int, float
        Offset the data by a constant value on a linear scale.
    """

    filtered_slices = reduced_slices.data.copy()
    for key, value in filter_by_q.items():
        keep = []
        for data in filtered_slices:
            test = getattr(data, key)
            if np.nanmin(test) >= np.nanmin(value)\
                    and np.nanmax(test) <= np.nanmax(value):
                keep.append(True)
            else:
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    sort_axis = [getattr(data, integrated_axis) for data in filtered_slices]
    sort_by_slice_axis = np.argsort(sort_axis)
    # filtered_slices = filtered_slices[sort_by_slice_axis]

    fig, ax = plt.subplots()
    for i, sort_i in enumerate(sort_by_slice_axis):
        data = filtered_slices[sort_i]
    # for i, data in enumerate(filtered_slices):
        q = np.copy(getattr(data, q_axis))
        Iq = np.copy(data.Iq)
        sort_q = np.argsort(q)
        ax.errorbar(q[sort_q],
                    Iq[sort_q]*10**(i*-1*offset_order) + offset_value*i,
                    label=getattr(data, integrated_axis),
                    fmt='o-', **kwargs)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
              title=plotting_tools.generate_formatted_axis_label(integrated_axis))

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel("Intensity")
    ax.set_xlabel(plotting_tools.generate_formatted_axis_label(q_axis))

    return fig, ax
