"""
Plotting functions for cdsaxs data classes.
"""

import warnings

import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np

from cdsaxs.data.metadata import METADATA_KEYWORDS
import cdsaxs.plotting._plotting_tools as plotting_tools
from cdsaxs.plotting._plotting_kwargs import (
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

    fig = plt.figure(fig)

    plt.errorbar(
        x, y,
        **{x: y for x, y in kwargs.items() if x in ERRORBAR_KWARGS})

    if log_scale_y:
        plt.yscale('log')
    if log_scale_x:
        plt.xscale('log')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

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
        log_scale=True,
        cmap='viridis',
        aspect='equal',
        vmin=None,
        vmax=None,
        color_mask='transparent',
        color_inf='black',
        color_nan='red',
        fig=None,
        **kwargs
        ):

    fig = plot_image(
        data2d.image,
        mask=data2d.mask,
        log_scale=log_scale,
        title=f"Data2D: {data2d.name}",
        axis0_vals=data2d.qdy,
        axis0_type='qdy',
        axis1_vals=data2d.qdx,
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
        log_scale=True,
        show_legend=True,
        xlim=None,
        ylim=None,
        fig=None,
        **kwargs
):
    fig = plt.figure(fig)

    if type(data1d) is not list:
        data1d = [data1d]

    for data in data1d:
        if q_axis is None:
            q_axis = data.q_axis

        plot_errorbar(
            x=getattr(data, q_axis),
            y=data._masked_Iq,
            fig=fig,
            log_scale_y=log_scale,
            show_legend=show_legend,
            xlim=xlim,
            ylim=ylim,
            xlabel=plotting_tools.generate_formatted_axis_label(q_axis),
            ylabel="Intensity",
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

    # plot the image roi used in the integration
    fig_box = plot_image(
        qslice.image_roi,
        mask=qslice.image_mask,
        log_scale=log_scale,
        title="Integration Box",
        axis0_vals=qslice.data2d.qdy[
            qslice.limits_axis0[0]:qslice.limits_axis0[1]],
        axis0_type='qdy',
        axis1_vals=qslice.data2d.qdx[
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
    **kwargs
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
            q_range = np.arange(min(limits_axis1), max(limits_axis1), 1)
            plot_line = q_range*slope + intercept
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


def plot_integrated_dataset(
        integrated_dataset,
        q_axis='qdx',
        y_axis='sample_phi_deg',
        log_scale=True,
        cmap='viridis',
        vmin=None,
        vmax=None,
        filter_by_q={},
        filter_by_metadata={},
        **kwargs):

    filtered_slices = integrated_dataset.qslices.copy()

    for key, value in filter_by_metadata.items():
        keep = []
        for data in filtered_slices:
            if key in METADATA_KEYWORDS:
                test = data.data2d.metadata[key]
            elif key in data.data2d.user_params.keys():
                test = data.data2d.user_params[key]
            else:
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

    x_vals = []
    y_vals = []
    color_vals = []
    order_vals = []

    for data in filtered_slices:
        x_vals.extend(list(getattr(data, q_axis)))
        order_vals.append(data.data2d.metadata[y_axis]
                          if y_axis in data.data2d.metadata.keys()
                          else data.data2d.user_params[y_axis])
        y_vals.extend(list(np.ones_like(getattr(data, q_axis))*order_vals[-1]))
        color_vals.extend(list(data.Iq))

    if vmin is None:
        vmin = np.max(
            [np.nanmin(color_vals), 0.1]
            ) if log_scale else 0
    if vmax is None:
        vmax = np.nanmax(color_vals)

    fig = plt.figure()
    if log_scale:
        norm = mpl_colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)

    data_plot = plt.scatter(
        x_vals, y_vals, c=color_vals,
        cmap=cmap, norm=norm,
        **{x: y for x, y in kwargs.items() if x in SCATTER_KWARGS})
    colorbar = plt.colorbar(data_plot)
    colorbar.set_label('Intensity')

    plt.title(integrated_dataset.name, wrap=True)
    plt.xlabel(plotting_tools.generate_formatted_axis_label(q_axis))
    plt.ylabel(plotting_tools.generate_formatted_axis_label(y_axis))

    return fig


def plot_reduced_dataset(
        reduced_dataset,
        log_scale=True,
        cmap='viridis',
        vmin=None,
        vmax=None,
        filter_by_q={},
        filter_by_metadata={},
        interpolated_data=False,
        **kwargs):

    filtered_slices = reduced_dataset.data.copy()

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
    wavelengths = []
    sample_phi_degs = []

    for data in filtered_slices:
        q_xaxis.extend(list(getattr(data, 'qsx')))
        q_yaxis.extend(list(getattr(data, 'qsz')))
        Iq = np.copy(data.Iq)
        Iq[data.mask] = np.nan
        Iqs.extend(list(Iq))
        wavelengths.append(data.wavelength_nm)
        sample_phi_degs.append(data.sample_phi_deg)

    q_xaxis = np.array(q_xaxis)
    q_yaxis = np.array(q_yaxis)
    Iqs = np.array(Iqs)

    if len(list(set(wavelengths))) > 1:
        warnings.warn(
            "You are using multiple wavelengths in your"
            "reduction, is that correct? For now we are showing data"
            "under the assumption of a single wavelength!")

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
        x_interp, y_interp, Iq_interp, filter_out =\
            plotting_tools.generate_interpolated_reduced_data(
                qsx=q_xaxis,
                qsz=q_yaxis,
                Iq=Iqs,
                wavelength_nm=wavelengths[0],
                sample_phi_deg_range=sample_phi_degs,
            )
        if log_scale:
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 100)
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
        offset_value=0):

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
                    fmt='o-')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
              title=plotting_tools.generate_formatted_axis_label(slice_axis))

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel("Intensity")
    ax.set_xlabel(plotting_tools.generate_formatted_axis_label(q_axis))

    plt.close()
    return fig
