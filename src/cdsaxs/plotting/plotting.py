"""
Plotting functions for cdsaxs data classes.
"""

import warnings

import matplotlib as mpl
import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from scipy.interpolate import griddata
import matplotlib.colors as mcolors

import cdsaxs.plotting._plotting_tools as plotting_tools
import cdsaxs.diffraction as diffraction
from cdsaxs.data.metadata import METADATA_KEYWORDS


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
                & mask)
    mask_nan = (np.isnan(plotting_image)
                & mask)

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
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log_scale\
        else mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = getattr(plt.cm, cmap)
    cmap.set_bad((0, 0, 0, 0))  # all nan's are transparent in plotting image
    im = plt.imshow(plotting_image, cmap=cmap,
                    # alpha=image_alpha,
                    aspect=aspect, zorder=10,
                    norm=norm
                    )

    # plot the mask/inf/nan image
    custom_cmap = mcolors.ListedColormap(custom_colors)
    custom_cmap.set_bad((0, 0, 0, 0))  # points not masked are transparent
    im_masks = plt.imshow(
        mask_image,
        cmap=custom_cmap,
        aspect=aspect,
        vmin=0, vmax=1,
        interpolation=None,
        zorder=1
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
    plt.title(f"{title}")
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
        **kwargs
):
    fig = plt.figure(fig)

    xmin, xmax = limits_axis1
    ymin, ymax = limits_axis0
    xmin -= 0.5
    xmax -= 0.5
    ymin -= 0.5
    ymax -= 0.5
    x = [xmin, xmin, xmax, xmax, xmin]
    y = [ymin, ymax, ymax, ymin, ymin]

    # fig = plt.figure(fig)
    # plt.errorbar(x, y, zorder=zorder, **kwargs)
    fig = plot_errorbar(
        x, y, zorder=zorder, show_legend=show_legend, fig=fig,
        **kwargs, fmt='-')

    if show_legend:
        plt.legend(bbox_to_anchor=(1, -0.2),
                loc='upper right',
                #bbox_transform=fig.axes[-1].transAxes
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

    plt.errorbar(x, y, **kwargs)

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
        plt.title(title)

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
            y=data.Iq_masked,
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
        color_slice='black',
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
        for bi, _, _ in qslice.background_boxes:
            fig_background = plot_errorbar(
                qslice.q,
                bi,
                fig=fig_background,
                show_legend=show_legend,
                label="None",
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
        color_nan=color_nan)

    # add integration box
    fig_image = plot_image_add_roi(
        qslice.limits_axis0,
        qslice.limits_axis1,
        fig=fig_image,
        color=color_integration_box,
        show_legend=show_legend,
        label="Integration",
        **kwargs
    )

    # add any existing background boxes
    if qslice.background_boxes is not None and show_backgrounds:
        for i, (_, limits0, limits1) in enumerate(qslice.background_boxes):
            fig_image = plot_image_add_roi(
                limits0,
                limits1,
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


def plot_QdyQdx_find_peaks(data, integrated_q_slice, peak_coords_array,
                           peak_coords_q, log_scale=True):

    fig = plot2D(data.image, axis0=data.qdy, axis1=data.qdx,
                 axis0_type='qdy', axis1_type='qdx', log_scale=log_scale)

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
        x=x, y=y, mode='lines', line=dict(color='red'), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=peak_coords_array[:, 1], y=peak_coords_array[:, 0], mode='markers',
        marker=dict(color='red'), showlegend=False
    ))

    # integrated 1D data

    fig_slice = go.Figure()

    if integrated_q_slice.q_axis == 'qdy':
        for x in peak_coords_q[:, 0]:
            fig_slice.add_trace(go.Scatter(
                x=[x, x],
                y=[np.nanmin(integrated_q_slice.Iq),
                   np.nanmax(integrated_q_slice.Iq)*1.1],
                mode='lines',
                line={'color': 'red'},
                showlegend=False
            ))
    else:
        for x in peak_coords_q[:, 1]:
            fig_slice.add_trace(go.Scatter(
                x=[x, x],
                y=[np.nanmin(integrated_q_slice.Iq),
                   np.nanmax(integrated_q_slice.Iq)*1.1],
                mode='lines',
                line={'color': 'red'},
                showlegend=False
            ))

    fig_slice.add_trace(go.Scatter(
        x=integrated_q_slice.q,
        y=integrated_q_slice.Iq,
        mode='lines+markers',
        marker={'color': 'darkcyan'},
        error_y=dict(
            type='data',
            array=integrated_q_slice.dIq,
            visible=True,
        ),
        showlegend=False
    ))

    if integrated_q_slice.mode == 'sum':
        y_axis_label = 'Total Intensity'
    elif integrated_q_slice.mode == 'mean':
        y_axis_label = 'Average Intensity'
    else:
        y_axis_label = 'Intensity'

    fig_slice = plotting_tools.plotly_update_axes(
        x_axis=integrated_q_slice.q_axis,
        y_axis=y_axis_label

    )

    fig_slice = plotting_tools.plotly_update_layout(width=500)

    if log_scale:
        fig_slice = plotting_tools.plotly_update_layout(yscale='log')
    else:
        fig_slice = plotting_tools.plotly_update_axes(
            yrange=(0, np.nanmax(integrated_q_slice.Iq)*1.05)
        )

    return fig, fig_slice


def plot_find_beam_center(data, integrated_q_slice, peak_coords_array,
                          peaks_q_array,
                          beam_center,
                          log_scale=True):

    fig, fig_slice = plot_QdyQdx_find_peaks(
        data, integrated_q_slice, peak_coords_array, peaks_q_array, 
        log_scale=log_scale
    )

    fig.add_vline(beam_center[1], line={'color': 'red', 'dash': 'dot'})
    fig.add_hline(beam_center[0], line={'color': 'red', 'dash': 'dot'})

    return fig, fig_slice


def plot_reduced_dataset(
    reduced_dataset,
    log_scale=True,
    interpolated_image=True,
    plot_marker_size=5,
    filters={},
):
    filtered_slices = reduced_dataset.data.copy()
    for key, value in filters.items():
        keep = []
        for data in filtered_slices:
            test = getattr(data, key)
            if np.nanmin(test) >= np.nanmin(value)\
                    and np.nanmax(test) <= np.nanmax(value):
                keep.append(True)
            else:
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    qszs = []
    qsxs = []
    Iqs = []
    wavelengths = []
    sample_phi_degs = []

    for data in filtered_slices:
        qsz = data.qsz
        qsx = data.qsx
        Iq = np.copy(data.Iq)
        mask = data.mask
        Iq[mask] = np.nan
        qszs.extend(list(qsz))
        qsxs.extend(list(qsx))
        Iqs.extend(list(Iq))
        wavelengths.append(data.wavelength_nm)
        sample_phi_degs.append(data.sample_phi_deg)

    qszs = np.array(qszs)
    qsxs = np.array(qsxs)
    Iqs = np.array(Iqs)

    if len(list(set(wavelengths))) > 1:
        warnings.warn(
            "You are using multiple wavelengths in your"
            "reduction, is that correct? For now we are showing data"
            "under the assumption of a single wavelength!")

    if log_scale:
        vmin = np.nanmin(np.log10(Iqs[Iqs > 0]))
        vmax = np.nanmax(np.log10(Iqs[Iqs > 0]))
    else:
        vmin = np.nanmin(0)
        vmax = np.nanmax(Iqs)

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(9)

    if not interpolated_image:
        colors = []
        cmap = mpl.colormaps['viridis']

        for Iq in Iqs:
            if np.isnan(Iq):
                # nan points are transparent
                colors.append((1, 1, 1, 0))
            elif Iq <= 0:
                if log_scale:
                    # negative or 0 intensity on log scale is black
                    colors.append((1, 1, 1, 1))
                else:
                    colors.append(cmap(0))
            elif Iq > 0 and Iq <= vmin:
                # intensity less than vmin are vmin color
                colors.append(cmap(0))
            elif Iq > vmin and Iq <= vmax:
                # apply colormap
                if log_scale:
                    colors.append(cmap((np.log10(Iq)-vmin)/(vmax-vmin)))
                else:
                    colors.append(cmap((Iq-vmin)/(vmax-vmin)))
            else:
                # intensity higher than vmax are vmax color
                colors.append(cmap(1))

        colors = np.array(colors)

        ax.scatter(qsxs,  qszs, s=plot_marker_size, marker='o', color=colors)
        ax.set_xlabel(plotting_tools.generate_formatted_axis_label('qsx'))
        ax.set_ylabel(plotting_tools.generate_formatted_axis_label('qsz'))

    else:
        wavelength_nm = wavelengths[0]
        sample_phi_deg_range = (np.nanmax(sample_phi_degs),
                                np.nanmin(sample_phi_degs))
        grid_x, grid_z, grid_Iq = plotting_tools(
            qsxs, qszs, Iqs, wavelength_nm, sample_phi_deg_range,
            grid_size=1000,
        )
        plt.contour(grid_x, grid_z, np.log10(grid_Iq), 1000, cmap='viridis',
                    vmin=vmin, vmax=vmax)
        ax.set_xlabel(plotting_tools.generate_formatted_axis_label('qsx'))
        ax.set_ylabel(plotting_tools.generate_formatted_axis_label('qsz'))

    if log_scale:
        norm = mpl_colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
    else:
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cm.viridis
    mappable = mpl_cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("Intensity")

    plt.title(reduced_dataset.name)

    plt.close()

    return fig


def plot_reduced_dataset_interactive(
    reduced_dataset,
    log_scale=True,
    interpolated_image=True,
    filters={}
):
    filtered_slices = reduced_dataset.data.copy()
    for key, value in filters.items():
        keep = []
        for data in filtered_slices:
            test = getattr(data, key)
            if np.nanmin(test) >= np.nanmin(value)\
                    and np.nanmax(test) <= np.nanmax(value):
                keep.append(True)
            else:
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    qszs = []
    qsxs = []
    Iqs = []
    wavelengths = []
    sample_phi_degs = []

    for data in filtered_slices:
        qsz = data.qsz
        qsx = data.qsx
        Iq = np.copy(data.Iq)
        mask = data.mask
        Iq[mask] = np.nan
        qszs.extend(list(qsz))
        qsxs.extend(list(qsx))
        Iqs.extend(list(Iq))
        wavelengths.append(data.wavelength_nm)
        sample_phi_degs.append(data.sample_phi_deg)

    qszs = np.array(qszs)
    qsxs = np.array(qsxs)
    Iqs = np.array(Iqs)

    if len(list(set(wavelengths))) > 1:
        warnings.warn(
            "You are using multiple wavelengths in your"
            "reduction, is that correct? For now we are showing data"
            "under the assumption of a single wavelength!")

    if log_scale:
        vmin = np.nanmin(np.log10(Iqs[Iqs > 0]))
        vmax = np.nanmax(np.log10(Iqs[Iqs > 0]))
    else:
        vmin = np.nanmin(0)
        vmax = np.nanmax(Iqs)

    if not interpolated_image:
        colors = []
        cmap = mpl.colormaps['viridis']

        for Iq in Iqs:
            if np.isnan(Iq):
                # nan points are transparent
                colors.append((1, 1, 1, 0))
            elif Iq <= 0:
                if log_scale:
                    # negative or 0 intensity on log scale is black
                    colors.append((1, 1, 1, 1))
                else:
                    colors.append(cmap(0))
            elif Iq > 0 and Iq <= vmin:
                # intensity less than vmin are vmin color
                colors.append(cmap(0))
            elif Iq > vmin and Iq <= vmax:
                # apply colormap
                if log_scale:
                    colors.append(cmap((np.log10(Iq)-vmin)/(vmax-vmin)))
                else:
                    colors.append(cmap((Iq-vmin)/(vmax-vmin)))
            else:
                # intensity higher than vmax are vmax color
                colors.append(cmap(1))

        colors = np.array(colors)

        fig = go.Figure(data=go.Scatter(
            x=qsxs, y=qszs, mode='markers',
            marker=dict(color=colors)
        ))

    else:
        wavelength_nm = wavelengths[0]
        sample_phi_deg_range = (np.nanmax(sample_phi_degs),
                                np.nanmin(sample_phi_degs))
        grid_x, grid_z, grid_Iq = plotting_tools(
            qsxs, qszs, Iqs, wavelength_nm, sample_phi_deg_range,
            grid_size=1000,
        )
        fig = go.Figure(data=go.Contour(
            x=grid_x, y=grid_z, z=grid_Iq
        ))

    fig.update_xaxes(
        title=plotting_tools.generate_formatted_axis_label('qsx'),
        ticks='outside')
    fig.update_yaxes(
        title=plotting_tools.generate_formatted_axis_label('qsz'),
        ticks='outside')

    if log_scale:
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
        width=500
    )

    fig.update_layout(
        coloraxis_colorbar={
            'title': {'text': 'Intensity', 'side': 'right'},
            'ticks': 'outside',
        })

    fig.update_layout({'title': reduced_dataset.name})

    return fig


def plot_integrated_dataset(
        integrated_dataset,
        q_axis='qdx',
        y_axis='sample_phi_deg',
        log_scale=True,
        filter_q=None,
        filters={}):

    filtered_slices = integrated_dataset.qslices.copy()
    for key, value in filters.items():
        keep = []
        for data in filtered_slices:
            if key in METADATA_KEYWORDS:
                test = data.data2d.metadata[key]
            else:
                test = getattr(data, key)
            if np.nanmin(test) >= np.nanmin(value)\
                    and np.nanmax(test) <= np.nanmax(value):
                keep.append(True)
            else:
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    filtered_slices = []
    if filter_q is not None:
        for data in integrated_dataset.qslices:
            test = getattr(data, filter_q)
            if np.nanmin(test) >= filter_range[0]\
                    and np.nanmax(test) <= filter_range[1]:
                filtered_slices.append(data)
    else:
        filtered_slices = integrated_dataset.qslices

    x_vals = []
    y_vals = []
    color_vals = []
    order_vals = []

    for data in filtered_slices:
        x_vals.append(getattr(data, q_axis))
        order_vals.append(data.data2d.metadata[y_axis]
                          if y_axis in data.data2d.metadata.keys()
                          else data.data2d.user_params[y_axis])
        y_vals.append(np.ones_like(x_vals[-1])*order_vals[-1])
        color_vals.append(data.Iq)

    sort_arrays = np.argsort(order_vals)
    x_vals = np.array(x_vals)[sort_arrays]
    y_vals = np.array(y_vals)[sort_arrays]
    color_vals = np.array(color_vals)[sort_arrays]
    order_vals = np.array(order_vals)[sort_arrays]

    if log_scale:
        vmin = np.log10(np.nanmin(color_vals[color_vals > 0]))
        vmax = np.log10(np.nanmax(color_vals))
    else:
        vmin = np.nanmin(color_vals[color_vals > 0])
        vmax = np.nanmax(color_vals)
    cmap = mpl.colormaps['viridis']

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(9)

    for x, y, cs, o in zip(x_vals, y_vals, color_vals, order_vals):
        if log_scale:
            colors = [cmap((np.log10(c)-vmin)/(vmax-vmin)) for c in cs]
        else:
            colors = [cmap((c-vmin)/(vmax-vmin)) for c in cs]

        ax.scatter(x, y, s=marker_size, marker='o', color=colors)

    ax.set_xlabel(plotting_tools.generate_formatted_axis_label(q_axis))
    ax.set_ylabel(plotting_tools.generate_formatted_axis_label(y_axis))

    if log_scale:
        norm = mpl_colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
    else:
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cm.viridis
    mappable = mpl_cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label('Intensity')

    plt.title(integrated_dataset.name)
    plt.close()

    return fig


def plot_integrated_dataset_interactive(
        integrated_dataset,
        q_axis='qdx',
        y_axis='sample_phi_deg',
        log_scale=True,
        filters={},
        marker_size=5):

    filtered_slices = integrated_dataset.qslices.copy()
    for key, value in filters.items():
        keep = []
        for data in filtered_slices:
            if key in METADATA_KEYWORDS:
                test = data.data2d.metadata[key]
            else:
                test = getattr(data, key)
            if np.nanmin(test) >= np.nanmin(value)\
                    and np.nanmax(test) <= np.nanmax(value):
                keep.append(True)
            else:
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    x_vals = []
    y_vals = []
    color_vals = []
    order_vals = []
    masks = []

    for data in filtered_slices:
        x_vals.append(getattr(data, q_axis))
        order_vals.append(data.data2d.metadata[y_axis]
                          if y_axis in data.data2d.metadata.keys()
                          else data.data2d.user_params[y_axis])
        y_vals.append(np.ones_like(x_vals[-1])*order_vals[-1])
        color_vals.append(data.Iq)
        masks.append(data.mask)

    sort_arrays = np.argsort(order_vals)
    x_vals = np.array(x_vals)[sort_arrays]
    y_vals = np.array(y_vals)[sort_arrays]
    masks = np.array(masks)[sort_arrays]
    color_vals = np.array(color_vals)[sort_arrays]
    order_vals = np.array(order_vals)[sort_arrays]

    if log_scale:
        vmin = np.log10(np.nanmin(color_vals[color_vals > 0]))
        vmax = np.log10(np.nanmax(color_vals))
    else:
        vmin = np.nanmin(color_vals[color_vals > 0])
        vmax = np.nanmax(color_vals)

    x_vals = x_vals.reshape(-1)
    y_vals = y_vals.reshape(-1)
    masks = masks.reshape(-1)
    color_vals = color_vals.reshape(-1)
    if log_scale:
        color_vals = np.log10(color_vals)

    color_vals[masks] = np.nan

    fig = go.Figure(data=go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color_vals,
            colorscale='Viridis',
            colorbar=dict(title=dict(text='Intensity', side='right'),
                          ticks='outside'),
            showscale=True,
            cmin=vmin,
            cmax=vmax
        )
    ))

    # fix how the tick labels look on the colorbar for log scale plots
    if log_scale:
        colorbar_ticks = list(np.arange(
            vmin, np.ceil(vmax) if vmax % 1 > 0 else np.ceil(vmax)+1, step=1))
        colorbar_labels = [10**x for x in colorbar_ticks]
        colorbar_labels = [f"{x:.{0}e}" for x in colorbar_labels]
        fig = plotting_tools.plotly_update_colorbar_ticks(
            fig, colorbar_ticks, colorbar_labels, type='scatter'
        )

    fig = plotting_tools.plotly_update_layout(
        width=500, title=integrated_dataset.name
    )
    fig = plotting_tools.plotly_update_axes(x_axis=q_axis, y_axis=y_axis)

    return fig


def plot_reduced_slices(
        reduced_slices,
        q_axis='qsz',
        slice_axis='qsx',
        filters={},
        log_scale=True,
        offset_order=0,
        offset_value=0):

    filtered_slices = reduced_slices.data.copy()
    for key, value in filters.items():
        keep = []
        for data in filtered_slices:
            test = getattr(data, key)
            if np.nanmin(test) >= np.nanmin(value)\
                    and np.nanmax(test) <= np.nanmax(value):
                keep.append(True)
            else:
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    sort_axis = [getattr(data, slice_axis) for data in filtered_slices]
    sort_by_slice_axis = np.argsort(sort_axis)
    filtered_slices = filtered_slices[sort_by_slice_axis]

    fig, ax = plt.subplots()
    for i, data in enumerate(filtered_slices):
        q = np.copy(getattr(data, q_axis))
        Iq = np.copy(data.Iq)
        sort_q = np.argsort(q)
        ax.errorbar(q[sort_q],
                    Iq[sort_q]*10**(i*offset_order) + offset_value*i,
                    label=getattr(data, slice_axis),
                    fmt='o-')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
              title=plotting_tools.generate_formatted_axis_label(slice_axis))

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel("Intensity")
    ax.set_xlabel(plotting_tools.generate_formatted_axis_label(q_axis))

    plt.close()
    return fig


def plot_reduced_slices_interactive(
        reduced_slices,
        q_axis='qsz',
        slice_axis='qsx',
        filters=None,
        log_scale=True,
        offset_order=0,
        offset_value=0):

    filtered_slices = reduced_slices.data.copy()
    for key, value in filters.items():
        keep = []
        for data in filtered_slices:
            test = getattr(data, key)
            if np.nanmin(test) >= np.nanmin(value)\
                    and np.nanmax(test) <= np.nanmax(value):
                keep.append(True)
            else:
                keep.append(False)
        filtered_slices = [x for x, k in zip(filtered_slices, keep) if k]

    sort_axis = [getattr(data, slice_axis) for data in filtered_slices]
    sort_by_slice_axis = np.argsort(sort_axis)
    filtered_slices = filtered_slices[sort_by_slice_axis]

    fig = None
    for i, data in enumerate(filtered_slices):
        q = np.copy(getattr(data, q_axis))
        Iq = np.copy(data.Iq)
        sort_q = np.argsort(q)
        if i == 0:
            fig = plotting_tools.plot1D_interactive(
                q[sort_q],
                Iq[sort_q]*10**(i*offset_order) + offset_value*i,
                axis_x_type=q_axis,
                axis_y_type="Intensity",
                name=getattr(data, slice_axis),
                showlegend=True,
                log_scale=log_scale
            )
        else:
            fig = plotting_tools.plot1D_add_trace_interactive(
                fig,
                q[sort_q],
                Iq[sort_q]*10**(i*offset_order) + offset_value*i,
                showlegend=True,
                name=getattr(data, slice_axis),
            )

    return fig
