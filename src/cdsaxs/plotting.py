"""
Plotting functions for cdsaxs data classes.
"""

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

import cdsaxs._plotting_tools as plotting_tools
from cdsaxs.tools import rotate_image


def plot2D(image: NDArray, axis0=None, axis1=None,
           axis0_type=None, axis1_type=None, title=None,
           log_scale=True, vmin=None, vmax=None):
    # TODO axis not rendering in vs code notebook - KNOWN ISSUE VSCODE/PLOTLY

    custom_vmin = True if vmin is not None else False
    custom_vmax = True if vmax is not None else False
    plot_image = np.copy(image)
    if log_scale:
        with np.errstate(divide='ignore', invalid='ignore'):
            plot_image = np.log10(plot_image)
        vmin = np.max([np.nanmin(plot_image[plot_image > -np.inf]), -1]) if not\
            custom_vmin else vmin
        vmax = np.nanmax(plot_image) if not custom_vmax else vmax
        # pixels with less than 'vmin' count will show up as black on the plots
        # need to set them as a custom value to filter later
        plot_image[image <= 0] = -10
        plot_image[np.isnan(image)] = None
        plot_image[(plot_image < vmin) & (image > 0)] = vmin

    else:
        vmin = 0 if not custom_vmin else vmin
        vmax = np.nanmax(plot_image) if not custom_vmax else vmax

    if log_scale and not custom_vmin and not custom_vmax:
        viridis_scale = plotly.colors.sample_colorscale(
            'Viridis', samplepoints=list(np.linspace(0, 1, 101)))
        custom_colorscale = []
        custom_colorscale.append([0, 'black'])

        overall_min = vmin-1e-15

        for val, color in zip(np.linspace(0, 1, 101), viridis_scale):
            scaled_val = vmin + val * (vmax - vmin)
            normalized_val = (scaled_val - overall_min) / (vmax - overall_min)
            custom_colorscale.append([normalized_val, color])
    else:
        custom_colorscale = 'viridis'
        overall_min = vmin

    fig = px.imshow(plot_image, zmin=overall_min, zmax=vmax,
                    color_continuous_scale=custom_colorscale, aspect='equal')

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
        colorbar_ticks = list(np.arange(
            vmin, np.ceil(vmax) if vmax%1>0 else np.ceil(vmax)+1, step=1))
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

    if title:
        fig.update_layout({'title': title})

    return fig


def plot_QdyQdx_integration(data, integrated_q_slice, log_scale=True,
                            vmin=None, vmax=None):

    image = np.copy(data.image)
    if integrated_q_slice.box_angle_deg != 0:
        image = rotate_image(
            image,
            degrees=integrated_q_slice.box_angle_deg,
            rotation_center=[integrated_q_slice.rotation_center[1],
                             integrated_q_slice.rotation_center[0]],
            resampling_mode=integrated_q_slice.rotation_sampling_mode,
        )
    fig = plot2D(image, axis0=data.qdy, axis1=data.qdx,
                 axis0_type='qdy', axis1_type='qdx', log_scale=log_scale,
                 vmin=vmin, vmax=vmax)

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
        y=integrated_q_slice.Iq,
        mode='lines+markers',
        error_y=dict(
            type='data',
            array=integrated_q_slice.dIq,
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
            {'range': (0, np.nanmax(integrated_q_slice.Iq)*1.05)}
        )

    return fig, fig_slice


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
            {'range': (0, np.nanmax(integrated_q_slice.Iq)*1.05)}
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


def plot_reduced_dataset(dataset, index=0, log_scale=True):

    reduced_dataset = dataset.reduced_datasets[index]

    qszs = []
    qsxs = []
    Iqs = []

    for data in reduced_dataset.values():
        qsz = data.qsz
        qsx = data.qsx
        Iq = data.Iq
        qszs.extend(list(qsz))
        qsxs.extend(list(qsx))
        Iqs.extend(list(Iq))

    qszs = np.array(qszs)
    qsxs = np.array(qsxs)
    Iqs = np.array(Iqs)

    if log_scale:
        # move vmin to one order of magniutde lower which will indicate
        # pixels with 0 counts
        vmin = np.nanmin(np.log10(Iqs[Iqs > 0]))
        vmax = np.nanmax(np.log10(Iqs[Iqs > 0]))
    else:
        vmin = np.nanmin(0)
        vmax = np.nanmax(Iqs)

    colors = []
    cmap = mpl.colormaps['viridis']

    for Iq in Iqs:
        # negative pixel values are shown as white
        if Iq < 0:
            colors.append((0, 0, 0, 0))
        # zero counts are shown as black on log scale or if the
        # linear color scale does not go down to 0
        elif Iq == 0:
            if not log_scale and vmin == 0:
                colors.append(cmap(0))
            else:
                colors.append((1, 1, 1, 1))
        elif Iq > 0:
            if log_scale:
                colors.append(cmap((np.log10(Iq)-vmin)/(vmax-vmin)))
            else:
                colors.append(cmap((Iq-vmin)/(vmax-vmin)))
        # all other pixels shown as white
        else:
            colors.append((0, 0, 0, 0))

    colors = np.array(colors)

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(9)

    ax.scatter(qsxs, qszs, s=5, marker='o', color=colors)
    ax.set_xlabel(plotting_tools.generate_axis_label_units('qsx'))
    ax.set_ylabel(plotting_tools.generate_axis_label_units('qsz'))

    if log_scale:
        norm = mpl_colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
    else:
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cm.viridis
    mappable = mpl_cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("Intensity")

    plt.title(dataset.name)

    plt.close()

    return fig


def plot_integrated_dataset(
        dataset,
        index=0,
        q_axis=None,
        order_by='sample_phi_deg',
        log_scale=True,
        ):

    if q_axis is None:
        # if the q_axis is not provided, pick the first one from the list
        # this should all be the same but in 2d cdsaxs may not always be
        q_axis = list(dataset.integrated_datasets[0].values())[0].q_axis

    data_names = [name for name, value in dataset.integrated_datasets[0].items()
                  if value.q_axis == q_axis]

    x_vals = []
    y_vals = []
    color_vals = []
    order_vals = []

    integrated_dataset = {key: value for key, value
                          in dataset.integrated_datasets[index].items()
                          if key in data_names}
    for name, data in integrated_dataset.items():
        x_vals.append(data.q)
        order_vals.append(dataset.datas[name].metadata[order_by])
        y_vals.append(np.ones_like(data.q)*order_vals[-1])
        color_vals.append(data.Iq)

    sort = np.argsort(order_vals)
    x_vals = np.array(x_vals)[sort]
    y_vals = np.array(y_vals)[sort]
    color_vals = np.array(color_vals)[sort]
    order_vals = np.array(order_vals)[sort]

    # colorbar range
    if log_scale:
        vmin = np.log10(np.nanmin(color_vals[color_vals > 0]))
        vmax = np.log10(np.nanmax(color_vals))
    else:
        vmin = np.nanmin(color_vals[color_vals > 0])
        vmax = np.nanmax(color_vals)
    cmap = mpl.colormaps['viridis']

    # create the plot

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(9)

    for i in range(0, len(order_vals)):
        xs = x_vals[i]
        ys = y_vals[i]
        cs = color_vals[i]
        os = order_vals[i]

        if log_scale:
            colors = [cmap((np.log10(c)-vmin)/(vmax-vmin)) for c in cs]
        else:
            colors = [cmap((c-vmin)/(vmax-vmin)) for c in cs]

        ax.scatter(xs, ys, s=5, marker='o', color=colors)

    ax.set_xlabel(plotting_tools.generate_axis_label_units(q_axis))
    ax.set_ylabel(plotting_tools.generate_axis_label_units(order_by))

    if log_scale:
        norm = mpl_colors.LogNorm(vmin=10**vmin, vmax=10**vmax)
    else:
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cm.viridis
    mappable = mpl_cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label('Intensity')

    plt.title(dataset.name+f" (Integrated Datasets {index})")

    plt.close()

    return fig


def plot_reduced_slices(dataset, index=0, q_slice_axis='qsx', log_scale=True,
                        offset_order=0, offset_value=0):

    reduced_slices = dataset.reduced_slices[index][q_slice_axis]

    fig, ax = plt.subplots()

    slices = np.sort([x for x in reduced_slices.keys()])
    for i, qsx in enumerate(slices):
        data = reduced_slices[qsx]
        sort_q = np.argsort(data.q)
        ax.errorbar(data.q[sort_q],
                    data.Iq[sort_q]*10**(i*offset_order)+offset_value*i,
                    label=np.round(qsx, 6),
                    fmt='o-')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
              title=plotting_tools.generate_axis_label_units(q_slice_axis))

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel("Intensity")
    ax.set_xlabel(plotting_tools.generate_axis_label_units(data.q_axis))

    plt.close()
    return fig
