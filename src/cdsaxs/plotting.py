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

import cdsaxs._plotting_tools as plotting_tools
import cdsaxs.diffraction as diffraction


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


def plot_reduced_dataset(
    reduced_dataset,
    log_scale=True,
    interpolated_image=True,
    plot_marker_size=5,
    filter_q=None,
    filter_range=None,
):
    filtered_slices = []
    if filter_q is not None:
        for data in reduced_dataset.data:
            test = getattr(data, filter_q)
            if np.nanmin(test) >= filter_range[0]\
                    and np.nanmax(test) <= filter_range[1]:
                filtered_slices.append(data)
    else:
        filtered_slices = reduced_dataset.data

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
        ax.set_xlabel(plotting_tools.generate_axis_label_units('qsx'))
        ax.set_ylabel(plotting_tools.generate_axis_label_units('qsz'))

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

    plt.title(reduced_dataset.name)

    plt.close()

    return fig


def plot_reduced_dataset_interactive(
    reduced_dataset,
    log_scale=True,
    interpolated_image=True,
    filter_q=None,
    filter_range=None,
):

    filtered_slices = []
    if filter_q is not None:
        for data in reduced_dataset.data:
            test = getattr(data, filter_q)
            if np.nanmin(test) >= filter_range[0]\
                    and np.nanmax(test) <= filter_range[1]:
                filtered_slices.append(data)
    else:
        filtered_slices = reduced_dataset.data

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
        title=plotting_tools.generate_axis_label_units('qsx'),
        ticks='outside')
    fig.update_yaxes(
        title=plotting_tools.generate_axis_label_units('qsz'),
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
        filter_range=None,
        marker_size=5):

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

    ax.set_xlabel(plotting_tools.generate_axis_label_units(q_axis))
    ax.set_ylabel(plotting_tools.generate_axis_label_units(y_axis))

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
        filter_q=None,
        filter_range=None,
        marker_size=5):

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

    color_vals[masks] = True

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

    fig.update_xaxes(
        title=plotting_tools.generate_axis_label_units(q_axis),
        ticks='outside')
    fig.update_yaxes(
        title=plotting_tools.generate_axis_label_units(y_axis),
        ticks='outside')

    if log_scale:
        colorbar_ticks = list(np.arange(
            vmin, np.ceil(vmax) if vmax % 1 > 0 else np.ceil(vmax)+1, step=1))
        colorbar_labels = [10**x for x in colorbar_ticks]
        colorbar_labels = [f"{x:.{0}e}" for x in colorbar_labels]
        fig.update_traces(
                marker_colorbar_tickvals=colorbar_ticks,
                marker_colorbar_ticktext=colorbar_labels,
                selector=dict(type='scatter')
            )
    fig.update_layout(
        width=500
    )

    fig.update_layout({'title': integrated_dataset.name})

    return fig


def plot_reduced_slices(
        reduced_slices,
        q_axis='qsz',
        slice_axis='qsx',
        filter_q=None,
        filter_range=None,
        log_scale=True,
        offset_order=0,
        offset_value=0):

    filtered_slices = []
    if filter_q is not None:
        for data in reduced_slices.data:
            test = getattr(data, filter_q)
            if np.nanmin(test) >= filter_range[0]\
                    and np.nanmax(test) <= filter_range[1]:
                filtered_slices.append(data)
    else:
        filtered_slices = reduced_slices.data

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
              title=plotting_tools.generate_axis_label_units(slice_axis))

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel("Intensity")
    ax.set_xlabel(plotting_tools.generate_axis_label_units(q_axis))

    plt.close()
    return fig


def plot_reduced_slices_interactive(
        reduced_slices,
        q_axis='qsz',
        slice_axis='qsx',
        filter_q=None,
        filter_range=None,
        log_scale=True,
        offset_order=0,
        offset_value=0):

    filtered_slices = []
    if filter_q is not None:
        for data in reduced_slices.data:
            test = getattr(data, filter_q)
            if np.nanmin(test) >= filter_range[0]\
                    and np.nanmax(test) <= filter_range[1]:
                filtered_slices.append(data)
    else:
        filtered_slices = reduced_slices.data

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
