"""
Plotting functions for cdsaxs data classes.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.gridspec as gridspec
import numpy as np
from numpy.typing import NDArray

import cdsaxs._plotting_tools as plotting_tools
from cdsaxs.data2d import DataQdyQdx
from cdsaxs.data1d import IntegratedQSlice


def plot2D(image: NDArray, axis0=None, axis1=None,
           axis0_type=None, axis1_type=None, title=None):
 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # handle pixels with 0 counts (want them to a color)
    # nan pixels will show up as white
    plot_image = np.copy(image)
    vmin = np.nanmin(plot_image[plot_image>0])
    vmax = np.nanmax(plot_image)
    plot_image[plot_image == 0] = vmin/10

    mappable = ax.imshow(plot_image,
                            norm=mpl_colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(mappable, ax=ax, label="Intensity")

    # vertical axis top to bottom
    ax.set_ylabel(
        plotting_tools.generate_axis_label_units(axis0_type)
        if axis0_type is not None
        else ""
    )
    if axis0 is not None:
        ticks, labels = plotting_tools.create_even_q_ticks(axis0)
        ax.set_yticks(ticks, labels)

    # horizontal axis left to right
    ax.set_xlabel(
        plotting_tools.generate_axis_label_units(axis1_type)
        if axis1_type is not None
        else ""
    )
    if axis0 is not None:
        ticks, labels = plotting_tools.create_even_q_ticks(axis1)
        ax.set_xticks(ticks, labels)

    if title:
        plt.title(title)

    # plt.show()
    plt.close()

    return fig


def plot_QdyQdx_integration(data: DataQdyQdx, integrated_q_slice: IntegratedQSlice,
                            slice_log=True):

    fig = plot2D(data.image, axis0=data.qdy, axis1=data.qdx,
                 axis0_type='qdy', axis1_type='qdx')

    gs = gridspec.GridSpec(2, 2)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    fig.axes[0].set_subplotspec(gs[0, 0])
    fig.axes[1].set_subplotspec(gs[0, 1])

    ax = fig.axes[0]
    xmin, xmax = integrated_q_slice.limits_axis1
    ymin, ymax = integrated_q_slice.limits_axis0
    x = [xmin, xmin, xmax, xmax, xmin]
    y = [ymin, ymax, ymax, ymin, ymin]
    ax.plot(x, y, color='red')
    
    ax = fig.axes[1]
    vmin, vmax = ax.viewLim.bounds[2:4]

    ax = fig.add_subplot(gs[1, 0])
    ax.errorbar(integrated_q_slice.q, integrated_q_slice.I, yerr=integrated_q_slice.dI, fmt='-')
    if slice_log:
        ax.set_yscale('log')
        ax.set_ylim(vmin, vmax*10)
    else:
        ax.set_ylim(0, np.nanmax(integrated_q_slice.I)*1.05)

    fig.tight_layout()

    plt.close()

    return fig
