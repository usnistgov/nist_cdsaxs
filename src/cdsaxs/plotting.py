"""
Plotting functions for cdsaxs data classes.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import numpy as np
from numpy.typing import NDArray

import cdsaxs._plotting_tools as plotting_tools


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
