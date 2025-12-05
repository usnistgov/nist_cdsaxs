"""
Reduction tools for processing the cdsaxs Dataset class.
"""
from __future__ import annotations

import numpy as np

from cdsaxs.data.reduced_slice import ReducedData1DSlice
from cdsaxs.data.dataset import (
    ReducedDataset,
    ReducedSlices
)
import cdsaxs.plotting.plotting as plotting


def slice_reduced_dataset(
    dataset: ReducedDataset,
    q_values,
    q_widths=0.001,
    q_axis='qsx',
    slice_axis='qsz',
    show_plot=True,
    plotting_kwargs={}
) -> ReducedSlices:
    """
    Extracts 1D data slices from a reduced dataset.

    Parameters
    ----------
    dataset : ReducedDataset
    q_values : list
    q_widths : float | list
    q_axis : str
    slice_axis : str
    show_plot : bool
    interpolated_image : bool

    Returns
    -------
    ReducedSlices
    """

    if not isinstance(q_widths, list):
        q_widths = [q_widths]*len(q_values)
    if len(q_widths) != len(q_values):
        raise ValueError(
            "The q_widths must be a single value or list with same length "
            "as q_values."
        )

    integrated_axis = q_axis
    q_axis = slice_axis

    q_ranges = [
        (val-width/2, val+width/2) for val, width in zip(q_values, q_widths)
    ]
    slices = []
    for i, (qmin, qmax) in enumerate(q_ranges):
        q = []
        Iq = []
        for data in dataset.datas:
            selection = np.where(
                    (getattr(data, integrated_axis) >= qmin) &
                    (getattr(data, integrated_axis) <= qmax)
                )[0]
            if len(selection) > 0\
                    and not data.mask[selection].any()\
                    and not np.isnan(data.Iq[selection]).any():
                q.append(np.nanmean(getattr(data, q_axis)[selection]))
                Iq.append(np.nanmean(data.Iq[selection]))
                # qsz.extend(list(data.qsz[selection]))
                # Iq.extend(list(data.Iq[selection]))

        reduced_slice = ReducedData1DSlice(
            q=np.array(q),
            Iq=np.array(Iq),
            q_axis=q_axis,
            integrated_axis=integrated_axis,
            slice_width=qmax-qmin
        )
        setattr(reduced_slice, integrated_axis,
                np.round(np.mean([qmin, qmax]), 4))
        slices.append(reduced_slice)

    reduced_slices = ReducedSlices(slices=slices)

    if show_plot:
        fig = plotting.plot_slice_reduced_dataset(
            reduced_dataset=dataset,
            q_bins=[(x, z, y) for (x, y), z in zip(q_ranges, q_values)],
            **plotting_kwargs
        )
    else:
        fig = None

    return reduced_slices, fig
