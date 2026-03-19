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
    offset_axis='qsy',
    show_plot=True,
    plotting_kwargs={},
    mode: str = 'mean',
) -> ReducedSlices:
    """
    Extracts 1D data slices from a reduced dataset.

    Parameters
    ----------
    dataset : ReducedDataset
    q_values : list
        List of the slice locations along the specified q_axis.
    q_widths : float | list
        Width of each slice bin. This can be a single value or a
        different value for each value in q_values.
    q_axis : str
        Q axis over which to integrate the data.
        Default is 'qsx'.
    slice_axis : str
        Q axis that will become the primary axis of the one-dimensional
        sliced data.
        Default value is 'qsz'.
    offset_axis : str
        The q-axis that was integrated over to produce the reduced
        data. Default is 'qsy'.
    show_plot : bool
        If True, a plot of the slices overlaid on the reduced data
        image will be shown.
    plotting_kwargs : dict
        Any keyword arguments to pass to the relevant plotting function.
    mode : str
        Set whether a mean or sum is performed on the intensity over
        the bin width. The q components will always be a mean.
        Default is 'mean'.

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
    q_axis = slice_axis         #TODO: FIX arbitrary renaming to match rest of q conventions

    q_ranges = [
        (val-width/2, val+width/2) for val, width in zip(q_values, q_widths)
    ]
    slices = []
    for i, (qmin, qmax) in enumerate(q_ranges):
        q_components = {
            'qsx': [],
            'qsy': [],
            'qsz': [],
            'qsr': [],
            'qs': [],
        }
        # q_offset = []
        Iq = []
        for data in dataset.datas:
            selection = np.where(
                    (getattr(data, integrated_axis) >= qmin) &
                    (getattr(data, integrated_axis) <= qmax)
                )[0]
            if len(selection) > 0\
                    and not data.mask[selection].any()\
                    and not np.isnan(data.Iq[selection]).any():
                for qstr, qlist in q_components.items(): 
                    qlist.append(
                        np.nanmean(getattr(data, qstr)[selection]))
                if mode == 'sum':
                    Iq.append(np.nansum(data.Iq[selection]))
                else:
                    Iq.append(np.nanmean(data.Iq[selection]))

        reduced_slice = ReducedData1DSlice(
            q=np.array(q_components[q_axis]),
            Iq=np.array(Iq),
            q_axis=q_axis,
            integrated_axis=integrated_axis,
            offset_axis=offset_axis,
            slice_width=qmax-qmin
        )
        # TODO: removing rounding to maintain data integrity
        setattr(reduced_slice, integrated_axis,
                np.round(np.mean([qmin, qmax]), 4)) #TODO: removing rounding to maintain data integrity
        for qstr, qlist in q_components.items():
            if qstr != integrated_axis:
                setattr(reduced_slice, qstr, np.array(qlist))
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
