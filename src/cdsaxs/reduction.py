"""
Reduction tools for processing the cdsaxs Dataset class.
"""
from __future__ import annotations

import numpy as np

from .data.reduced_slice import ReducedData1DSlice
from .data.dataset import (
    ReducedDataset,
    ReducedSlices
)
from .plotting import plotting


def qy_row_center(data, target_qsy_Ainv=0.0, offset_qdy_px=None):
    """Locate a row numerically; positive pixel offsets point upward.

    No peak detection or scientific quality checks are performed. Sample-qsy
    targets use interpolation at the calibrated beam column, in A^-1.
    """
    beam_y, beam_x = data.metadata["center_px_detector"]
    if offset_qdy_px is not None:
        return float(beam_y - offset_qdy_px)
    column = int(np.clip(round(beam_x), 0, data.image.shape[1] - 1))
    axis = data.qsy[:, column]
    valid = np.flatnonzero(np.isfinite(axis))
    order = valid[np.argsort(axis[valid])]
    return float(np.interp(target_qsy_Ainv, axis[order], order))


def integrate_qy_row(dataset, height_px, width_px, target_qsy_Ainv=0.0,
                     offset_qdy_px=None, background_offsets_px=None,
                     mode="sum", row_id="qy0"):
    """Integrate a fixed qy row using native masks and fixed backgrounds.

    Unlike the harness, this helper does not validate containment, background
    contrast, or peak presence. Box dimensions and displacements are pixels.
    Backgrounds may also be a fixed dictionary keyed by dataset frame IDs.
    """
    ranges_y, ranges_x = {}, {}
    for key, data in dataset.datas.items():
        center = qy_row_center(data, target_qsy_Ainv, offset_qdy_px)
        ranges_y[key], ranges_x[key] = data.get_box_dims_size(
            height_px, width_px,
            center=(center, data.metadata["center_px_detector"][1]))
    reduced = dataset.integrate_dataset(
        mode=mode, axis=0, range_qdy_px=ranges_y, range_qdx_px=ranges_x,
        subtract_background_offset=background_offsets_px)
    for profile in reduced.datas:
        profile.row_id = row_id
        for component in ("Iq", "qsx", "qsy", "qsz", "qsr", "qs"):
            profile.mask |= ~np.isfinite(getattr(profile, component))
    return reduced


def slice_reduced_dataset(
    dataset: ReducedDataset,
    q_values,
    q_widths=0.001,
    q_axis='qsx',
    slice_axis='qsz',
    offset_axis='qsy',
    show_plot=True,
    plotting_kwargs={},
    mode: str = 'sum',
) -> ReducedSlices:
    """
    Extracts 1D data slices from a reduced dataset.

    Parameters
    ----------
    dataset : ReducedDataset
        Reduced dataset containing one-dimensional reduced patterns to slice.
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
        Default is 'sum'.

    Returns
    -------
    reduced_slices : ReducedSlices
        Collection of extracted one-dimensional slices.
    fig : matplotlib.figure.Figure | None
        Figure showing the selected slice regions when show_plot is True;
        otherwise None.

    Notes
    -----
    Each returned slice records ``source_indices`` (indices into dataset.datas
    for accepted measurements) and ``selected_pixel_counts`` (one count per
    input profile). These describe the native selection before CSV filtering.
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
        source_indices = []
        selected_pixel_counts = []
        plotting_data = {
            'qsx': [],
            'qsz': [],
            'Iq': [],
        }
        for source_index, data in enumerate(dataset.datas):
            selection = np.where(
                    (getattr(data, integrated_axis) >= qmin) &
                    (getattr(data, integrated_axis) <= qmax)
                )[0]
            selected_pixel_counts.append(len(selection))
            if len(selection) > 0\
                    and not data.mask[selection].any()\
                    and not np.isnan(data.Iq[selection]).any():
                source_indices.append(source_index)
                for qstr, qlist in q_components.items():
                    qlist.append(
                        np.nanmean(getattr(data, qstr)[selection]))
                if mode == 'sum':
                    Iq.append(np.nansum(data.Iq[selection]))
                else:
                    Iq.append(np.nanmean(data.Iq[selection]))
                plotting_data['qsx'].extend(list(data.qsx[selection]))
                plotting_data['qsz'].extend(list(data.qsz[selection]))
                plotting_data['Iq'].extend(list(data.Iq[selection]))

        reduced_slice = ReducedData1DSlice(
            q=np.array(q_components[q_axis]),
            Iq=np.array(Iq),
            q_axis=q_axis,
            integrated_axis=integrated_axis,
            offset_axis=offset_axis,
            slice_width=qmax-qmin,
            plotting_data=plotting_data,
        )
        reduced_slice.source_indices = np.asarray(source_indices, dtype=int)
        reduced_slice.selected_pixel_counts = np.asarray(selected_pixel_counts, dtype=int)
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
