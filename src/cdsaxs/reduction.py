"""
Reduction tools for processing the cdsaxs Dataset class.
"""
from __future__ import annotations

import numpy as np

from cdsaxs.data.reduced_data1d import ReducedData1D, ReducedData1DSlice
from cdsaxs.data.dataset import IntegratedDataset
from cdsaxs.data.dataset import ReducedDataset, ReducedSlices
import cdsaxs.diffraction as diffraction


def reduce_dataset(dataset: IntegratedDataset) -> ReducedDataset:
    """
    Reduce an integrated dataset (set of 1D q-slices from the detector
    data for a phi-scan) and convert from detector coordinate space
    to sample coordinate space 'qsx', 'qsz', and 'Iq'.
    TODO: implement filtering

    Parameters
    ----------
    dataset : IntegratedDataset
        Instance of IntegratedDataset containing instances of QSlice.

    Returns
    -------
    ReducedDataset
    """

    reduced_data = []

    for qslice in dataset.qslices:
        qdx = qslice.qdx
        qdy = qslice.qdy
        Iq = qslice.Iq
        dIq = qslice.dIq
        mask = qslice.mask
        wavelength_nm = qslice.data2d.metadata['wavelength_nm'],
        sample_phi_deg = qslice.data2d.metadata.get('sample_phi_deg', 0)\
            + qslice.data2d.metadata.get('sample_phi_offset_deg', 0)
        sample_chi_deg = qslice.data2d.metadata.get('sample_chi_deg', 0)
        sample_omega_deg = qslice.data2d.metadata.get('sample_chi_deg', 0)

        qsz, qsx, _, _ = diffraction.qxz_to_qz_qx(
            qdx,
            qdy if qdy is not None else np.zeros_like(qdx, dtype=float),
            wavelength_nm,
            sample_phi_deg,
            samplethetaoffset=0  # already accounted for this above!
        )

        data = ReducedData1D(
            qsx,
            Iq,
            q_axis='qsx',
            wavelength_nm=wavelength_nm,
            sample_phi_deg=sample_phi_deg,
            sample_chi_deg=sample_chi_deg,
            sample_omega_deg=sample_omega_deg,
            dIq=dIq,
            mask=mask,
            qsz=qsz,
            qdx=qdx,
            qdy=qdy,
        )

        reduced_data.append(data)

    return ReducedDataset(data=reduced_data)


def slice_reduced_dataset(
    dataset: ReducedDataset,
    q_values,
    q_widths=0.001,
    q_axis='qsx',
    slice_axis='qsz',
    show_plot=True,
    interpolated_image=False,
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
        for data in dataset.data:
            selection = np.where(
                    (getattr(data, integrated_axis) >= qmin) &
                    (getattr(data, integrated_axis) <= qmax)
                )[0]
            if len(selection) > 0\
                    and not data.mask[selection].any()\
                    and not np.isnan(data.Iq[selection]).any():
                q.append(np.nanmean(getattr(data, q_axis)))
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

    return reduced_slices

    # if show_plot:

    #     fig = plotting.plot_reduced_dataset(dataset,
    #                                         index=reduced_index,
    #                                         log_scale=True,
    #                                         interpolated_image=interpolated_image,
    #                                         plot_marker_size=plot_marker_size)

    #     max_qsz = 0
    #     min_qsz = 0

    #     for data in dataset.reduced_datasets[reduced_index].values():
    #         max_qsz = max(max_qsz, np.nanmax(data.qsz))
    #         min_qsz = min(min_qsz, np.nanmin(data.qsz))

    #     for q, q_range in zip(q_values, q_ranges):
    #         fig.axes[0].vlines(q, min_qsz, max_qsz, color='red', linestyles='dashed', zorder=100)
    #         fig.axes[0].axvspan(q_range[0], q_range[1], facecolor='red', alpha=0.3, zorder=100) 
    # else:
    #     fig = None

    # if in_place:
    #     if reduced_index not in dataset.reduced_slices.keys():
    #         dataset.reduced_slices[reduced_index] = {}
    #     dataset.reduced_slices[reduced_index][q_axis] = slices
    #     return fig
    # else:
    #     return slices
