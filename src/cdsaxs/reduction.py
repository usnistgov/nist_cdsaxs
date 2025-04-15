import matplotlib.pyplot as plt
import numpy as np

from cdsaxs.data1d import Data1D
from cdsaxs.dataset import Dataset
from cdsaxs_gui_legacy import diffraction
import cdsaxs.plotting as plotting


def create_reduced_QszQsx(dataset: Dataset,
                          integrated_index: int = 0,
                          in_place: bool = True):
    """
    Creates dictionary of qsz vs. qsx one-dimensional data (sample frame)
    built from one-dimensional integrated data of I vs. qdy (detector frame).
    """
    if dataset.integrated_datasets is None:
        raise ValueError(
            "No integrated datasets to work with."
        )
    integrated_dataset = dataset.integrated_datasets[integrated_index]
    reduced_dataset = {}

    for key, data in dataset.datas.items():
        integrated_q_slice = integrated_dataset[key]

        if integrated_q_slice.q_axis == 'qdx':
            qsz, qsx, _, _ = diffraction.qxz_to_qz_qx(
                integrated_q_slice.q,
                np.zeros(shape=integrated_q_slice.q.shape, dtype=np.float64),
                data.metadata['wavelength_nm'],
                data.metadata['sample_phi_deg'],
                samplethetaoffset=0
            )

            reduced_dataset[key] = {'qsz': qsz,
                                    'qsx': qsx,
                                    'Iq': np.copy(integrated_q_slice.Iq)}

        elif integrated_q_slice.q_axis == 'qdy':
            # TODO: implement 2D CDSAXS
            raise ValueError(
                "Currently only integrated I vs. qdx data can be reduced."
            )

        else:
            raise ValueError(
                "Currently only integrated I vs. qdx data can be reduced."
            )

    if in_place:
        if dataset.reduced_datasets is None:
            dataset.reduced_datasets = {}
        dataset.reduced_datasets[integrated_index] = reduced_dataset


def slice_reduced_dataset(
    dataset: Dataset,
    reduced_index=0,
    q_values=[],
    q_widths=0.001,
    q_axis='qsx',
    find_peaks=False,
    peak_params={},
    in_place: bool = True,
    show_plot=True):
    """
    Creates dictionary of qsz vs. qsx one-dimensional data (sample frame)
    built from one-dimensional integrated data of I vs. qdy (detector frame).
    """

    if not isinstance(q_widths, list):
        q_widths = [q_widths]*len(q_values)
    else:
        if len(q_widths) != len(q_values):
            raise ValueError(
                "q_widths must be float or list with same length as q_values."
            )
        
    q_ranges = [
        (val-width, val+width) for val, width in zip(q_values, q_widths)]

    slices = {}
    for i, q_range in enumerate(q_ranges):
        qsz = []
        Iq = []
        
        for data in dataset.reduced_datasets[reduced_index].values():
            selection = np.where((data[q_axis] >= q_range[0]) &
                                 (data[q_axis] < q_range[1]) &
                                 (data["Iq"] >= 0))[0]
            if len(selection)>0:
                qsz.append(np.nanmean(data['qsz'][selection]))
                Iq.append(np.nanmean(data['Iq'][selection]))

        slices[q_values[i]] = Data1D(q=qsz, Iq=Iq, q_axis='qsz')
    
    if show_plot:

        fig = plotting.plot_reduced_dataset(dataset, index=reduced_index,
                                            log_scale=True)
        
        max_qsz = 0
        min_qsz = 0

        for data in dataset.reduced_datasets[reduced_index].values():
            max_qsz = max(max_qsz, np.nanmax(data["qsz"]))
            min_qsz = min(min_qsz, np.nanmin(data["qsz"]))
        
        for q in q_values:
            fig.axes[0].vlines(q, min_qsz, max_qsz, color='red')

    else:
        fig = None

    if in_place:
        if reduced_index not in dataset.reduced_slices.keys():
            dataset.reduced_slices[reduced_index] = {}
        dataset.reduced_slices[reduced_index][q_axis] = slices
        return fig
    else:
        return slices
        

            
    
    