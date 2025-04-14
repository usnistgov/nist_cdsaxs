import numpy as np

from cdsaxs.dataset import Dataset
from cdsaxs_gui_legacy import diffraction


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

