"""
This module includes container classes for datasets, i.e. many of 1D or
2D data instances.
"""

from __future__ import annotations

import warnings

import numpy as np

from cdsaxs.data2d import DataQdyQdx
from cdsaxs.sample import Sample
import cdsaxs.plotting as plotting


class Dataset():
    """
    A container class for a set of DataQdyQdx instances that make up a
    single CD-SAXS measurement for a sample.

    Attributes
    ----------
    datas : dict
        Dictionary containing the DataQdyQdx objects. The key for each
        instance is DataQdyQdx.name. Be cautious if the name attribute
        was kept as default (filename) as it may result in non-unique
        keys. If you are pulling all data from the same data directory,
        however, this will not be a problem.
    name : str
        Custom name of the dataset.
    sample : Sample
        Instance of the Sample class that details the sample measured
        when collecting the dataset. Information such as sample
        thickness and attenuation coefficients should be added here to
        enable the relevant data corrections.

    Optional Attributes
    -------------------
    integrated_datasets : dict
        Dictionary of dictionaries containing integrated data. The key
        corresponds to an index (ordered by integration).
        For each inner ditionary, the keys align with the datas.keys() and
        the values are instances of IntegratedDataSlices. These
        dictionaries are produced by the cdsaxs integrators.

    reduced_datastets : dict

    reduced_slices : dict


    """

    def __init__(
            self,
            datas: list[DataQdyQdx] = None,
            name: str = None,
            sample: str = None
    ):
        if datas:
            self.add_data(datas)
        else:
            self.datas = {}

        self.name = name
        self.sample = sample
        self.integrated_datasets = {}
        self.reduced_datasets = {}
        self.reduced_slices = {}

    def add_data(self, datas: DataQdyQdx | list[DataQdyQdx]):
        """Add one or more DataQdyQdx instances to the dataset."""
        datas = [datas] if isinstance(datas, DataQdyQdx) else datas
        for data in datas:
            if data.name in self.datas.keys():
                raise ValueError(
                    "You do not have unique names for DataQdyQdx instances."
                )
            else:
                self.datas[data.name] = data

    def remove_data(self, datas: DataQdyQdx | list[DataQdyQdx]):
        """Remove one or more DataQdyQdx instances from the dataset."""
        datas = [datas] if isinstance(datas, DataQdyQdx) else datas
        for data in datas:
            try:
                del self.datas[data.name]
            except KeyError:
                warnings.warn(f"Could not delete {data.name} data as it was"
                              "not part of the dataset.")

    def assign_sample(self, sample: Sample):
        """
        Assign the measured sample with an instance of the Sample class.
        """
        # TODO: implement required sample checks
        self.sample = sample

    def update_all_metadata(self, metadata: dict, overwrite: bool = True):
        """
        Add or update metadata for all DataQdyQdx stored in this Dataset.
        Existing metadata parameters can be updated by keeping the
        overwrite argument to True.

        Parameters
        ----------
        metadata : dict
            Key : value pairs of accepted metadata (key) and their
            values. See DataQdyQdx class docstring for list of accepted
            keywords.
        overwrite : bool
            If set to True, any metadata provided to this method will
            overwrite the existing value in the instance if it already
            exists in self.metadata.
            Default value is True.
        """

        for data in self.datas.values():
            data.update_metadata(metadata=metadata, overwrite=overwrite)

    def update_user_params_for_all(self, params: dict, overwrite: bool = True):
        """
        Add key: value pairs to the user params for all DataQdyQdx.
        Existing parameters can be updated by keeping the overwrite
        argument as True.

        Parameters
        ----------
        params : dict
            Key : value pairs of user-specified parameters for this
            data instance.
        overwrite : bool
            If set to True, any parameters provided to this method will
            overwrite the existing value in this instance if it already
            exists in self.uer_params.
            Default value is True.
        """

        for data in self.datas.values():
            data.update_user_params(params=params, overwrite=overwrite)

    def plot_datas(self, keys='ALL'):
        """
        Plot one or more DataQdyQdx in the dataset.

        Parameters
        ----------
        keys : list | str
            If set to 'ALL', a list of all figures for all datas will
            be returned. Otherwise, it can be a single data key or
            a list of data keys to plot.
        """

        figs = []

        if keys == 'ALL':
            keys = list(self.datas.keys())
        elif isinstance(keys, str):
            keys = [keys]
        else:
            pass

        for key in keys:
            figs.append(self.datas[key].plot_data(return_fig=True))

        return figs

    def plot_integrated_dataset(
            self,
            index=0,
            q_axis=None,
            order_by='sample_phi_deg',
            log_scale=True):
        """
        Plot the slices extracted from integrated a dataset of DataQdxQdy.

        """
        fig = plotting.plot_integrated_dataset(
            self,
            index=index,
            q_axis=q_axis,
            order_by=order_by,
            log_scale=log_scale,
        )

        return fig

    def plot_reduced_dataset(
            self,
            index=0,
            log_scale=True
    ):
        """
        Plot the Qsz vs. Qsx reduced dataset after integration.
        """
        fig = plotting.plot_reduced_dataset(
            self,
            index=index,
            log_scale=log_scale
        )

        return fig

    def plot_reduced_slices(
            self,
            index=0,
            q_slice_axis='qsx',
            log_scale=True,
            offset_order=0,
            offset_value=0
    ):

        fig = plotting.plot_reduced_slices(
            self,
            index=index,
            q_slice_axis=q_slice_axis,
            log_scale=True,
            offset_order=0,
            offset_value=0,
        )

        return fig

    def save_reduced_slices(self, filename, index=0, q_slice_axis='qsx',
                            decimals=5):
        """
        Returns the slected reduced slices set currently stored in the
        dataset. The user must specify the index of the set of slices
        as well as the q_slice_axis. The number of decimal places the
        slice positions are rounded at can be changed with the
        decimals keyword argument.

        NOTE: currently only a q_slice_axis of 'qsx' is accepted or
        formatted appropriately in the output file.
        TODO: generalize this in the future.
        """

        reduced_slices = self.reduced_slices[index][q_slice_axis]

        length = 0
        for key, val in reduced_slices.items():
            length = np.max((length, val.q.shape[0]))

        datas = []

        for key, val in reduced_slices.items():
            q = val.q
            Iq = val.Iq

            select = Iq > 0

            if len(q[select]) < length:
                new_q = np.hstack(([r'$q_z (\AA^{-1})$'],
                                   q.astype(str)[select],
                                   [""]*(length-len(q[select]))))
            else:
                new_q = np.hstack(([r'$q_z (\AA^{-1})$'],
                                   q.astype(str)[select]))

            if len(q[select]) < length:
                new_Iq = np.hstack(([f'qx = {np.round(key, decimals)}'],
                                    Iq.astype(str)[select],
                                    [""]*(length-len(Iq[select]))))
            else:
                new_Iq = np.hstack(([f'qx = {np.round(key, decimals)}'],
                                    Iq.astype(str)[select]))

            datas.append(new_q)
            datas.append(new_Iq)

        datas = np.array(datas).T
        np.savetxt(filename, datas, delimiter=',', fmt='%s')
