"""
This module includes container classes for datasets, i.e. many of 1D or
2D data instances.
"""

from __future__ import annotations

import warnings

from cdsaxs.data2d import DataQdyQdx
from cdsaxs.sample import Sample


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
    integrated_datasets : list
        TODO: this shouldn't be a list
        List of dictionaries containing integrated data. For each
        ditionary (dataset), the keys align with the datas.keys() and
        the values are instances of IntegratedDataSlices. These
        dictionaries are produced by the cdsaxs integrators.

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
        self.integrated_datasets = None
        self.reduced_datasets = None

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
            figs.append(self.datas[key].plot_data())

        return figs
