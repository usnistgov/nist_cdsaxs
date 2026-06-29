"""
This module contains classes for handling one-dimensional data, I vs. q.

Data1D : General one-dimensional scattering data class for I vs. q.
IntegratedQSlice(Data1D) : Child class of Data1D. Contains one-
    dimensional scattering data extracted from integration across a
    defined area of two-dimensional scattering data. Holds historic
    information about the generation of the integrated slice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..tools import default_mask
from .metadata import ACCEPTED_Q_AXES
from ..plotting import plotting


class Data1D():

    def __init__(self,
                 q: NDArray,
                 Iq: NDArray,
                 q_axis: str,
                 dIq: NDArray = None,
                 mask: NDArray = None,
                 **kwargs):

        """
        Simple one-dimensional spectra of scattering intensity vs. q.

        Parameters
        ----------
        q : scattering vector
        Iq : scattering intensity as a function of q
        q_axis : str
            The axis for the provided scattering vector q from the
            accepted list below. This will be designated as the primary
            axis, but any of the other axes can be provided as keyword
            arguments (see **kwargs section below).
        dIq : uncertainity along I, default is None
            primary_q_axis : set the primary q-axis (listed below) for this
            dataset. This can then be called with the basic 'q' attribute.
            If left as None, the default axis will be set randomly to one
            of the provided keyword arguments with the same length as Iq.
        mask : NDArray
            One-dimensional boolean array of same dimension as Iq that
            are True at values that shoudl be masked out for all
            operations.
            All points that are nan will be masked out by default. It
            will NOT mask out inf or -inf by default; this is different
            behavior than the 2D data classes.

        **kwargs
        --------
        Any of the accepted scattering vectors or their components
        below can be provided as keyword arguments and become attributes
        of this class. They must all be the same length as Iq or a single
        value if applicable to the whole dataset.

        qdy : Scattering vector component along y axis of detector frame.
        qdx : Scattering vector component along x axis of detector frame.
        qdz : Scattering vector component along z axis of detector frame
        qd  : Scattering vector in the detector frame.
        qsy : Scattering vector component along y axis of sample frame.
        qsx : Scattering vector component along x axis of sample frame.
        qsz : Scattering vector component along z axis of sample frame.
        qsr : Scattering vector component derived from
            sqrt(qsx^2 + qsy^2).
        qs  : Scattering vector in the sample frame.
        qby : Scattering vector component along y axis of the lab frame
        qbx : Scattering vector component along x axis of the lab frame
        qbz : Scattering vector comopnent along z axis of the lab frame; in
            the lab frame the beam path is aligned to the z-axis
        qbr : Scattering vector component derived from
            sqrt(qbx^2 + qby^2).
        qb  : Scattering vector in the beam/lab frame; when the detector is
            positioned normal to the incident beam, the lab and detector
            coordinates will align
        """
        if q_axis in kwargs.keys():
            raise ValueError(
                f"You have provided {q_axis} twice, once as the positional"
                " argument q and another as a keyword argument. Please "
                "use only one."
            )
        if q_axis not in ACCEPTED_Q_AXES:
            raise ValueError(
                f"{q_axis} is not an accepted q-axis."
            )
        else:
            for axis in ACCEPTED_Q_AXES:
                setattr(self, axis, None)
            self.q_axis = q_axis

        # check q axes with length of Iq
        self.Iq = np.array(Iq).reshape(-1).astype(float)
        self._raw_Iq = np.copy(Iq)

        self.mask = default_mask(self.Iq)
        if mask is not None:
            self.mask_points(mask)

        q = np.array(q).reshape(-1).astype(float)
        if len(q) != len(self.Iq):
            raise ValueError(
                f"Iq and q need to be the same length. They were provided "
                f"with lengths of {len(q)} for q and {len(self.Iq)}"
                f"for Iq."
            )
        else:
            self.q = q

        for key, q_key in kwargs.items():
            if isinstance(q_key, float) or isinstance(q_key, int):
                q_key = np.array(q_key).reshape(-1)
            if len(q_key) != len(self.Iq) and len(q_key) != 1:
                raise ValueError(
                    f"'{key}' and 'I(q)' need to be of the same length."
                    f"They were provided with lengths {len(q_key)} and {len(Iq)}."
                )
            else:
                setattr(self, key, q_key)

        if dIq is not None:
            dIq = np.array(dIq).reshape(-1).astype(float)
            if len(dIq) != len(self.Iq):
                raise ValueError(
                    f"Iq and dIq need to be the same length. They were provided "
                    f"with lengths of {len(dIq)} for dIq and {len(self.Iq)}"
                    f"for Iq."
                )
        self.dIq = dIq

        self._data_transformations = []

    @property
    def q(self):
        return getattr(self, self.q_axis)

    @q.setter
    def q(self, new_q):
        new_q = np.array(new_q).reshape(-1).astype(float)
        if len(new_q) != len(self.Iq):
            raise ValueError(
                "The length of q should equal that of Iq."
            )
        setattr(self, self.q_axis, new_q)

    @property
    def _masked_Iq(self):
        masked = np.copy(self.Iq)
        masked[self.mask] = np.nan
        return masked

    def linear_interpolation(
            self,
            q_points: NDArray,
            mode='linear',
            q_axis=None):
        """
        Linearly interpolates the one-dimensional dataset and extract
        intensity values at the specified interpolated q-values. Please
        refer to the numpy.interp documentation for in-depth description
        of the interpolation method used.

        The user is asked to carefully consider this operation to
        ensure that artifacts are not introduced into the dataset and
        that the interpolated points and mode are appropriate.

        Parameters
        ----------
        q_points : NDArray
            q-values at which to extract interpolated intensities.
        mode : str
            Interpolation mode. The interpolation performed is linear,
            but this can be performed on either log axis if desired and
            appropriate for the dataset type. This may give unexpected
            results if you are using log scale of the intensity axis
            and there are values of 0 at one or more points.
            Accepted keywords are:
                'log'    : interpolates on log Iq vs log q
                'log_q'  : interpolates on Iq vs. log q
                'log_Iq' : interpolates on log Iq vs.
                'linear' : interpolates on Iq vs. q
            Default value is 'linear'.
        q_axis : str
            Specify the q_axis to use for the interpolation. If not
            specified, the primary q_axis set to q will be used.

        Returns
        -------
        NDArray : Interpolated I(q) values.
        NDArray : Interpolated q values. This may be different than the
            q_points provided were outside the range of q.
        """
        use_q = np.copy(
            getattr(self, q_axis) if q_axis is not None else self.q)

        # sort q and also remove masked points
        sort_arrays = np.argsort(use_q[~self.mask])
        sorted_q = use_q[~self.mask][sort_arrays]
        sorted_Iq = self.Iq[~self.mask][sort_arrays]

        # remove requested q points outside of sorted q range
        q_points = np.array(q_points)
        keep_points = (q_points <= np.nanmax(sorted_q)) &\
                      (q_points >= np.nanmin(sorted_q))
        q_points = q_points[keep_points]
        interpolated_q_pass = np.copy(q_points)

        if mode not in ['linear', 'log_q', 'log_Iq', 'log']:
            raise ValueError(
                f"Interpoaltion mode {mode} is not recognized. Please use"
                "either 'linear', 'log', 'log_q', or 'log_Iq'."
            )

        if mode == 'log_q' or mode == 'log':
            sorted_q = np.log10(sorted_q)
            interpolated_q_pass = np.log10(interpolated_q_pass)
        if mode == 'log_Iq' or mode == 'log':
            sorted_Iq = np.log10(sorted_Iq)

        interpolated_Iq = np.interp(
            x=interpolated_q_pass,
            xp=sorted_q,
            fp=sorted_Iq
        )
        if mode == 'log' or mode == 'log_Iq':
            interpolated_Iq = np.power(10, interpolated_Iq)

        return q_points, interpolated_Iq

    def scale_data(self, value):
        """
        Scale the data by the specified value or array of values that
        match the dimensions of Iq.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.Iq.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")

        self.Iq = self.Iq*value
        self._data_transformations.append(("scale", value))

    def normalize_data(self, value):
        """
        Scale the data by the reciprocal of the specified value or array
        of values that match the dimensions of Iq.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
            value_r = 1/value
        else:
            value_r = np.reciprocal(value)
            if value_r.shape != self.Iq.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")

        self.Iq = self.Iq*value_r
        self._data_transformations.append(("normalize", value))

    def subtract_from_data(self, value):
        """
        Subtract a specified single value or an array of values that
        matches the dimensions of Iq from the Iq data.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.Iq.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")
        self.Iq = self.Iq-value
        self._data_transformations.append(("subtract", value))

    def add_to_data(self, value):
        """
        Add a specified single value or an array of values that
        matches the dimensions of Iq to the Iq data.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.Iq.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")
        self.Iq = self.Iq+value
        self._data_transformations.append(("add", value))

    def reset_data_transformations(self):
        """
        Resets any normailzation, scaling, added or subtracted values
        applied to the Iq data.
        """
        self.Iq = np.copy(self._raw_Iq)
        self._data_transformations = []

    def abs_q(self, q_axis=None, resort=True):
        """
        Apply absolute value to the primary q axis, or another q axis if
        the q_axis argument is specified. If resort is left as True,
        all data will be resorted to order the q axis that was transformed
        to the absolute value.

        Caution, this transformation to the data cannot be reversed.
        """
        if q_axis is None:
            q_axis = self.q_axis
        abs_q = np.abs(getattr(self, q_axis))
        setattr(self, q_axis, abs_q)

        if resort:
            resort_q = np.argsort(abs_q)
            for key in ACCEPTED_Q_AXES:
                q_vals = getattr(self, key)
                if q_vals is not None:
                    setattr(self, key, q_vals[resort_q])
            self.Iq = self.Iq[resort_q]
            if self.dIq is not None:
                self.dIq = self.dIq[resort_q]
            self.mask = self.mask[resort_q]

    def mask_points(self, mask):
        """
        Add points to the data mask. This will not unmask any previously
        masked points in the image.

        Parameters
        ----------
        mask : NDArray
            One-dimensional boolean array of same dimensions as the
            Iq. Points that are True will be masked out for
            all data operations. This will NOT unmask any previously
            masked points.
        """
        mask = mask.reshape(-1)
        if len(mask) != len(self.Iq):
            raise ValueError(
                "Mask does not match shape of Iq."
            )
        self.mask += mask

    def overwrite_mask(self, mask):
        """
        Set a new mask for the data. This will unmask all previously
        masked points and only mask the points provided to this
        function call.

        Parameters
        ----------
        mask : NDArray
            One-dimensional boolean array of same dimensions as the
            Iq. Points that are True will be masked out for
            all data operations. This will unmask any previously
            masked points.
        """
        self.mask = self.mask*False + mask

    def reset_mask(self):
        """
        Reset the mask to only mask out pixels with values of nan.
        """
        self.mask = np.isnan(self.Iq)  # mask out nan

    def plot_data(
            self,
            q_axis=None,
            log_scale_y=False,
            log_scale_x=False,
            show_legend=True,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel="Intensity",
            xticks=None,
            xticks_labels=None,
            yticks=None,
            yticks_labels=None,
            fig=None,
            title=None,
            **kwargs
    ):
        """
        Plot the 1D data using matplotlib.pyplot.errorbar.

        Parameters
        ----------
        q_axis : str
            Set the q component to be used along the x-axis of the plot.
            If left as the default, None, the primary q_axis of data1d
            will be used.
        log_scale_y : bool, optional
            Convert y-axis to a log scale by setting to true.
            Default value is False.
        log_scale_x : bool, optional
            Convert x-axis to a log scale by setting to true.
            Default value is False.
        show_legend : bool, optional
            Show the legend on the plot by setting to True.
            Default value is True.
        xlim : tuple, list, optional
            Set the plotting range along the x-axis, (xmin, xmax).
            Default is None.
        ylim : tuple, list, optional
            Set the plotting range along the y-axis, (ymin, ymax).
            Default is None.
        xlabel : str, optional
            Set the label on the x-axis.
            Default will be a formatted name of the selected q_axis.
        ylabel : str, optional
            Set the label on the y-axis.
            Default is None.
        xticks : list, optional
            Specify the tick locations along the x-axis.
            Default is None.
        xticks_labels : list, optional
            Specify the labels for each tick location specified in xticks.
            Default is None.
        yticks : list, optional
            Specify the tick locations along the y-axis.
            Default is None.
        yticks_labels : list, optional
            Specify the labels for each tick location specified in yticks.
            Default is None.
        fig : matplotlib.figure, optional
            Pass along the matplotlib figure instance if the trace should
            be added to the figure rather than craeting a new one.
            Default is None.
        title : str, optional
            Set the plot title.
            Default is None.

        **kwargs
        --------
        Any additional keyword arguments accepted by the
        matplotlib.pyplot.errorbar() method will be passed through
        to the plotting function. See the matplotlib documentation
        for more information.
        """

        fig = plotting.plot_data1d(
            self,
            q_axis=q_axis,
            log_scale_y=log_scale_y,
            log_scale_x=log_scale_x,
            show_legend=show_legend,
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
            xticks_labels=xticks_labels,
            yticks=yticks,
            yticks_labels=yticks_labels,
            fig=fig,
            title=title,
            **kwargs
        )

        return fig
