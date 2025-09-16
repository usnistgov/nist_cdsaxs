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

from cdsaxs.metadata import ACCEPTED_Q_AXES


class Data1D():

    def __init__(self,
                 q: NDArray,
                 Iq: NDArray,
                 q_axis: NDArray,
                 dIq: NDArray = None,
                 dq: NDArray = None):

        """
        Simple one-dimensional spectra of scattering intensity vs. q.

        Attributes
        ----------
        q : scattering vector, units Ang^-1
        Iq : scattering intensity as a function of q
        q_axis : Defines q as one of the accepted axes listed below.
        dIq : uncertainity along I, default is None
        dq : uncertainty along q, optional, default is None

        Accepted Axes
        -------------
        qdy : Scattering vector component along y axis of detector frame.
        qdx : Scattering vector component along x axis of detector frame.
        qdz : Scattering vector component along z axis of detector frame
        qd  : Scattering vector in the detector frame.
        qsy : Scattering vector component along y axis of sample frame.
        qsx : Scattering vector component along x axis of sample frame.
        qsz : Scattering vector component along z axis of sample frame.
        qs  : Scattering vector in the sample frame.
        qby : Scattering vector component along y axis of the lab frame
        qbx : Scattering vector component along x axis of the lab frame
        qbz : Scattering vector comopnent along z axis of the lab frame; in
            the lab frame the beam path is aligned to the z-axis 
        qb  : Scattering vector in the beam/lab frame; when the detector is
            positioned normal to the incident beam, the lab and detector
            coordinates will align
        """

        if len(q) != len(Iq):
            raise ValueError(
                f"'q' and 'I(q)' need to be of the same length."
                f"They were provided with lengths {len(q)} and {len(Iq)}."
            )

        self.q = np.array(q).reshape(-1).astype(float)
        self.Iq = np.array(Iq).reshape(-1).astype(float)

        if len(self.q) == 0 or len(self.Iq) == 0:
            raise ValueError(
                "Cannot create instance of Data1D with an empty dataset."
                )

        if q_axis not in ACCEPTED_Q_AXES:
            raise ValueError(
                f"{q_axis} is not an accepted q axis. Please select from: "
                f"{ACCEPTED_Q_AXES}"
            )
        else:
            self.q_axis = q_axis

        if dIq is not None and len(dIq) != len(q):
            raise ValueError(
                "dIq needs to be of the same length as q and Iq."
            )
        self.dIq = np.array(dIq).reshape(-1).astype(float)

        if dq is not None and len(dq) != len(q):
            raise ValueError(
                "dq needs to be of the same length as q and Iq."
            )
        self.dq = np.array(dq).reshape(-1).astype(float)

        self._data_transformations = []

    def linear_interpolation(
            self,
            interpolated_q_points: NDArray,
            mode='log',
            _alternative_q=None):
        """
        Linearly interpolatexs the one-dimensional dataset and extract
        intensity values at the specified interpolated q-values. Please
        refer to the numpy.interp documentation for in-depth description
        of the interpolation method used.

        The user is asked to carefully consider this operation to
        ensure that artifacts are not introduced into the dataset and
        that the interpolated points and mode are appropriate.

        Parameters
        ----------
        interpolated_q_points : NDArray
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
            Default value is 'log'.
        _alternative_q : NDArray
            Alternative q-values used as in the interpolation.
            This is useful for datasets that have mulitple scattering
            vector axes (e.g. reduced data).

        Returns
        -------
        NDArray : Interpolated q values.
        NDArray : Interpolated I(q) values.
        """

        interpolated_q_pass = np.array(interpolated_q_points)

        use_q = np.copy(self.q) if _alternative_q is None else _alternative_q
        sort_arrays = np.argsort(use_q)
        sorted_q = use_q[sort_arrays]
        sorted_Iq = np.copy(self.Iq)[sort_arrays]

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

        return interpolated_q_points, interpolated_Iq

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
        for transform, value in reversed(self._data_transformations):
            if transform == "add":
                self.subtract_from_data(value)
            elif transform == "subtract":
                self.add_to_data(value)
            elif transform == "normalize":
                self.scale_data(value)
            elif transform == "scale":
                self.normalize_data(value)
        self._data_transformations = []


class QSlice(Data1D):

    def __init__(
            self,
            q: NDArray,
            Iq: NDArray,
            q_axis: str,
            name: str,
            limits_axis0: tuple[int, int],
            limits_axis1: tuple[int, int],
            mode: str,
            axis: int,
            image_box: NDArray,
            q_int: float,
            q_int_axis: str,
            dIq: NDArray = None,
            dq: NDArray = None,
            dq_int: float = None
    ):
        """
        Child class of Data1D that includes information about the
        integration performed to create a 1D slice from a defined region of
        interest in a two-dimensional dataset or image.

        Parameters
        ----------
        q : NDArray
            Scattering vector, units Ang^-1
        Iq : NDArray
            Scattering intensity as a function of q
        q_axis : str
            Defines q as one of the accepted axes.
            See Data1D for more details on accepted axes.
        name : str
            Unique name associated to the data image integrated.
            This is the key associated to the 2D data object in the dataset.
        limits_axis0 : tuple
            Indexing limits of the image region of interest
            in the first dimension, [min, max).
        limits_axis1 : tuple
            Indexing limits of the image region of interest
            in the second dimension, [min, max).
        mode : str
            Integration mode of either 'sum' or 'mean'.
        axis : int
            Axis over which integration was performed, 0 or 1.
        image_box : NDArray
            Two dimensional region of interest selected from the original
            image over which the integration was performed.
        q_int : float
            Scattering vector of the axis that the summation or mean was
            performed over. This will be the average value across the
            reduced pixels.
        q_int_axis : str
            Defines q_int as one of the accepted axes.
            See Data1D for more details on accepted axes.
        dIq : NDArray, optional
            Uncertainity along I.
            Default is None
        dq : NDArray, optional
            Uncertainty along q.
            Default is None
        dq_int: float, optional
            Uncertaintly along q_int.
            Default is None.

        """

        # Base class init
        super().__init__(q=q, Iq=Iq, q_axis=q_axis, dIq=dIq, dq=dq)
        self.q_int = q_int
        self.q_int_axis = q_int_axis
        self.dq_int = dq_int

        self.name = name

        if len(limits_axis0) != 2 or len(limits_axis1) != 2:
            raise ValueError(
                "Length of axis limits should be two (min and max).")
        self.limits_axis0 = tuple(limits_axis0)
        self.limits_axis1 = tuple(limits_axis1)

        if mode not in ['sum', 'mean']:
            raise ValueError(f"An integration mode of {mode} is not"
                             "accepted.")
        self.mode = mode

        if axis not in [0, 1]:
            raise ValueError(
                f"Integration axis {axis} not understood; use 0 or 1.")
        self.axis = axis

        # make sure that the length of q matches the non-integrated axis
        if (axis == 0 and q.shape[0] != (limits_axis1[1] - limits_axis1[0])
            ) or (axis == 1 and q.shape[0] != (limits_axis0[1]
                                               - limits_axis0[0])):
            raise ValueError(
                f"Your data has a length of {q.shape[0]} after integrating"
                f"along axis {axis}, but this does not align with"
                f"the limits of {limits_axis1 if axis==0 else limits_axis0} "
                f"along axis {1-axis}."
            )

        self.image_box = image_box

        # make place to store original data if the mirror q method is called
        self._data_before_mirror = None

    def mirror_q(self):
        """
        Mirror the data across q = 0. This is identical to taking the
        absolute value of q for every data point. No additional
        resampling or interpolation is performed.
        """
        # keep history of original q and Iq
        self._data_before_mirror = [
            np.copy(self.q), np.copy(self.Iq),
            np.copy(self.dIq) if self.dIq is not None else None,
            np.copy(self.dq) if self.dq is not None else None
        ]

        self.q = np.abs(self.q)

        # re-sort arrays
        sort_arrays = np.argsort(self.q)
        self.q = self.q[sort_arrays]
        self.Iq = self.Iq[sort_arrays]
        if self.dIq is not None:
            self.dIq = self.dIq[sort_arrays]
        if self.dq is not None:
            self.dq = self.dq[sort_arrays]

    def reset_mirrored_q(self):

        self.q = self._data_before_mirror[0]
        self.Iq = self._data_before_mirror[1]
        self.dIq = self._data_before_mirror[2]
        self.dq = self._data_before_mirror[3]

        self._data_before_mirror = None


class ReducedData(Data1D):

    def __init__(self, Iq,
                 qsx=None, qsy=None, qsz=None, primary_axis='qsz',
                 sample_phi_deg=None,
                 sample_chi_deg=None,
                 sample_omega_deg=None,
                 wavelength_nm=None):
        """
        Reduced 1D scattering data that includes one or more axes in
        the sample frame.

        Attributes
        ----------
        Iq : NDArray
            Scattering intensity as a function of q axes.
        qsx : NDArray, optional
            Scattering vector x component in sample frame.
            Default is None.
        qsy : NDArray, optional
            Scattering vector y component in sample frame.
            Default is None.
        qsz : NDArray, optional
            Scattering vector z component in sample frame.
            Default is None.
        primary_axis : str, optional
            Set the default primary axis as 'qsx', 'qsy', or 'qsz'.
            The default primary axis is 'qsz'.
        sample_phi_deg : float
            Sample rotation angle phi, corrected for any offsets
            during processing of the data in detector space.
            Default is None.
        sample_chi_deg : float
            Sample rotation angle chi, corrected for any offsets
            during processing of the data in detector space.
            Default is None.
        sample_omega_deg : float
            Sample rotation angle omega, corrected for any offsets
            during processing of the data in detector space.
            Default is None.
        wavelength_nm : float
            Source wavelength during calculation of sample frame
            scattering vector from detector frame scattering
            vector. Units of nanometers.
            Default is None.
        """
        if primary_axis == 'qsz':
            q = qsz
        elif primary_axis == 'qsy':
            q = qsy
        elif primary_axis == 'qsx':
            q = qsx
        else:
            raise ValueError(
                f"Did not recognize primary axis {primary_axis}."
            )
        super().__init__(q=q, Iq=Iq, q_axis=primary_axis)

        if qsx is not None:
            self.qsx = np.array(qsx).reshape(-1)
            if len(self.Iq) != len(self.qsx):
                raise ValueError(
                    "Dimensions of Iq and qsx do not match."
                )
        else:
            self.qsx = None

        if qsy is not None:
            self.qsy = np.array(qsy).reshape(-1)
            if len(self.Iq) != len(self.qsy):
                raise ValueError(
                    "Dimensions of Iq and qsy do not match."
                )
        else:
            self.qsy = None

        if qsz is not None:
            self.qsz = np.array(qsz).reshape(-1)
            if len(self.Iq) != len(self.qsz):
                raise ValueError(
                    "Dimensions of Iq and qsz do not match."
                )
        else:
            self.qsz = None

        self.sample_phi_deg = sample_phi_deg
        self.sample_chi_deg = sample_chi_deg
        self.sample_omega_deg = sample_omega_deg
        self.wavelength_nm = wavelength_nm

    def linear_interpolation(self, interpolated_q_points, q_axis, mode='log'):
        """
        Linearly interpolate the one-dimensional dataset and extract
        intensity values at the specified interpolated q-values. Please
        refer to the numpy.interp documentation for in-depth description
        of the interpolation method used.

        The user is asked to carefully consider this operation to
        ensure that artifacts are not introduced into the dataset and
        that the interpolated points and mode are appropriate.

        Parameters
        ----------
        interpolated_q_points : NDArray
            q-values at which to extract interpolated intensities.
        q_axis : str
            Specify which q-axis should be used for the interpolation,
            either 'qsx', 'qsy', or 'qsz'.
        mode : str, optional
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
            Default value is 'log'.

        Returns
        -------
        NDArray : Interpolated q values.
        NDArray : Interpolated I(q) values.
        """

        if q_axis == 'qsx':
            alternative_q = self.qsx
        elif q_axis == 'qsy':
            alternative_q = self.qsy
        elif q_axis == 'qsz':
            alternative_q = self.qsz
        else:
            raise ValueError(
                f"The q_axis {q_axis} is not recognized."
            )
        return super().linear_interpolation(
            interpolated_q_points, mode, _alternative_q=alternative_q)
