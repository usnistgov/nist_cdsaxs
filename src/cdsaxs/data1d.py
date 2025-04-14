"""
This module contains 1-dimensional data classes of:

Data1D : General one-dimensional scattering data class for I vs. q.
IntegratedQSlice(Data1D) : Child class of Data1D. Contains one-
    dimensional scattering data extracted from integration across a
    defined area of two-dimensional scattering data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ACCEPTED_Q_AXES = [
    "qdy", "qdx", "qd", "qsy", "qsx", "qsz", "qs"
]


class Data1D():
    """
    Simple one-dimensional spectra of scattering intensity vs. q.

    Attributes
    ----------
    q : scattering vector
    Iq : scattering intensity as a function of q
    q_axis : Defines q as one of the accepted axes listed below.
    dIq : uncertainity along I, default is None
    dq : uncertainty along q, optional, default is None

    Accepted Axes
    -------------
    qdy : Scattering vector component along y axis of detector frame.
    qdx : Scattering vector component along x axis of detector frame.
    qd  : Scattering vector in the detector frame.
    qsy : Scattering vector component along y axis of sample frame.
    qsx : Scattering vector component along x axis of sample frame.
    qsz : Scattering vector component along z axis of sample frame.
    qs  : Scattering vector in the sample frame.
    """

    def __init__(self,
                 q: NDArray,
                 Iq: NDArray,
                 q_axis: NDArray,
                 dIq: NDArray = None,
                 dq: NDArray = None):

        if len(q) != len(Iq):
            raise ValueError(
                f"'q' and 'I(q)' need to be of the same length."
                f"They were provided with lengths {len(q)} and {len(Iq)}."
            )
        self.q = np.array(q)
        self.Iq = np.array(Iq)
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
        self.dIq = np.array(dIq)

        if dq is not None and len(dq) != len(q):
            raise ValueError(
                "dq needs to be of the same length as q and Iq."
            )
        self.dq = np.array(dq)

    def interpolate(self,
                    interpolated_q,
                    mode='log'):
        """
        Interpolate the one-dimensional dataset and extract intensity
        values at the specified interpolated q-values. Please refer
        to the numpy.interp documentation for in-depth description
        of the interpolation method used.

        Parameters
        ----------
        interpolated_q : NDArray
            q-values at which to extract interpolated intensities.
        mode : str
            Interpolation mode. The interpolation performed is linear,
            but this can be performed on the log-scale data, which is
            many scattering cases can limit introduction of artifacts
            in regions of sparse data. To select regular linear
            interpolation, set mode to 'linear'. To select interpolation
            on the log-scale data (log I vs. log q), set mode to 'log'.
            Default value is 'log'.

        Returns
        -------
        NDArray : Interpolated q values.
        NDArray : Interpolated I(q) values.
        """

        interpolated_q = np.array(interpolated_q)

        sorted_q = self.q[np.argsort(self.q)]
        sorted_Iq = self.Iq[np.argsort(self.q)]

        if mode == 'log':
            sorted_q = np.log10(sorted_q)
            sorted_Iq = np.log10(sorted_Iq)
        elif mode == 'linear':
            pass
        else:
            raise ValueError(
                f"Interpoaltion mode {mode} is not recognized. Please use"
                "eiter 'linear' or 'log'."
            )

        interpolated_Iq = np.interp(
            x=np.log10(interpolated_q) if mode == 'log' else interpolated_q,
            xp=sorted_q,
            fp=sorted_Iq
        )
        if mode == 'log':
            interpolated_Iq = np.power(10, interpolated_Iq)

        return interpolated_q, interpolated_Iq


class IntegratedQSlice(Data1D):
    """
    Child class of DataSlice that includes information about the
    integration performed to create the slice.

    Attributes
    ----------
    name : Unique name associated to the data image integrated.
    limits_axis0 : Indexing limits in the first dimension, [min, max).
    limits_axis1 : Indexing limits in the second dimension, [min, max).
    mode : Integration mode of either 'sum' or 'mean'
    integration_axis : Axis over which integration was performed, 0 or 1.
    """

    def __init__(self,
                 q: NDArray,
                 Iq: NDArray,
                 q_axis: NDArray,
                 name: str,
                 limits_axis0: tuple[int, int],
                 limits_axis1: tuple[int, int],
                 mode: str,
                 integration_axis: int,
                 dIq: NDArray = None,
                 dq: NDArray = None):

        # Base class init
        super().__init__(q=q, Iq=Iq, q_axis=q_axis, dIq=dIq, dq=dq)

        self.name = name
        self.limits_axis0 = limits_axis0
        self.limits_axis1 = limits_axis1
        self.mode = mode
        self.integration_axis = integration_axis
