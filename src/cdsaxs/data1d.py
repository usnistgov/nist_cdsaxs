"""
This module contains 1-dimensional data classes of:

Data1D : General one-dimensional scattering data class for I vs. q.
IntegratedQSlice(Data1D) : Child class of Data1D. Contains one-
    dimensional scattering data extracted from integration across a
    defined area of two-dimensional scattering data.
"""

from __future__ import annotations

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
    I : scattering intensity
    q_axis : Defines q as one of the accepted axes listed below.
    dI : uncertainity along I, default is None
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

    def __init__(self, q: NDArray, I: NDArray, q_axis: NDArray,
                 dI: NDArray = None, dq: NDArray = None):
        self.q = q
        self.I = I
        if q_axis not in ACCEPTED_Q_AXES:
            raise ValueError(
                f"{q_axis} is not an accepted q axis. Please select from: "
                f"{ACCEPTED_Q_AXES}"
            )
        else:
            self.q_axis = q_axis

        self.dI = dI
        self.dq = dq


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
    integration_axis : Axis over which integration was performed, either 0 or 1.
    """

    def __init__(self, q: NDArray, I: NDArray, q_axis: NDArray,
                 name: str, limits_axis0: tuple[int, int],
                 limits_axis1: tuple[int, int], mode: str,
                 integration_axis: int, dI: NDArray = None,
                 dq: NDArray = None):

        # Base class init
        super().__init__(q=q, I=I, q_axis=q_axis, dI=dI, dq=dq)

        self.name = name
        self.limits_axis0 = limits_axis0
        self.limits_axis1 = limits_axis1
        self.mode = mode
        self.integration_axis = integration_axis
