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

from cdsaxs.data.data1d import Data1D


class QSlice(Data1D):

    def __init__(
            self,
            q: NDArray,
            Iq: NDArray,
            q_axis: str,
            data2d,
            limits_axis0: tuple[int, int],
            limits_axis1: tuple[int, int],
            integration_mode: str,
            integration_axis: int,
            image_roi: NDArray,
            image_mask: NDArray,
            background=None,
            mask: NDArray = None,
            dIq: NDArray = None,
            **kwargs
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
        mask : NDArray
            One-dimensional boolean array of same dimension as Iq that
            are True at values that shoudl be masked out for all
            operations.
            All points that are nan will be masked out by default. It
            will NOT mask out inf or -inf by default; this is different
            behavior than the 2D data classes.
        data2d : DataQdyQdx
            Instance of DataQdyQdx used to create the slice. This will
            create a pointer to that data instance in the original
            dataset rather than a copy of that instance. Be cautious as
            the underlying data could change after the creation of
            the q slice. However, the region of interest from the image
            actually used in the integration will be saved as
            the image_roi attribute of this class.
        limits_axis0 : tuple
            Indexing limits of the image region of interest
            in the first dimension, [min, max).
        limits_axis1 : tuple
            Indexing limits of the image region of interest
            in the second dimension, [min, max).
        integration_mode : str
            Integration mode of either 'sum' or 'mean'.
        integration_axis : int
            Axis over which integration was performed, 0 or 1.
        image_roi : NDArray
            Two dimensional region of interest selected from the original
            image over which the integration was performed.
        image_mask : NDArray
            Two dimensional array marking masked pixels during the
            operation.
        background : NDArray | float
            Background intensity subtracted during the integration step.
            This should be a single value or an array of same length as
            Iq.
        dIq : NDArray, optional
            Uncertainity along I.
            Default is None

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
        qs  : Scattering vector in the sample frame.
        qby : Scattering vector component along y axis of the lab frame
        qbx : Scattering vector component along x axis of the lab frame
        qbz : Scattering vector comopnent along z axis of the lab frame; in
            the lab frame the beam path is aligned to the z-axis
        qb  : Scattering vector in the beam/lab frame; when the detector is
            positioned normal to the incident beam, the lab and detector
            coordinates will align
        """

        # Base class init
        super().__init__(
            q=q, Iq=Iq, q_axis=q_axis, dIq=dIq, mask=mask, **kwargs
        )


        self.data2d = data2d

        # define limits of roi
        if len(limits_axis0) != 2 or len(limits_axis1) != 2:
            raise ValueError(
                "Length of axis limits should be two (min and max).")
        self.limits_axis0 = tuple(limits_axis0)
        self.limits_axis1 = tuple(limits_axis1)

        # define integration mode used
        if integration_mode not in ['sum', 'mean']:
            raise ValueError(f"An integration mode of {integration_mode} "
                             "is not accepted.")
        self.integration_mode = integration_mode

        # define integrated axis
        if integration_axis not in [0, 1]:
            raise ValueError(
                f"Integration axis {integration_axis} not understood; "
                "use 0 or 1.")
        self.integration_axis = integration_axis

        # make sure that the length of q matches the non-integrated axis
        if (integration_axis == 0
            and q.shape[0] != (limits_axis1[1] - limits_axis1[0])
            ) or (integration_axis == 1
                  and q.shape[0] != (limits_axis0[1] - limits_axis0[0])):
            raise ValueError(
                f"Your data has a length of {q.shape[0]} after integrating"
                f"along axis {integration_axis}, but this does not align with"
                f"the limits of "
                f"{limits_axis1 if integration_axis==0 else limits_axis0} "
                f"along axis {1-integration_axis}."
            )

        # ensure that the image roi and mask provided have dimensions that
        # correspond to the limits on axes 0 and 1
        self.image_roi = image_roi
        self.image_mask = image_mask

        if self.image_roi.shape != (
            limits_axis0[1] - limits_axis0[0],
            limits_axis1[1] - limits_axis1[0]
        ):
            raise ValueError(
                "The image_roi provided does not have dimensions corresponding"
                "to the limits for axes 0 and 1. The image has dimensions "
                f"of {self.image_roi.shape} but a shape of "
                f"{(limits_axis0[1] - limits_axis0[0], limits_axis1[1] - limits_axis1[0])} was expected."
            )
        if self.image_mask.shape != self.image_roi.shape:
            raise ValueError(
                "The image mask should have same dimensions as image roi."
            )

        if background is not None:
            background = np.array(background).reshape(-1).astype(float)
            if len(background) == 1 or len(background) == len(self.Iq):
                self.background = background
            else:
                raise ValueError(
                    "The background should be a single float value or"
                    "an array of same length as Iq."
                )

    def mirror_q(self, q_axis=None, resort=True):
        """
        Mirror the data across q = 0. This is identical to taking the
        absolute value of q for every data point. No additional
        resampling or interpolation is performed.

        The primary q-axis set as the q attribute will be used in this
        operation unless another q_axis is specified.

        If resort is left as True, all data will be ordered based on a
        resorting of the q axis that underwent the absolute value
        operation.

        Caution, this operation cannot be undone and the instance will
        have to be regenerated.
        """
        super().abs_q(q_axis=q_axis, resort=resort)
