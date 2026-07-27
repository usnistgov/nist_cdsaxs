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

from .._dtypes import REAL_DTYPE
from .data1d import Data1D


class ReducedData1D(Data1D):

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
            background_Iq=None,
            background_qslices=None,
            mask: NDArray = None,
            dIq: NDArray = None,
            wavelength_nm=None,
            sample_phi_deg=None,
            sample_chi_deg=None,
            sample_omega_deg=None,
            **kwargs
    ):
        """
        Child class of Data1D that includes information about the
        integration performed to create a 1D slice from a defined region of
        interest in a two-dimensional dataset or image.

        Parameters
        ----------
        q : NDArray
            Primary scattering vector values in Ang^-1.
        Iq : NDArray
            Scattering intensity as a function of q
        q_axis : str
            Name of the accepted q axis represented by q.
            See Data1D for the list of accepted axes.
        data2d : DataQdyQdx
            Source 2D dataset used to create the reduced profile. This stores
            a reference to the original objec as a pointer rather than a copy.
            Be cautions as the underling data could change after the creation of
            the q slice. However, the region of interest from the image actually
            used in the the integration will be saved as image_roi and will not change.
        limits_axis0 : tuple[int, int]
            Index limits [min, max) for the selected region of interest along
            axis 0 of the source image.
        limits_axis1 : tuple[int, int]
            Index limits [min, max) for the selected region of interest along
            axis 1 of the source image.
        integration_mode : str
            Integration mode used to reduce the ROI. Accepted values are
            'sum' and 'mean'.
        integration_axis : int
            Image axis over which the ROI was integrated. Accepted values are
            0 and 1.
        image_roi : NDArray
            Two-dimensional region of interest extracted from the source image
            and used for the integration.
        image_mask : NDArray
            Two-dimensional boolean mask for image_roi, where True marks
            pixels excluded from the integration.
        background_Iq : array-like | float, optional
            Background intensity subtracted from the reduced profile. Provide
            either a single value or an array with the same length as Iq.
        background_qslices : object, optional
            Background slice data associated with the reduction, if available.
        mask : NDArray, optional
            One-dimensional boolean mask for the reduced profile, where True
            marks points excluded from downstream operations. NaN values are
            masked automatically. Inf and -inf are not masked automatically.
        dIq : NDArray, optional
            Uncertainty values associated with Iq.
        wavelength_nm : float, optional
            X-ray wavelength in nm associated with the reduced profile.
        sample_phi_deg : float, optional
            Sample phi angle in degrees.
        sample_chi_deg : float, optional
            Sample chi angle in degrees.
        sample_omega_deg : float, optional
            Sample omega angle in degrees.

        **kwargs
        --------
        Any of the accepted scattering vectors or their components
        below can be provided as keyword arguments and become attributes of
        this class. Each value must either match the length of Iq or be a
        scalar that applies to the full profile.

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
        qbz : Scattering vector component along z axis of the lab frame; in
            the lab frame the beam path is aligned to the z-axis
        qb  : Scattering vector in the beam/lab frame; when the detector is
            positioned normal to the incident beam, the lab and detector
            coordinates will align

        Additionally, any of the scattering vectors or their components
        for the selected region of interest can be provided by appending
        _roi to the name. For example, qdy_roi would be a two-dimensional
        array containing qdy values for image_roi.

        """

        roi_kwargs = {key: value
                      for key, value in kwargs.items()
                      if '_roi' in key}
        other_kwargs = {key: value
                        for key, value in kwargs.items()
                        if '_roi' not in key}
        # Base class init
        super().__init__(
            q=q, Iq=Iq, q_axis=q_axis, dIq=dIq, mask=mask, **other_kwargs
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

        for key, value in roi_kwargs.items():
            setattr(self, key, value)

        if background_Iq is not None:
            background_Iq = np.asarray(background_Iq, dtype=REAL_DTYPE).reshape(-1)
            if len(background_Iq) == 1 or len(background_Iq) == len(self.Iq):
                self.background_Iq = background_Iq
            else:
                raise ValueError(
                    "The background should be a single float value or"
                    "an array of same length as Iq."
                )
        else:
            self.background_Iq = None
        self.background_qslices = background_qslices

        self.sample_phi_deg = sample_phi_deg
        self.sample_chi_deg = sample_chi_deg
        self.sample_omega_deg = sample_omega_deg
        self.wavelength_nm = wavelength_nm

    def mirror_q(self, q_axis=None, resort=True):
        """
        Mirror the data across q = 0 by taking the absolute value of q.
        No additional resampling or interpolation is performed.

        The primary q axis stored in q is used unless q_axis is specified.

        If resort is True, all arrays are reordered to match the mirrored
        q axis.

        Caution: This operation modifies the instance in place and cannot be undone.

        Parameters
        ----------
        q_axis : str, optional
            Name of the q axis to mirror. If None, the primary q axis is used.
        resort : bool, optional
            If True, reorder the data after mirroring so q remains sorted.

        Returns
        -------
        mirrored_data : None
            This method updates the current instance in place.
        """
        super().abs_q(q_axis=q_axis, resort=resort)
