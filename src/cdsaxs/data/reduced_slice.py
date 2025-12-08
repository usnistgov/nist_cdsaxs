"""
This module contains classes for handling one-dimensional data, I vs. q.

Data1D : General one-dimensional scattering data class for I vs. q.
IntegratedQSlice(Data1D) : Child class of Data1D. Contains one-
    dimensional scattering data extracted from integration across a
    defined area of two-dimensional scattering data. Holds historic
    information about the generation of the integrated slice.
"""

from __future__ import annotations

from numpy.typing import NDArray

from cdsaxs.data.data1d import Data1D


# class ReducedData1D(Data1D):
#     """
#     ReducedData is a child class of Data1D with additional attributes
#     required for conversion of detector coordinate space scattering data
#     to the sample coordinate space.
#     """

#     def __init__(self,
#                  q: NDArray,
#                  Iq: NDArray,
#                  q_axis: str,
#                  wavelength_nm: float,
#                  sample_phi_deg,
#                  sample_chi_deg,
#                  sample_omega_deg,
#                  dIq: NDArray = None,
#                  mask: NDArray = None,
#                  **kwargs):

#         """
#         Simple one-dimensional spectra of scattering intensity vs. q.

#         Parameters
#         ----------
#         q : scattering vector
#         Iq : scattering intensity as a function of q
#         q_axis : str
#             The axis for the provided scattering vector q from the
#             accepted list below. This will be designated as the primary
#             axis, but any of the other axes can be provided as keyword
#             arguments (see **kwargs section below).
#         wavelength_nm : float
#             Source wavelength during calculation of sample frame
#             scattering vector from detector frame scattering
#             vector. Units of nanometers.
#             Default is None.
#         sample_phi_deg : float
#             Sample rotation angle phi, corrected for any offsets
#             during processing of the data in detector space.
#         sample_chi_deg : float
#             Sample rotation angle chi, corrected for any offsets
#             during processing of the data in detector space.
#         sample_omega_deg : float
#             Sample rotation angle omega, corrected for any offsets
#             during processing of the data in detector space.
#         dIq : uncertainity along I, default is None
#             primary_q_axis : set the primary q-axis (listed below) for this
#             dataset. This can then be called with the basic 'q' attribute.
#             If left as None, the default axis will be set randomly to one
#             of the provided keyword arguments with the same length as Iq.
#         mask : NDArray
#             One-dimensional boolean array of same dimension as Iq that
#             are True at values that shoudl be masked out for all
#             operations.
#             All points that are nan will be masked out by default. It
#             will NOT mask out inf or -inf by default; this is different
#             behavior than the 2D data classes.

#         **kwargs
#         --------
#         Any of the accepted scattering vectors or their components
#         below can be provided as keyword arguments and become attributes
#         of this class. They must all be the same length as Iq or a single
#         value if applicable to the whole dataset.

#         qdy : Scattering vector component along y axis of detector frame.
#         qdx : Scattering vector component along x axis of detector frame.
#         qdz : Scattering vector component along z axis of detector frame
#         qd  : Scattering vector in the detector frame.
#         qsy : Scattering vector component along y axis of sample frame.
#         qsx : Scattering vector component along x axis of sample frame.
#         qsz : Scattering vector component along z axis of sample frame.
#         qs  : Scattering vector in the sample frame.
#         qby : Scattering vector component along y axis of the lab frame
#         qbx : Scattering vector component along x axis of the lab frame
#         qbz : Scattering vector comopnent along z axis of the lab frame; in
#             the lab frame the beam path is aligned to the z-axis 
#         qb  : Scattering vector in the beam/lab frame; when the detector is
#             positioned normal to the incident beam, the lab and detector
#             coordinates will align
#         """
#         super().__init__(
#             q=q, Iq=Iq, q_axis=q_axis, dIq=dIq, mask=mask, **kwargs
#         )

#         self.sample_phi_deg = sample_phi_deg
#         self.sample_chi_deg = sample_chi_deg
#         self.sample_omega_deg = sample_omega_deg
#         self.wavelength_nm = wavelength_nm


class ReducedData1DSlice(Data1D):
    """
    ReducedData is a child class of Data1D with additional attributes
    required for conversion of detector coordinate space scattering data
    to the sample coordinate space.
    """

    def __init__(self,
                 q: NDArray,
                 Iq: NDArray,
                 q_axis: str,
                 integrated_axis: str,
                 offset_axis: str,
                 slice_width: float,
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
        integrated_axis : str
            The axis that was integrated over to generate this slice.
            This will point to the keyword argument value that should
            also be provided.
        offset_axis : str
            The axis that is orthogonal to both the q_axis and the
            integrated_axis. This will point to the keyword argument
            value that should also be provided.
        slice_width : float
            Width of the integration box that generated this slice.
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
        qs  : Scattering vector in the sample frame.
        qby : Scattering vector component along y axis of the lab frame
        qbx : Scattering vector component along x axis of the lab frame
        qbz : Scattering vector comopnent along z axis of the lab frame; in
            the lab frame the beam path is aligned to the z-axis 
        qb  : Scattering vector in the beam/lab frame; when the detector is
            positioned normal to the incident beam, the lab and detector
            coordinates will align
        """
        super().__init__(
            q=q, Iq=Iq, q_axis=q_axis, dIq=dIq, mask=mask, **kwargs
        )

        self.slice_axis = integrated_axis
        self.offset_axis = offset_axis
        self.slice_width = slice_width
