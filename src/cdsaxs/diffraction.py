# -*- coding: utf-8 -*-
"""
This is part of the CDSAXS Data Processing GUI.
This module contains the diffraction equations.
"""

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
from numpy.typing import NDArray


def _detector_phi_corr(
        detector_phi_deg: float,
        detector_phi0_deg: float = 0,
        detector_phiscale: float = 1,
):
    """
    Calculates the angle of rotation of the detector about the sample
    in the xz plane (counterclockwise about positive y_s axis).

    Parameters
    ----------
    detector_phi_deg : float
        Nominal angle position of the detector rotated counterclockwise
        about the positive y axis at the sample position.
        Units are degrees.
    detector_phi0_deg : float
        Nominal angle position of the detector with in a normal
        incidence position.
        Units are degrees.
        Default value is 0.
    detector_phiscale : float
        Scaling factor applied to the corrected detector theta.

    Returns
    -------
    float
        Angle of rotation of the detector about the sample
        counterclockwise about the positive y axis at the sample
        position.
        Units are radians.
    """
    det_corr = detector_phiscale * (detector_phi_deg - detector_phi0_deg)
    det_corr_rad = np.deg2rad(det_corr)

    return det_corr_rad


def center_px_beam_to_detector(
        center_px_beam,
        pixel_size_um,
        sdd_cm,
        detector_phi_deg: float = 0,
        detector_y_mm: float = 0,
        detector_phi0_deg: float = 0,
        detector_y0_mm: float = 0,
        detector_phiscale: float = 1,
):
    """
    Converts the beam center position from beam coordinate space (which
    aligns with the detector coordinate space when the detector is at
    normal incidence with the beam) to detector coordinate space.

    Parameters
    ----------
    center_px_beam : tuple
        Beam center position in pixels in beam coordinate space or in
        detector space when the detector is at normal incidence.
    pixel_size_um : float
        Pixel size in microns.
    sdd_cm : float
        Sample to detector distance in cm.
    detector_phi_deg : float
        Nominal angle position of the detector rotated counterclockwise
        about the positive y axis at the sample position.
        Units are degrees.
    detector_y_mm : float
        Nominal y-position of detector after a translation along the
        positive y-axis.
    detector_phi0_deg : float
        Nominal angle position of the detector with in a normal
        incidence position.
        Units are degrees.
        Default value is 0.
    detector_y0_mm : float
        Nominal y-position of the detector in the reference position
        where the beam center position y in detector coordinate
        space matches the beam center position y in beam coordinate
        space.
        Default value is 0.
    detector_phiscale : float
        Scaling factor applied to the corrected detector theta.

    Returns
    -------
    tuple
        Beam center position in detector coordinate space.
    """

    det_phi_rad = _detector_phi_corr(
        detector_phi_deg=detector_phi_deg,
        detector_phi0_deg=detector_phi0_deg,
        detector_phiscale=detector_phiscale
    )

    offset_x_cm = sdd_cm * np.tan(det_phi_rad)
    offset_x_px = offset_x_cm * 1e4 / pixel_size_um

    offset_y_mm = detector_y_mm - detector_y0_mm
    offset_y_px = offset_y_mm * 1e3 / pixel_size_um

    center_px_detector = (
        float(center_px_beam[0] + offset_y_px),
        float(center_px_beam[1] + offset_x_px)
    )

    return center_px_detector


def center_px_detector_to_beam(
        center_px_detector,
        pixel_size_um,
        sdd_cm,
        detector_phi_deg: float = 0,
        detector_y_mm: float = 0,
        detector_phi0_deg: float = 0,
        detector_y0_mm: float = 0,
        detector_phiscale: float = 1,
):
    """
    Converts the beam center position from detector coordinate space to
    beam coordinate space (which aligns with the detector coordinate
    space when the detector is at normal incidence with the beam) to
    detector coordinate space.

    Parameters
    ----------
    center_px_deteector : tuple
        Beam center position in pixels in detector coordinate space.
    pixel_size_um : float
        Pixel size in microns.
    sdd_cm : float
        Sample to detector distance in cm.
    detector_phi_deg : float
        Nominal angle position of the detector rotated counterclockwise
        about the positive y axis at the sample position.
        Units are degrees.
    detector_y_mm : float
        Nominal y-position of detector after a translation along the
        positive y-axis.
    detector_phi0_deg : float
        Nominal angle position of the detector with in a normal
        incidence position.
        Units are degrees.
        Default value is 0.
    detector_y0_mm : float
        Nominal y-position of the detector in the reference position
        where the beam center position y in detector coordinate
        space matches the beam center position y in beam coordinate
        space.
        Default value is 0.
    detector_phiscale : float
        Scaling factor applied to the corrected detector theta.

    Returns
    -------
    tuple
        Beam center position in beam coordinate space.
    """

    det_phi_rad = _detector_phi_corr(
        detector_phi_deg=detector_phi_deg,
        detector_phi0_deg=detector_phi0_deg,
        detector_phiscale=detector_phiscale
    )

    offset_x_cm = sdd_cm * np.tan(det_phi_rad)
    offset_x_px = offset_x_cm * 1e4 / pixel_size_um

    offset_y_mm = detector_y_mm - detector_y0_mm
    offset_y_px = offset_y_mm * 1e3 / pixel_size_um

    center_px_beam = (
        center_px_detector[0] - offset_y_px,
        center_px_detector[1] - offset_x_px
    )

    return center_px_beam


def _detector_px_to_gamma_rad(
        center_px_beam: tuple,
        detector_shape_px: tuple,
        pixel_size_um: float,
        sdd_cm: float,
        detector_phi_deg: float = 0,
        detector_phi0_deg: float = 0,
        detector_phiscale: float = 1,
):
    """
    Calculate gamma, the horizontal scattering angle component, for
    each pixel in a detector image.

    Parameters
    ----------
    center_px_beam : tuple
        Beam center position in beam coordinate space.
    detector_shape_px : tuple
        Shape of the detector in number of pixels (y, x).
    pixel_size_um : float
        Pixel size in microns.
    sdd_cm : float
        Sample to detector distance in cm.
    detector_phi_deg : float
        Nominal angle position of the detector rotated counterclockwise
        about the positive y axis at the sample position.
        Units are degrees.
        Default value is 0.
    detector_phi0_deg : float
        Nominal angle position of the detector with in a normal
        incidence position.
        Units are degrees.
        Default value is 0.
    detector_phiscale : float
        Scaling factor applied to the corrected detector theta.

    Returns
    -------
    NDArray
        Horizontal scattering angle component, gamma, for each pixel.
    """
    # for each pixel calculate distance in x direction from the pixel
    # of normal incidence on the detector from the sample position
    x_px = -1*(
        np.tile(
            np.arange(0, detector_shape_px[1]),
            (detector_shape_px[0], 1)
        ) - center_px_beam[1]
    )
    # gamma is the horizontal angle of the scattered beam (x-direction)
    # from the incident beam so account for detector_phi too
    det_phi_corr = _detector_phi_corr(
        detector_phi_deg=detector_phi_deg,
        detector_phi0_deg=detector_phi0_deg,
        detector_phiscale=detector_phiscale
    )
    gamma = np.arctan2(x_px * pixel_size_um * 1e-4, sdd_cm) + det_phi_corr

    return gamma


def _detector_px_gamma_to_delta_rad(
        center_px_detector: tuple,
        gamma_rad: NDArray,
        detector_shape_px: tuple,
        pixel_size_um: float,
        sdd_cm: float,
        detector_phi_deg: float = 0,
        detector_phi0_deg: float = 0,
        detector_phiscale: float = 1,
):
    """
    Calculate delta, the vertical scattering angle component, for
    each pixel in a detector image after the horizontal scattering angle,
    gamma, has been applied.

    Parameters
    ----------
    center_px_detector : tuple
        Beam center position in detector coordinate space.
    gamma_rad : NDArray
        Horizontal scattering angle component in units of radians.
    detector_shape_px : tuple
        Shape of the detector in number of pixels (y, x).
    pixel_size_um : float
        Pixel size in microns.
    sdd_cm : float
        Sample to detector distance in cm.
    detector_phi_deg : float
        Nominal angle position of the detector rotated counterclockwise
        about the positive y axis at the sample position.
        Units are degrees.
        Default value is 0.
    detector_phi0_deg : float
        Nominal angle position of the detector with in a normal
        incidence position.
        Units are degrees.
        Default value is 0.
    detector_phiscale : float
        Scaling factor applied to the corrected detector theta.

    Returns
    -------
    float
        Scattering vector q_b in the beam coordinate space.
    float
        The y-component of the scattering vector, q_by.
    float
        The x-component of the scattering vector, q_bx.
    float
        The z-component of the scattering vector, q_bz.
    """
    det_phi_corr = _detector_phi_corr(
        detector_phi_deg=detector_phi_deg,
        detector_phi0_deg=detector_phi0_deg,
        detector_phiscale=detector_phiscale
    )

    # for each pixel calculate the distance in the y direction from the
    # pixel of normal incidence on the detector from the sample position
    y_px = -1*(np.tile(
            np.arange(0, detector_shape_px[0]),
            (detector_shape_px[1], 1)
        ).T - center_px_detector[0])

    # delta is the verticla angle of the scattered beam (y-direction)
    # after gamma has already been applied
    delta = np.arctan2(
        y_px * pixel_size_um * 1e-4,
        sdd_cm/np.cos(gamma_rad - det_phi_corr)
    )
    return delta


def detector_px_to_qbyxz(
        center_px: tuple,
        detector_shape_px: tuple,
        pixel_size_um: float,
        wavelength_nm: float,
        sdd_cm: float,
        center_coordinate_space: str = 'beam',
        detector_phi_deg: float = 0,
        detector_y_mm: float = 0,
        detector_phi0_deg: float = 0,
        detector_y0_mm: float = 0,
        detector_phiscale: float = 1,
):
    """
    Calculates qb, qby, qbx, and qbz (beam coordinate space) at all
    pixel positions in a detector.

    Parameters
    ----------
    center_px : tuple
        Beam center position.
    detector_shape_px : tuple
        Shape of the detector in number of pixels (y, x).
    pixel_size_um : float
        Pixel size in microns.
    wavelength_nm : float
        Source wavelength in nanometers.
    sdd_cm : float
        Sample to detector distance in cm.
    center_coordinate_space : str
        Coordinate space in which the beam center position is provided
        for center_px. This could be either 'beam' if it is in beam
        coordinate space (or detector space when the detector is at
        normal incidence) or this could be set to 'detector' if it is
        in detector coordinate space.
        Note that if there is no rotation or translation of the detector
        the center_px is the same for both detector and beam coordinate
        space.
        Default value is 'beam'.
    detector_phi_deg : float
        Nominal angle position of the detector rotated counterclockwise
        about the positive y axis at the sample position.
        Units are degrees.
        Default value is 0.
    detector_y_mm : float
        Nominal y-position of detector after a translation along the
        positive y-axis.
        Default value is 0.
    detector_phi0_deg : float
        Nominal angle position of the detector with in a normal
        incidence position.
        Units are degrees.
        Default value is 0.
    detector_y0_mm : float
        Nominal y-position of the detector in the reference position
        where the beam center position y in detector coordinate
        space matches the beam center position y in beam coordinate
        space.
        Default value is 0.
    detector_phiscale : float
        Scaling factor applied to the corrected detector theta.

    Returns
    -------
    float
        Scattering vector q_b in the beam coordinate space.
    float
        The y-component of the scattering vector, q_by.
    float
        The x-component of the scattering vector, q_bx.
    float
        The z-component of the scattering vector, q_bz.
    """

    # we will need the beam center position in both beam and detector
    # coordinates; if the detector phi and y are both 0, then these
    # coordinate systems align
    if center_coordinate_space.lower() == 'beam':
        center_px_beam = center_px
        center_px_detector = center_px_beam_to_detector(
            center_px_beam=center_px_beam,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_y_mm=detector_y_mm,
            detector_phi0_deg=detector_phi0_deg,
            detector_y0_mm=detector_y0_mm,
            detector_phiscale=detector_phiscale
        )
    elif center_coordinate_space.lower() == 'detector':
        center_px_detector = center_px
        center_px_beam = center_px_detector_to_beam(
            center_px_detector=center_px_detector,
            pixel_size_um=pixel_size_um,
            sdd_cm=sdd_cm,
            detector_phi_deg=detector_phi_deg,
            detector_y_mm=detector_y_mm,
            detector_phi0_deg=detector_phi0_deg,
            detector_y0_mm=detector_y0_mm,
            detector_phiscale=detector_phiscale
        )
    else:
        raise ValueError(
            "The center_coordinate_space keyword argument should be"
            "'beam' or 'detector.'"
        )

    gamma = _detector_px_to_gamma_rad(
        center_px_beam=center_px_beam,
        detector_shape_px=detector_shape_px,
        pixel_size_um=pixel_size_um,
        sdd_cm=sdd_cm,
        detector_phi_deg=detector_phi_deg,
        detector_phi0_deg=detector_phi0_deg,
        detector_phiscale=detector_phiscale,
    )

    delta = _detector_px_gamma_to_delta_rad(
        center_px_detector=center_px_detector,
        gamma_rad=gamma,
        detector_shape_px=detector_shape_px,
        pixel_size_um=pixel_size_um,
        sdd_cm=sdd_cm,
        detector_phi_deg=detector_phi_deg,
        detector_phi0_deg=detector_phi0_deg,
        detector_phiscale=detector_phiscale,
    )

    # calculate the q components in the beam coordinate space
    qb, qby, qbx, qbz = calculate_q_beam(
        gamma_deg=np.rad2deg(gamma),
        delta_deg=np.rad2deg(delta),
        wavelength_nm=wavelength_nm
    )

    return qb, qby, qbx, qbz


def calculate_q(theta_deg, wavelength_nm):
    """
    Calculate the scattering vector q given a full scattering angle
    theta and the wavelength. The scattering vector is returned in
    units of inverse Angstroms.

    Parameters
    ----------
    theta_deg : float | NDArray
        Scattering angle in degrees.
    wavelength_nm : float
        Source wavelength in nanometers.

    Returns
    -------
    q : float
        Scattering vector in units of inverse Angstroms.
    """
    q = 4 * np.pi * np.sin(np.deg2rad(theta_deg)/2) / (wavelength_nm * 10)

    return q


def calculate_q_beam(gamma_deg, delta_deg, wavelength_nm):
    """
    Calculate the beam coordinate q and its components from the
    scattering angle components gamma and delta.

    Parameters
    ----------
    gamma_deg : NDArray, float
        Scattering angle component along the x-axis.
        Units of degrees.
    delta_deg : NDArray, float
        Scattering angle component along the y-axis after gamma has
        been applied.
        Units of degrees.
    wavelength_nm : float
        Source wavelenght in nanometers.

    Returns
    -------
    float
        Scattering vector q_b in the beam coordinate space.
    float
        The y-component of the scattering vector, q_by.
    float
        The x-component of the scattering vector, q_bx.
    float
        The z-component of the scattering vector, q_bz.
    """
    # inverse Angstroms
    ko_ang = 2*np.pi/(wavelength_nm*10)

    # convert angles from degrees to radians
    gamma_rad = np.deg2rad(gamma_deg)
    delta_rad = np.deg2rad(delta_deg)

    # scattering vector components
    qbx = ko_ang * np.sin(gamma_rad) * np.cos(delta_rad)
    qby = ko_ang * np.sin(delta_rad)
    qbz = ko_ang * (np.cos(gamma_rad)*np.cos(delta_rad) - 1)

    # full scattering vector in beam coordinate space
    qb = np.sqrt(qbx**2 + qby**2 + qbz**2)

    # return in order y, x, z to match all other cd-saxs code
    return qb, qby, qbx, qbz


def active_rotation_Rx(omega_deg):
    """
    Calculate the active rotation matrix for angle omega
    counterclockwise about the positive x-axis.

    Parameters
    ----------
    omega_deg : float
        Counterclockwise rotation about the positive x-axis.

    Returns
    -------
    NDArray, 3 x 3
        Active rotation matrix, Rx.
    """
    omega_rad = np.deg2rad(omega_deg)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(omega_rad), -np.sin(omega_rad)],
        [0, np.sin(omega_rad), np.cos(omega_rad)],
    ])

    return Rx


def active_rotation_Ry(phi_deg):
    """
    Calculate the active rotation matrix for angle phi
    counterclockwise about the positive y-axis.

    Parameters
    ----------
    phi_deg : float
        Counterclockwise rotation about the positive y-axis.

    Returns
    -------
    NDArray, 3 x 3
        Active rotation matrix, Ry.
    """
    phi_rad = np.deg2rad(phi_deg)
    Ry = np.array([
        [np.cos(phi_rad), 0, np.sin(phi_rad)],
        [0, 1, 0],
        [-np.sin(phi_rad), 0, np.cos(phi_rad)],
    ])

    return Ry


def active_rotation_Rz(chi_deg):
    """
    Calculate the active rotation matrix for angle chi
    counterclockwise about the positive z-axis.

    Parameters
    ----------
    chi_deg : float
        Counterclockwise rotation about the positive z-axis.

    Returns
    -------
    NDArray, 3 x 3
        Active rotation matrix, Rz.
    """
    chi_rad = np.deg2rad(chi_deg)
    Rz = np.array([
        [np.cos(chi_rad), -np.sin(chi_rad), 0],
        [np.sin(chi_rad), np.cos(chi_rad), 0],
        [0, 0, 1],
    ])

    return Rz


def passive_rotation_Qx(omega_deg):
    """
    Calculate the active rotation matrix for angle omega
    counterclockwise about the positive x-axis.

    Parameters
    ----------
    omega_deg : float
        Counterclockwise rotation about the positive x-axis.

    Returns
    -------
    NDArray, 3 x 3
        Active rotation matrix, Qx.
    """
    omega_rad = np.deg2rad(omega_deg)
    Qx = np.array([
        [1, 0, 0],
        [0, np.cos(omega_rad), np.sin(omega_rad)],
        [0, -np.sin(omega_rad), np.cos(omega_rad)],
    ])

    return Qx


def passive_rotation_Qy(phi_deg):
    """
    Calculate the passive rotation matrix for angle phi
    counterclockwise about the positive y-axis.

    Parameters
    ----------
    phi_deg : float
        Counterclockwise rotation about the positive y-axis.

    Returns
    -------
    NDArray, 3 x 3
        Passive rotation matrix, Qy.
    """
    phi_rad = np.deg2rad(phi_deg)
    Qy = np.array([
        [np.cos(phi_rad), 0, -np.sin(phi_rad)],
        [0, 1, 0],
        [np.sin(phi_rad), 0, np.cos(phi_rad)],
    ])

    return Qy


def passive_rotation_Qz(chi_deg):
    """
    Calculate the passive rotation matrix for angle chi
    counterclockwise about the positive z-axis.

    Parameters
    ----------
    chi_deg : float
        Counterclockwise rotation about the positive z-axis.

    Returns
    -------
    NDArray, 3 x 3
        Passive rotation matrix, Qz.
    """
    chi_rad = np.deg2rad(chi_deg)
    Qz = np.array([
        [np.cos(chi_rad), np.sin(chi_rad), 0],
        [-np.sin(chi_rad), np.cos(chi_rad), 0],
        [0, 0, 1],
    ])

    return Qz


def get_passive_matrix(axis: str, angle_deg: float):
    """
    Returns the passive matrix for the selected axis and rotation angle.
    """
    if axis.lower() == 'x':
        Q = passive_rotation_Qx(omega_deg=angle_deg)
    elif axis.lower() == 'y':
        Q = passive_rotation_Qy(phi_deg=angle_deg)
    elif axis.lower() == 'z':
        Q = passive_rotation_Qz(chi_deg=angle_deg)
    else:
        raise ValueError(
            f"Axis of {axis} is not recognized; specify 'x', 'y', or 'z'."
        )

    return Q


def get_active_matrix(axis: str, angle_deg: float):
    """
    Returns the active matrix for the selected axis and rotation angle.
    """
    if axis.lower() == 'x':
        R = active_rotation_Rx(omega_deg=angle_deg)
    elif axis.lower() == 'y':
        R = active_rotation_Ry(phi_deg=angle_deg)
    elif axis.lower() == 'z':
        R = active_rotation_Rz(chi_deg=angle_deg)
    else:
        raise ValueError(
            f"Axis of {axis} is not recognized; specify 'x', 'y', or 'z'."
        )

    return R


def passive_intrinsic_rotation_matrix(
        first_axis: str,
        first_angle_deg: float,
        second_axis: str = None,
        second_angle_deg: float = None,
        third_axis: str = None,
        third_angle_deg: float = None,
        ):
    """
    Generate the passive rotation matrix for an intrinsic series of
    rotations. Each rotation in an intrinsic rotation is performed on
    the coordinate system after the previous rotation.

    Parameters
    ----------
    first_axis : str
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
    first_angle_deg : float
        Angle of rotation counterclockwise about the first_axis.
    second_axis : str, optional
        Axis about which the second rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    second_angle_deg : float, optional
        Angle of rotation counterclockwise about the second_axis.
        Units are degrees.
        Default is None.
    third_axis : str, optional
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    third_angle_deg : float, optional
        Angle of rotation counterclockwise about the third_axis.
        Units are degrees.
        Default is None.

    Returns
    -------
    NDArray, 3 x 3
        Passive rotation matrix for the series of intrinsic rotations
        specified.
    """

    Q1 = get_passive_matrix(first_axis, first_angle_deg)

    if second_axis is not None:
        Q2 = get_passive_matrix(second_axis, second_angle_deg)
    else:
        Q2 = np.identity(3)

    if third_axis is not None:
        Q3 = get_passive_matrix(third_axis, third_angle_deg)
    else:
        Q3 = np.identity(3)

    # for intrinsic, active rotation matrix is in order of operations
    # therefore, the passive rotation matrix is in reverse order
    Q = np.matmul(np.matmul(Q3, Q2), Q1)

    return Q


def passive_extrinsic_rotation_matrix(
        first_axis: str,
        first_angle_deg: float,
        second_axis: str = None,
        second_angle_deg: float = None,
        third_axis: str = None,
        third_angle_deg: float = None,
        ):
    """
    Generate the passive rotation matrix for an extrinsic series of
    rotations. Each rotation in an extrincsic rotation is performed on
    the original coordinate system before any rotations.

    Parameters
    ----------
    first_axis : str
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
    first_angle_deg : float
        Angle of rotation counterclockwise about the first_axis.
    second_axis : str, optional
        Axis about which the second rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    second_angle_deg : float, optional
        Angle of rotation counterclockwise about the second_axis.
        Units are degrees.
        Default is None.
    third_axis : str, optional
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    third_angle_deg : float, optional
        Angle of rotation counterclockwise about the third_axis.
        Units are degrees.
        Default is None.

    Returns
    -------
    NDArray, 3 x 3
        Passive rotation matrix for the series of extrinsic rotations
        specified.
    """

    Q1 = get_passive_matrix(first_axis, first_angle_deg)

    if second_axis is not None:
        Q2 = get_passive_matrix(second_axis, second_angle_deg)
    else:
        Q2 = np.identity(3)

    if third_axis is not None:
        Q3 = get_passive_matrix(third_axis, third_angle_deg)
    else:
        Q3 = np.identity(3)

    # for extrinsic, active rotation matrix is in reverse order of operations
    # therefore, the passive rotation matrix is in the order of rotations
    Q = np.matmul(np.matmul(Q1, Q2), Q3)

    return Q


def active_intrinsic_rotation_matrix(
        first_axis: str,
        first_angle_deg: float,
        second_axis: str = None,
        second_angle_deg: float = None,
        third_axis: str = None,
        third_angle_deg: float = None,
        ):
    """
    Generate the active rotation matrix for an intrinsic series of
    rotations. Each rotation in an intrinsic rotation is performed on
    the coordinate system after the previous rotation.

    Parameters
    ----------
    first_axis : str
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
    first_angle_deg : float
        Angle of rotation counterclockwise about the first_axis.
    second_axis : str, optional
        Axis about which the second rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    second_angle_deg : float, optional
        Angle of rotation counterclockwise about the second_axis.
        Units are degrees.
        Default is None.
    third_axis : str, optional
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    third_angle_deg : float, optional
        Angle of rotation counterclockwise about the third_axis.
        Units are degrees.
        Default is None.

    Returns
    -------
    NDArray, 3 x 3
        Active rotation matrix for the series of intrinsic rotations
        specified.
    """

    R1 = get_active_matrix(first_axis, first_angle_deg)

    if second_axis is not None:
        R2 = get_active_matrix(second_axis, second_angle_deg)
    else:
        R2 = np.identity(3)

    if third_axis is not None:
        R3 = get_active_matrix(third_axis, third_angle_deg)
    else:
        R3 = np.identity(3)

    # for intrinsic, active rotation matrix is in order of operations
    R = np.matmul(np.matmul(R1, R2), R3)

    return R


def active_extrinsic_rotation_matrix(
        first_axis: str,
        first_angle_deg: float,
        second_axis: str = None,
        second_angle_deg: float = None,
        third_axis: str = None,
        third_angle_deg: float = None,
        ):
    """
    Generate the active rotation matrix for an extrinsic series of
    rotations. Each rotation in an extrincsic rotation is performed on
    the original coordinate system before any rotations.

    Parameters
    ----------
    first_axis : str
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
    first_angle_deg : float
        Angle of rotation counterclockwise about the first_axis.
    second_axis : str, optional
        Axis about which the second rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    second_angle_deg : float, optional
        Angle of rotation counterclockwise about the second_axis.
        Units are degrees.
        Default is None.
    third_axis : str, optional
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    third_angle_deg : float, optional
        Angle of rotation counterclockwise about the third_axis.
        Units are degrees.
        Default is None.

    Returns
    -------
    NDArray, 3 x 3
        Active rotation matrix for the series of extrinsic rotat ions
        specified.
    """

    R1 = get_active_matrix(first_axis, first_angle_deg)

    if second_axis is not None:
        R2 = get_active_matrix(second_axis, second_angle_deg)
    else:
        R2 = np.identity(3)

    if third_axis is not None:
        R3 = get_active_matrix(third_axis, third_angle_deg)
    else:
        R3 = np.identity(3)

    # for extrinsic, active rotation matrix is in reverse order of operations
    # therefore, the passive rotation matrix is in the order of rotations
    R = np.matmul(np.matmul(R3, R2), R1)

    return R


def calculate_q_beam_to_sample(
    qby,
    qbx,
    qbz,
    sample_phi_deg: float,
    sample_chi_deg: float = 0,
    sample_omega_deg: float = 0,
    rotation: str = 'extrinsic',
    first_axis: str = 'y',
    second_axis: str = None,
    third_axis: str = None,
):
    """
    Calculate the scattering vector components in the sample coordinate
    space from the beam coordinate space. The beam coordinate space
    aligns with detector coordinate space when the detector is in a
    normal incidence position with the beam prior to any rotations or
    translations of the detector.

    Parameters
    ----------
    qby : NDArray
        The y-component to the scattering vector in the beam coordinate
        system.
        Units are in inverse Angstroms.
    qbx : NDArray
        The x-component to the scattering vector in the beam coordinate
        system.
        Units are in inverse Angstroms.
    qbz : NDArray
        The z-component to the scattering vector in the beam coordinate
        system.
        Units are in inverse Angstroms.
    sample_phi_deg : float
        Sample rotation angle counterclockwise about the positive y-axis
        at the sample position.
    sample_chi_deg : float, optional
        Sample rotation angle counterclockwise about the positive z-axis
        at the sample position.
        Default value is 0.
    sample_omega_deg : float, optional
        Sample rotation angle counterclockwise about the positive x-axis
        at the sample position.
        Default value is 0.
    rotation : str
        Define the series of sample rotations as 'extrinsic' or
        'intrinsic'.
        Default value is 'extrinsic'.
    first_axis : str
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default value is 'y' for standard CD-SAXS with one rotation
        of sample_phi_deg about the positive y-axis at the sample.
    second_axis : str, optional
        Axis about which the second rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.
    third_axis : str, optional
        Axis about which the first rotation is performed.
        Options are 'x', 'y', or 'z'.
        Default is None.

    Returns
    -------
    float
        Scattering vector, q_s, in the sample coordinate space.
    float
        The y-component of the scattering vector, q_sy, in sample
        coordinate space.
    float
        The x-component of the scattering vector, q_sx in sample
        coordinate space.
    float
        The z-component of the scattering vector, q_sz in sample
        coordinate space.
    """
    # check that the proper axes are provided for rotation angles
    if sample_chi_deg != 0 and sample_omega_deg != 0 and (
        second_axis is None or third_axis is None
    ):
        raise ValueError(
            "Non-zero values for chi and omega angles were provided but"
            "the second and third rotation axes were not specified."
        )
    elif (sample_chi_deg != 0 or sample_omega_deg != 0) and second_axis is None:
        raise ValueError(
            "You gave a non-zero value for chi or omega, but the second"
            "rotation axis was not specified."
        )

    # create qb matrix
    qb = np.array([qbx, qby, qbz]).reshape(3, -1)

    # assign rotation angles to proper order variables
    rot_angles = {
        'y': sample_phi_deg,
        'x': sample_omega_deg,
        'z': sample_chi_deg,
    }
    first_angle_deg = rot_angles[first_axis]
    second_angle_deg = rot_angles[second_axis] if second_axis is not None else 0
    third_angle_deg = rot_angles[third_axis] if third_axis is not None else 0

    # get passive rotation matrix
    if rotation.lower() == 'extrinsic':
        rot = passive_extrinsic_rotation_matrix(
            first_axis=first_axis,
            first_angle_deg=first_angle_deg,
            second_axis=second_axis,
            second_angle_deg=second_angle_deg,
            third_axis=third_axis,
            third_angle_deg=third_angle_deg,
        )
    elif rotation.lower() == 'intrinsic':
        rot = passive_intrinsic_rotation_matrix(
            first_axis=first_axis,
            first_angle_deg=first_angle_deg,
            second_axis=second_axis,
            second_angle_deg=second_angle_deg,
            third_axis=third_axis,
            third_angle_deg=third_angle_deg,
        )

    qs = np.matmul(rot, qb).reshape(3, qbx.shape[0], qbx.shape[1])

    # return in order of qs, qsy, qsx, qsz to match all cd-saxs code
    return np.linalg.norm(qs, axis=0), qs[1, :, :], qs[0, :, :], qs[2, :, :]


def detector_px_to_q(
        center_px: tuple,
        detector_shape_px: tuple,
        pixel_size_um: float,
        wavelength_nm: float,
        sdd_cm: float,
        sample_phi_deg: float,
        sample_chi_deg: float = 0,
        sample_omega_deg: float = 0,
        center_coordinate_space: str = 'beam',
        sample_rotation: str = 'extrinsic',
        sample_rotation_first_axis: str = 'y',
        sample_rotation_second_axis: str = None,
        sample_rotation_third_axis: str = None,
        detector_phi_deg: float = 0,
        detector_y_mm: float = 0,
        detector_phi0_deg: float = 0,
        detector_y0_mm: float = 0,
        detector_phiscale: float = 1,
):
    """
    Calculates the scattering vectors in both beam coordinate space and
    sample coordinate space for all pixels in a detector image.

    Parameters
    ----------
    center_px : tuple
        Beam center position.
    detector_shape_px : tuple
        Shape of the detector in number of pixels (y, x).
    pixel_size_um : float
        Pixel size in microns.
    wavelength_nm : float
        Source wavelength in nanometers.
    sdd_cm : float
        Sample to detector distance in cm.
    center_coordinate_space : str
        Coordinate space in which the beam center position is provided
        for center_px. This could be either 'beam' if it is in beam
        coordinate space (or detector space when the detector is at
        normal incidence) or this could be set to 'detector' if it is
        in detector coordinate space.
        Note that if there is no rotation or translation of the detector
        the center_px is the same for both detector and beam coordinate
        space.
        Default value is 'beam'.
    detector_phi_deg : float
        Nominal angle position of the detector rotated counterclockwise
        about the positive y axis at the sample position.
        Units are degrees.
        Default value is 0.
    detector_y_mm : float
        Nominal y-position of detector after a translation along the
        positive y-axis.
        Default value is 0.
    detector_phi0_deg : float
        Nominal angle position of the detector with in a normal
        incidence position.
        Units are degrees.
        Default value is 0.
    detector_y0_mm : float
        Nominal y-position of the detector in the reference position
        where the beam center position y in detector coordinate
        space matches the beam center position y in beam coordinate
        space.
        Default value is 0.
    detector_phiscale : float
        Scaling factor applied to the corrected detector theta.

    Returns
    -------
    float
        Scattering vector q_b in the beam coordinate space.
    tuple
        Components of the scattering vector in beam coordinate space:
        (qby, qbx, qbz)
    float
        Scattering vector q_b in the beam coordinate space.
    tuple
        Components of the scattering vector in beam coordinate space:
        (qby, qbx, qbz)
    """
    qb, qby, qbx, qbz = detector_px_to_qbyxz(
        center_px=center_px,
        detector_shape_px=detector_shape_px,
        pixel_size_um=pixel_size_um,
        wavelength_nm=wavelength_nm,
        sdd_cm=sdd_cm,
        center_coordinate_space=center_coordinate_space,
        detector_phi_deg=detector_phi_deg,
        detector_y_mm=detector_y_mm,
        detector_phi0_deg=detector_phi0_deg,
        detector_y0_mm=detector_y0_mm,
        detector_phiscale=detector_phiscale
    )

    qs, qsy, qsx, qsz = calculate_q_beam_to_sample(
        qby=qby, qbx=qbx, qbz=qbz,
        sample_phi_deg=sample_phi_deg,
        sample_chi_deg=sample_chi_deg,
        sample_omega_deg=sample_omega_deg,
        rotation=sample_rotation,
        first_axis=sample_rotation_first_axis,
        second_axis=sample_rotation_second_axis,
        third_axis=sample_rotation_third_axis
    )

    return qb, (qby, qbx, qbz), qs, (qsy, qsx, qsz)

# def qxz_pixels_to_qxz(qxz_pixels, lambda_nm, pixel_um, SDD_cm, detector_thetascale=1, detector_theta=0, detector_theta0=0):
#     detector_theta_corr_rad = np.pi * detector_thetascale * (detector_theta - detector_theta0) / 180
#     px_thetas_qxz_rad_uncorr = np.arctan2(1e-4 * qxz_pixels * pixel_um, SDD_cm)
#     # [A^-1] = 0.1 * [nm^-1] * sin(atan(1e-4 * [um] / [cm]) + [deg] * pi / 180)
#     qxzs = (0.4 * np.pi / lambda_nm) * np.sin(0.5 * (px_thetas_qxz_rad_uncorr + detector_theta_corr_rad))
#     return qxzs


# def qy_pixels_to_qy(qy_pixels, lambda_nm, pixel_um, SDD_cm, detector_x=0, detector_x0=0):
#     detector_x_corr_cm = 0.1 * (detector_x - detector_x0)
#     y_angles_rad = np.arctan2(1e-4 * qy_pixels * pixel_um + detector_x_corr_cm, SDD_cm)
#     # [A^-1] = 0.1 * [nm^-1] * sin(atan((1e-4 * [um] + 0.1 * [mm]) / [cm]))
#     qys = (0.4 * np.pi / lambda_nm) * np.sin(0.5 * y_angles_rad)
#     return qys


# def qxz_to_qz_qx(qxzs, qys, lambda_nm, sample_theta, samplethetaoffset=0):
#     # positive sample_theta should result in positive qz
#     # [rad] = asin([A^-1] * [nm] / 0.1)
#     px_thetas_rad = np.arcsin(np.sqrt(qxzs ** 2 + qys ** 2) * lambda_nm / 0.4 / np.pi) * 2
#     px_thetas_qxz_rad = np.arcsin(qxzs * lambda_nm / 0.4 / np.pi) * 2
#     # samplethetaoffset_ = samplethetaoffset * (2 * (qxzs >= 0).astype(int) - 1)
#     sample_theta_corr_rad = (sample_theta + samplethetaoffset) * np.pi / 180
#     qzs = qxzs * np.sin(sample_theta_corr_rad + px_thetas_qxz_rad / 2)
#     qxs = qxzs * np.cos(sample_theta_corr_rad + px_thetas_qxz_rad / 2)
#     return qzs, qxs, px_thetas_rad, sample_theta_corr_rad


# def sample_theta_qx_to_qxz(sample_thetas, qxs, lambda_nm):
#     # [] = 10 * [A^-1] * [nm] * sin([deg] * pi / 180)
#     a = 10 * qxs * lambda_nm * np.sin(sample_thetas * np.pi / 180)
#     # [] = cos([deg] * pi / 180)
#     b = np.pi * np.cos(sample_thetas * np.pi / 90)
#     # [A^-1] = 0.1 * [nm^-1] * sqrt(cos([rad]) * sqrt(100 * [A^-2] * [nm^2]))
#     qxzs = (np.sqrt(np.pi) / 5 / lambda_nm) * np.sqrt(
#         np.pi + b - a - np.cos(sample_thetas * np.pi / 180) * np.sqrt(2) * np.sqrt(
#             np.pi ** 2 - 50 * qxs ** 2 * lambda_nm ** 2 + np.pi * b - 2 * np.pi * a))
#     return qxzs


# def qxz_to_detector_theta(qxzs, qxzs_px, lambda_nm, pixel_um, SDD_cm, detector_thetascale=1, detector_theta0=0):
#     half_px_thetas_qxz_rad = np.arcsin(qxzs * lambda_nm / 0.4 / np.pi)
#     detector_theta_corr_rad = 2 * half_px_thetas_qxz_rad - np.arctan2(1e-4 * qxzs_px * pixel_um, SDD_cm)
#     # [deg] = (180 / pi) * atan(1e-4 * [um] / [cm])
#     detector_thetas = detector_theta0 + (180 / np.pi / detector_thetascale) * detector_theta_corr_rad
#     return detector_thetas


def qx_qz_to_sample_theta(lambda_nm, qxs, qzs):
    qxzs = np.sqrt(qzs ** 2 + qxs ** 2)
    half_px_thetas_qxz_rad = np.arcsin(qxzs * lambda_nm / 0.4 / np.pi)
    sample_thetas_rad = -half_px_thetas_qxz_rad + np.arcsin(qzs / qxzs)
    return sample_thetas_rad, 2 * half_px_thetas_qxz_rad


# def sample_theta_px_theta_to_qz_qx(sample_thetas, px_thetas, lambda_nm):
#     qxzs = (np.sin(px_thetas * np.pi / 360) * 0.4 * np.pi / lambda_nm)
#     qzs = qxzs * np.sin((sample_thetas + px_thetas / 2) * np.pi / 180)
#     qxs = qxzs * np.cos((sample_thetas + px_thetas / 2) * np.pi / 180)
#     return qzs, qxs


# def debye_waller(DW_factor, qxs, qzs):
#     DW_corr = np.exp(-(qxs ** 2 + qzs ** 2) * DW_factor ** 2)
#     return DW_corr
