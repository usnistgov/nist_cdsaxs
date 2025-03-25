r"""
This module contains diffraction related equations for the cdsaxs
package.

:math:`\theta` refers tot eh full scattering angle, and so the
scattering vector, :math:`q` is calculated by:
..math::

    q = \frac{4 \pi}{\lambda} \sin\left(\frac{\theta}{2}\right)
"""
import numpy as np
import numpy.typing as npt


def detector_theta_calc(
        detector_thetascale: float,
        detector_theta: float,
        detector_theta0: float
):
    """
    Calculates the angle of rotation of the detector about the sample in
    the qxz plane.

    TODO: provide some more information of geometry and scale parameter

    Parameters
    ----------
    detector_thetascale : float
        TODO: add parameter definition.
    detector_theta : float
        Nominal angle position of the detector rotated about the sample.
        Positive angle direction corresponds to positive qxz direction.
        Units are degrees.
    detector_theta0 : float
        Nominal angle position of the detector when the direct beam path
        is normal to the detector plane.
        Units are degrees.

    Returns
    -------
    float
        Angle of rotation of the detector about the sample in the qxz
        plane.
        Units are radians.
    """

    # rotation angle of the detector in radians
    detector_theta_corr_rad = np.deg2rad(detector_thetascale
                                         * (detector_theta - detector_theta0))
    return detector_theta_corr_rad


def center_px_qxz_on_img(
        center_px_qxz: int,
        pixel_size_qxz: float,
        SDD: float,
        detector_thetascale: float,
        detector_theta: float,
        detector_theta0: float,
):
    """
    Returns the beam center pixel position along qxz on a scattering
    image collected with a detector that has been rotated about the
    sample position in the qxz plane.

    TODO: insert a schematic that demonstrates this geometry.

    Parameters
    ----------
    center_px_qxz : int
        Beam center pixel position when the detector is in the 0 angle
        position, i.e., direct beam path normal to the detector plane.
    pixel_size_qxz : float
        Pixel size in the qxz direction.
        Units are in micrometers.
    SDD : float
        Sample to detector distance, i.e., distance from sample position
        to the detector along the vector normal to the detector plane.
        Units are in centimeters.
    detector_thetascale : float
        TODO: add parameter definition.
    detector_theta : float
        Nominal angle position of the detector rotated about the sample.
        Positive angle direction corresponds to positive qxz direction.
        Units are degrees.
    detector_theta0 : float
        Nominal angle position of the detector when the direct beam path
        is normal to the detector plane.
        Units are degrees.

    Returns
    -------
    int
        Pixel position of the beam center (:math:`q_{xz}=0) along the
        :math:`q_{xz}` axis.
    """
    SDD_px = SDD * 1e4 / pixel_size_qxz

    # rotation angle of the detector in radians
    detector_theta_corr_rad = detector_theta_calc(
        detector_thetascale=detector_thetascale, detector_theta=detector_theta,
        detector_theta0=detector_theta0
    )

    # [px] = [px] - [px] * tan([rad])
    center_px_qxz_on_img = center_px_qxz\
        - SDD_px * np.tan(detector_theta_corr_rad)

    return center_px_qxz_on_img


def center_px_qy_on_img(
        center_px_qy: int,
        pixel_size_qy: float,
        detector_x: float,
        detector_x0: float,
):
    """
    Returns the beam center pixel position along qy on a scattering
    image collected with a detector that has been translated along the
    qy axis.
    TODO: confirm above summary of the center_px_qy_on_img function

    TODO: insert a schematic that demonstrates this geometry.

    Parameters
    ----------
    center_px_qy : int
        Beam center pixel position when the detector is in the 0
        position prior to any translation.
    pixel_size_qy : float
        Pixel size in the qy direction.
        Units are in micrometers.
    detector_x : float
        Nominal position of the detector translated in the qy direction.
        Positive position corresponds to positive qy direction.
        Units are in millimeters.
    detector_x0 : float
        Nominal position of the detector along the qy axis when it is
        in the 0-position prior to any translation.
        Units are in millimeters.

    Returns
    -------
    int
        Pixel position of the beam center (:math:`q_{xz}=0) along the
        :math:`q_{xz}` axis.
    """
    # translation distance in pixels
    # [px] = 1000 * [mm] / [um]
    detector_x_corr_px = 1000 * (detector_x - detector_x0) / pixel_size_qy

    center_px_qy_on_img = center_px_qy - detector_x_corr_px
    return center_px_qy_on_img


def qxz_pixels_to_qxz(
        qxz_pixels: npt.NDArray[np.float64],
        wavelength: float,
        pixel_size_qxz: float,
        SDD: float,
        detector_thetascale: float = 1,
        detector_theta: float = 0,
        detector_theta0: float = 0,
):
    r"""
    Converts pixel position on detector in the qxz direction to the
    scattering vector :math:`q_{xz}` in units of :math:`\AA^{-1}`

    TODO: include schematic and equations of the geometry here.

    Parameters
    ----------
    qxz_pixels : ndarray
        Distance in pixels from the beam center position long qxz.
    wavelength : float
        Source wavelength.
        Units are in nanometers.
    pixel_size_qxz : float
        Pixel size in the qxz direction.
    SDD : float
        Sample to detector distance, i.e., distance from sample position
        to the detector along the vector normal to the detector plane.
        Units are in centimeters.
    detector_thetascale : float, optional
        TODO: add parameter definition
        Deafult value is 1.
    detector_theta : float, optional
        Nominal angle position of the detector rotated about the sample.
        Positive angle direction corresponds to positive qxz direction.
        Units are degrees.
        Default value is 0.
    detector_theta0 : float, optional
        Nominal angle position of the detector when the direct beam path
        is normal to the detector plane.
        Units are degrees.
        Default value is 0.

    Returns
    -------
    ndarray
        Scattering vector :math:`q_{xz}` in the units of :math:`\AA^{-1}`.
    """

    # angle of rotation of detector about sample in qzx plane
    detector_theta_corr_rad = detector_theta_calc(
        detector_thetascale=detector_thetascale, detector_theta=detector_theta,
        detector_theta0=detector_theta0
    )

    # scattering angle from center position of detector
    # [rad] = atan(1e-4 * [um] / [cm], [cm])
    px_thetas_qxz_rad_uncorr = np.arctan2(1e-4 * qxz_pixels * pixel_size_qxz,
                                          SDD)
    # [A^-1] = 0.1 * [nm^-1] * sin(atan(1e-4 * [um] / [cm]) + [deg] * pi / 180)

    # corrected scattering angle accounting for detector rotation
    px_thetas_qzx_rad_corr = px_thetas_qxz_rad_uncorr + detector_theta_corr_rad

    # convert to qxz
    # [Ang]^-1 = 0.1 * [nm]^-1 * sin([rad])
    qxzs = (0.4 * np.pi / wavelength) * np.sin(0.5 * px_thetas_qzx_rad_corr)

    return qxzs


def qy_pixels_to_qy(
        qy_pixels: npt.NDArray[np.float64],
        wavelength: float,
        pixel_size_qy: float,
        SDD: float,
        detector_x: float = 0,
        detector_x0: float = 0
):
    r"""
    Converts pixel position on detector in the qy direction to the
    scattering vector :math:`q_{y}` in units of :math:`\AA^{-1}`

    TODO: include schematic and equations of the geometry here.

    Parameters
    ----------
    qy_pixels : ndarray
        Distance in pixels from the beam center position long qxz.
    wavelength : float
        Source wavelength.
        Units are in nanometers.
    pixel_size_qy : float
        Pixel size in the qxz direction.
    SDD : float
        Sample to detector distance, i.e., distance from sample position
        to the detector along the vector normal to the detector plane.
        Units are in centimeters.
    detector_x : float
        Nominal position of the detector translated in the qy direction.
        Positive position corresponds to positive qy direction.
        Units are in millimeters.
    detector_x0 : float
        Nominal position of the detector along the qy axis when it is
        in the 0-position prior to any translation.
        Units are in millimeters.

    Returns
    -------
    ndarray
        Scattering vector :math:`q_{y}` in the units of :math:`\AA^{-1}`.
    """
    detector_y_pixels_offset = center_px_qy_on_img(
        center_px_qy=0, pixel_size_qy=pixel_size_qy, detector_x=detector_x,
        detector_x0=detector_x0
    )

    # scattering angles
    # [rad] = atan(1e-4 * [um] / [cm])
    y_angles_rad = np.arctan2(
        1e-4 * (qy_pixels - detector_y_pixels_offset) * pixel_size_qy, SDD
    )

    # convert to qxz
    # [Ang]^-1 = 0.1 * [nm]^-1 * sin([rad])
    qys = (0.4 * np.pi / wavelength) * np.sin(0.5 * y_angles_rad)

    return qys


def qxz_to_qz_qx(
        qxzs: npt.NDArray[np.float64],
        qys: float | npt.NDArray[np.float64],
        wavelength: float,
        sample_theta: float,
        samplethetaoffset: float = 0
):
    r"""
    Convert qxz to qz and qx components.
    Positive scattering angles result in positive qz values.

    TODO: put in geometry schematic/information

    Parameters
    ----------
    qxzs : ndarray
        Scattering vector component in xz direction, :math:`q_{xz}`.
        Units are in :math:`\AA^{-1}`.
    qys : float | ndarray
        Scattering vector component in the y direction, :math:`q_y`.
        Parameter can be a single float value (applied to all qxzs) or
        an ndarray matching the dimensions of qxzs.
        Units are in :math:`\AA^{-1}`.
    wavelength : float
        Source wavelength.
        Units are in nanometers.
    sample_theta : float
        Rotation angle of sample about the axis parallel to qy of the
        detector and passing through the direct beam path. An angle of 0
        is defined when the sample is in a position of normal incidence.
        If needed, this can be corrected with an angle offset via the
        sample_theta_offset parameter.
        Units are degrees.
    sample_theta_offset : float, optional
        Optional correction to the sample rotation angle to ensure an
        angle of 0 degrees is when the sample is at a position of normal
        indicence.
        Default value is 0.

    Returns
    -------
    ndarray
        Scattering vector, :math:`q_z`.
    ndarray
        Scattering vector, :math:`q_x`.
    ndarray
        Scattering angles. Units are radians.
    float
        Corrected sample rotation angle. If sample_theta_offset is not
        provided or kept at the default value of 0, this is equal to the
        value of sample_theta.

    
    """

    # positive sample_theta should result in positive qz
    # [rad] = asin([A^-1] * [nm] / 0.1)

    # scattering angles
    thetas_rad = np.arcsin(
        np.sqrt(qxzs ** 2 + qys ** 2) * wavelength / 0.4 / np.pi) * 2
    thetas_qxz_rad = np.arcsin(qxzs * wavelength / 0.4 / np.pi) * 2

    samplethetaoffset_ = samplethetaoffset * (2 * (qxzs >= 0).astype(int) - 1)
    sample_theta_corr_rad = np.deg2rad(sample_theta + samplethetaoffset_)

    # TODO: add in documentation coordinate systems and sign conventions
    qzs = qxzs * np.sin(sample_theta_corr_rad + thetas_qxz_rad / 2)
    qxs = qxzs * np.cos(sample_theta_corr_rad + thetas_qxz_rad / 2)

    return qzs, qxs, thetas_rad, sample_theta_corr_rad


def sample_theta_qx_to_qxz(sample_thetas, qxs, lambda_nm):
    # [] = 10 * [A^-1] * [nm] * sin([deg] * pi / 180)
    a = 10 * qxs * lambda_nm * np.sin(sample_thetas * np.pi / 180)
    # [] = cos([deg] * pi / 180)
    b = np.pi * np.cos(sample_thetas * np.pi / 90)
    # [A^-1] = 0.1 * [nm^-1] * sqrt(cos([rad]) * sqrt(100 * [A^-2] * [nm^2]))
    qxzs = (np.sqrt(np.pi) / 5 / lambda_nm) * np.sqrt(
        np.pi + b - a - np.cos(sample_thetas * np.pi / 180) * np.sqrt(2) * np.sqrt(
            np.pi ** 2 - 50 * qxs ** 2 * lambda_nm ** 2 + np.pi * b - 2 * np.pi * a))
    return qxzs


def qxz_to_detector_theta(qxzs, qxzs_px, lambda_nm, pixel_um, SDD_cm, detector_thetascale=1, detector_theta0=0):
    half_px_thetas_qxz_rad = np.arcsin(qxzs * lambda_nm / 0.4 / np.pi)
    detector_theta_corr_rad = 2 * half_px_thetas_qxz_rad - np.arctan2(1e-4 * qxzs_px * pixel_um, SDD_cm)
    # [deg] = (180 / pi) * atan(1e-4 * [um] / [cm])
    detector_thetas = detector_theta0 + (180 / np.pi / detector_thetascale) * detector_theta_corr_rad
    return detector_thetas


def qx_qz_to_sample_theta(lambda_nm, qxs, qzs):
    qxzs = np.sqrt(qzs ** 2 + qxs ** 2)
    half_px_thetas_qxz_rad = np.arcsin(qxzs * lambda_nm / 0.4 / np.pi)
    sample_thetas_rad = -half_px_thetas_qxz_rad + np.arcsin(qzs / qxzs)
    return sample_thetas_rad, 2 * half_px_thetas_qxz_rad


def sample_theta_px_theta_to_qz_qx(sample_thetas, px_thetas, lambda_nm):
    qxzs = (np.sin(px_thetas * np.pi / 360) * 0.4 * np.pi / lambda_nm)
    qzs = qxzs * np.sin((sample_thetas + px_thetas / 2) * np.pi / 180)
    qxs = qxzs * np.cos((sample_thetas + px_thetas / 2) * np.pi / 180)
    return qzs, qxs


def debye_waller(DW_factor, qxs, qzs):
    DW_corr = np.exp(-(qxs ** 2 + qzs ** 2) * DW_factor ** 2)
    return DW_corr

