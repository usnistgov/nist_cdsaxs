# -*- coding: utf-8 -*-
"""
This is part of the CDSAXS Data Processing GUI.
This module contains the diffraction equations.
"""

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np


# all thetas refer to full angles, so the formula for scattering vector is q = 4 pi sin(theta/2) / lambda

def center_px_qxz_on_img(center_px_qxz, pixel_um, SDD_cm, detector_thetascale, detector_theta, detector_theta0):
    SDD_px = SDD_cm / 1e-4 / pixel_um
    detector_theta_corr_rad = np.pi * detector_thetascale * (detector_theta - detector_theta0) / 180
    # [px] = ([cm] / 1e-4 / [um]) * tan([deg] * pi / 180)
    center_px_qxz_on_img = center_px_qxz - SDD_px * np.tan(detector_theta_corr_rad)
    return center_px_qxz_on_img


def center_px_qy_on_img(center_px_qy, pixel_um, detector_x, detector_x0):
    detector_x_corr_px = 1000 * (detector_x - detector_x0) / pixel_um
    # [px] = 1000 * [mm] / [um]
    center_px_qy_on_img = center_px_qy - detector_x_corr_px
    return center_px_qy_on_img


def qxz_pixels_to_qxz(qxz_pixels, lambda_nm, pixel_um, SDD_cm, detector_thetascale=1, detector_theta=0, detector_theta0=0):
    detector_theta_corr_rad = np.pi * detector_thetascale * (detector_theta - detector_theta0) / 180
    px_thetas_qxz_rad_uncorr = np.arctan2(1e-4 * qxz_pixels * pixel_um, SDD_cm)
    # [A^-1] = 0.1 * [nm^-1] * sin(atan(1e-4 * [um] / [cm]) + [deg] * pi / 180)
    qxzs = (0.4 * np.pi / lambda_nm) * np.sin(0.5 * (px_thetas_qxz_rad_uncorr + detector_theta_corr_rad))
    return qxzs


def qy_pixels_to_qy(qy_pixels, lambda_nm, pixel_um, SDD_cm, detector_x=0, detector_x0=0):
    detector_x_corr_cm = 0.1 * (detector_x - detector_x0)
    y_angles_rad = np.arctan2(1e-4 * qy_pixels * pixel_um + detector_x_corr_cm, SDD_cm)
    # [A^-1] = 0.1 * [nm^-1] * sin(atan((1e-4 * [um] + 0.1 * [mm]) / [cm]))
    qys = (0.4 * np.pi / lambda_nm) * np.sin(0.5 * y_angles_rad)
    return qys


def qxz_to_qz_qx(qxzs, qys, lambda_nm, sample_theta, samplethetaoffset=0):
    # positive sample_theta should result in positive qz
    # [rad] = asin([A^-1] * [nm] / 0.1)
    px_thetas_rad = np.arcsin(np.sqrt(qxzs ** 2 + qys ** 2) * lambda_nm / 0.4 / np.pi) * 2
    px_thetas_qxz_rad = np.arcsin(qxzs * lambda_nm / 0.4 / np.pi) * 2
    # samplethetaoffset_ = samplethetaoffset * (2 * (qxzs >= 0).astype(int) - 1)
    sample_theta_corr_rad = (sample_theta + samplethetaoffset) * np.pi / 180
    qzs = qxzs * np.sin(sample_theta_corr_rad + px_thetas_qxz_rad / 2)
    qxs = qxzs * np.cos(sample_theta_corr_rad + px_thetas_qxz_rad / 2)
    return qzs, qxs, px_thetas_rad, sample_theta_corr_rad


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
