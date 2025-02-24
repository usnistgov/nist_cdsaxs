# -*- coding: utf-8 -*-
"""
This is part of the CDSAXS Data Processing GUI.
This module generates .txt macro files for ALS beamline 11.0.1.2.
"""

from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import *

import io
import numpy as np
import pandas as pd

from CDSAXS_gui import diffraction, matplotlibwidget

EV_NM = 1239.84


def save_frame(frame, path_or_stream):
    """Saves dataframe containing target run parameters to tab-delimited file"""
    frame.to_csv(path_or_stream, sep='\t', index=False)

    def fix_frame(stream):
        """removes last tab from first line, note that the last column is exposure time"""
        lines = stream.readlines()
        if len(lines) == 0:
            return
        lines[0] = lines[0].replace('\t\n', '\n')
        stream.seek(0)
        stream.write(''.join(lines))
    if isinstance(path_or_stream, io.TextIOBase):
        path_or_stream.seek(0)
        fix_frame(path_or_stream)
    else:
        with open(path_or_stream, 'r+') as stream:
            fix_frame(stream)


def load_frame(path_or_stream):
    if isinstance(path_or_stream, io.TextIOBase):
        lines = path_or_stream.readlines()
    else:
        with open(path_or_stream) as stream:
            lines = stream.readlines()
    if len(lines) == 0:
        return pd.DataFrame()
    lines[0] = lines[0].replace('\n', '\t\n')
    with io.StringIO() as f:
        f.write(''.join(lines))
        f.seek(0)
        frame = pd.read_table(f)
    colnames = frame.columns.values
    colnames[-1] = ''
    frame.columns = colnames
    return frame


def make_frame(energy_ev, polarization, sample_xs, sample_ys, sample_zs, sample_thetas,
               detector_thetas, suppressors, exposures):
    """Makes dataframe containing target run parameters, args are 1-D or scalars (scalar args will be repeated for entire frame)"""
    items = []
    if energy_ev is not None:
        items.append(('Beamline Energy', energy_ev))
    if polarization is not None:
        items.append(('EPU Polarization', polarization))
    if sample_xs is not None:
        items.append(('Sample X', sample_xs))
    if sample_ys is not None:
        items.append(('Sample Y', sample_ys))
    if sample_zs is not None:
        items.append(('Sample Z', sample_zs))
    if sample_thetas is not None:
        items.append(('Sample Theta', sample_thetas))
    if detector_thetas is not None:
        items.append(('CCD Theta', detector_thetas))
    if suppressors is not None:
        items.append(('Higher Order Suppressor', suppressors))
    if exposures is not None:
        items.append(('', exposures))
    frame = pd.DataFrame.from_items(items)
    return frame


def insert_ref(frame, num_scans_to_insert=1, min_jump=30):
    """Inserts extra scans to track beam intensity
     Extra scans have same parameters as the normal incidence scan with their energy and EPU
     Scans are inserted at beginning and end, at jumps in sample theta, and optionally at intervals between those jumps"""
    index = ['Beamline Energy', 'EPU Polarization', 'Sample X', 'Sample Y',
             'Sample Z', 'Sample Theta', 'CCD Theta', 'Higher Order Suppressor', '']
    new_frame = pd.DataFrame(columns=index, dtype=np.float)
    filt = frame[frame['Sample Theta'] == -90]
    lastrow = 0
    lastsampletheta = frame['Sample Theta'].iloc[0]

    def series_to_insert(r):
        s = filt[(filt['EPU Polarization'] == frame['EPU Polarization'].iloc[r]) & (filt['Beamline Energy'] == frame['Beamline Energy'].iloc[r])]
        if s.shape[0] > 1:
            print('Warning: more than one possible normal incidence scan, copying the first one')
        elif s.shape[0] < 1:
            print('Error: missing a normal incidence scan to copy')
        return s.iloc[0]

    new_frame = new_frame.append(series_to_insert(lastrow), ignore_index=True)  # add scan at beginning
    for row, sample_theta in enumerate(frame['Sample Theta'].append(pd.Series(np.nan))):
        if (abs(sample_theta - lastsampletheta) >= min_jump) or (row == frame.shape[0]):
            for i in range(num_scans_to_insert):  # add scans when jump is detected
                new_frame = new_frame.append(frame.iloc[round(lastrow + i * (row - lastrow) / num_scans_to_insert): round(lastrow + (i + 1) * (row - lastrow) / num_scans_to_insert)], ignore_index=True)
                new_frame = new_frame.append(series_to_insert(lastrow), ignore_index=True)
            if (row < frame.shape[0]) and ((frame['EPU Polarization'].iloc[row] != frame['EPU Polarization'].iloc[row - 1]) or (frame['Beamline Energy'].iloc[row] != frame['Beamline Energy'].iloc[row - 1])):
                new_frame = new_frame.append(series_to_insert(row), ignore_index=True)  # add scan at beginning of new set
            lastrow = row
        lastsampletheta = sample_theta
    return new_frame


def insert_filler(frame, filler_exposure, filler_x, filler_y, filler_z):
    """Insert filler scan lasting filler_exposure seconds to dataframe in between each EPU Polarization change"""
    new_frame = pd.DataFrame()
    lastrow = 0
    lastpol = frame['EPU Polarization'].iloc[0]
    for row, pol in enumerate(frame['EPU Polarization']):
        if pol != lastpol:
            filler_series = frame.iloc[row].copy()
            filler_series[''] = filler_exposure
            filler_series['Sample X'] = filler_x
            filler_series['Sample Y'] = filler_y
            filler_series['Sample Z'] = filler_z
            new_frame = new_frame.append(frame.iloc[lastrow:row], ignore_index=True)
            new_frame = new_frame.append(filler_series, ignore_index=True)
            lastrow = row
            lastpol = pol
    new_frame = new_frame.append(frame.iloc[lastrow:], ignore_index=True)
    return new_frame


def find_peak_order(energy_ev, sample_thetas, detector_thetas, pitch_nm=140, pixel_um=27, SDD_cm=7.18,
                    beamcenter_xz_px=595, detector_theta0=-0.843, detector_thetascale=0.979, qxz_px_start=40):
    """Find lowest peak order visible on the detector on a set of scans"""
    qx_peaks = np.arange(20) * 0.2 * np.pi / pitch_nm
    sample_thetas = -np.asarray(sample_thetas) - 90
    lambda_nm = EV_NM / energy_ev
    qxz_px = qxz_px_start - beamcenter_xz_px
    qxzs = diffraction.qxz_pixels_to_qxz(qxz_px, lambda_nm, pixel_um, SDD_cm, detector_thetascale, detector_thetas, detector_theta0)
    _, qxs, _, _ = diffraction.qxz_to_qz_qx(qxzs, 0, lambda_nm, sample_thetas)
    peak_orders = np.searchsorted(qx_peaks, qxs)
    return peak_orders


def peak_order_to_neg_sample_thetas(peak_order, energy_ev, sample_theta, pitch_nm=140):
    """Given a peak order and a fits file sample theta that corresponds to (qx, qz), find the fits file sample theta that corresponds to (qx, -qz)"""
    qx = peak_order * 0.2 * np.pi / pitch_nm
    sample_theta = -sample_theta - 90
    lambda_nm = EV_NM / energy_ev
    qxz = diffraction.sample_theta_qx_to_qxz(sample_theta, qx, lambda_nm)
    qz, _, _, _ = diffraction.qxz_to_qz_qx(qxz, 0, lambda_nm, sample_theta, samplethetaoffset=0)
    neg_sample_thetas_rad, _ = diffraction.qx_qz_to_sample_theta(lambda_nm, qx, -qz)
    neg_sample_theta = -np.rad2deg(neg_sample_thetas_rad) - 90
    return neg_sample_theta


def make_detector_thetas(sample_thetas, avoid_peaks, energy_ev, pitch_nm=140, pixel_um=27, SDD_cm=7.18,
                         beamcenter_xz_px=595, detector_theta0=-0.843, detector_thetascale=0.979, qxz_px_start=40, offset=0.002):
    """Makes detector_thetas array to avoid the given and lower order peaks to improve dynamic range of run

    Args:
        sample_thetas: -90 is normal, as in dataframe, txt and fits files
            in rest of python code, sample theta is -(fits sample theta) - 90
        avoid_peaks: length = len(sample_thetas), 1 for first peak, etc.
        offset: amount to avoid peaks by in qx, units A^-1

    Returns:
        detector_thetas: length = len(sample_thetas)
    """
    sample_thetas = -np.asarray(sample_thetas) - 90
    lambda_nm = EV_NM / energy_ev
    avoid_qx = np.asarray(avoid_peaks) * 0.2 * np.pi / pitch_nm + offset
    qxz_px = qxz_px_start - beamcenter_xz_px
    qxzs = diffraction.sample_theta_qx_to_qxz(sample_thetas, avoid_qx, lambda_nm)
    detector_thetas = diffraction.qxz_to_detector_theta(
        qxzs, qxz_px, lambda_nm, pixel_um, SDD_cm, detector_thetascale, detector_theta0).round(2)
    return detector_thetas


def make_detector_thetas_new(sample_thetas, first_peaks, energy_ev, pitch_nm=140, pixel_um=27, SDD_cm=7.18,
                             beamcenter_xz_px=595, detector_theta0=-0.843, detector_thetascale=0.979, qxz_px_start=50):
    """Makes detector_thetas array to avoid the given and lower order peaks to improve dynamic range of run
    Floors detector_theta so motor doesn't need to move as often

    Args:
        sample_thetas: -90 is normal, as in dataframe, txt and fits files
            in rest of python code, sample theta is -(fits sample theta) - 90
            distance from normal must continuously increase within each group to avoid decreasing detector_theta
        first_peaks: for each sample_theta, the first peak to collect, starts at 1,
            length = len(sample_thetas) (typically same value repeated within each sample_theta group)

    Returns:
        detector_thetas: length = len(sample_thetas)
    """
    OFFSET_PEAK_ORDER_MIN = 0.2  # minimum amount to decrease first qx from first peak, in fractions of peak order
    OFFSET_FIRST_PEAK_ORDER_MIN = 0.5
    MAX_DETECTOR_THETA = 40
    FLOOR_INTERVAL = 1
    sample_thetas = -np.asarray(sample_thetas) - 90
    lambda_nm = EV_NM / energy_ev
    first_qxs = (np.asarray(first_peaks) - OFFSET_PEAK_ORDER_MIN) * 0.2 * np.pi / pitch_nm
    for i in range(len(sample_thetas)):
        if first_peaks[i] == 1:
            first_qxs[i] = (first_peaks[i] - OFFSET_FIRST_PEAK_ORDER_MIN) * 0.2 * np.pi / pitch_nm
    qxz_px = qxz_px_start - beamcenter_xz_px
    qxzs = diffraction.sample_theta_qx_to_qxz(sample_thetas, first_qxs, lambda_nm)
    detector_thetas = np.floor(diffraction.qxz_to_detector_theta(qxzs, qxz_px, lambda_nm, pixel_um, SDD_cm, detector_thetascale, detector_theta0) / FLOOR_INTERVAL) * FLOOR_INTERVAL
    for i in range(len(detector_thetas)):
        if np.isnan(detector_thetas[i]) or (detector_thetas[i] > MAX_DETECTOR_THETA):
            detector_thetas[i] = detector_thetas[i - 1]
    return detector_thetas


def make_sample_ys_zs(y90, z0, z180, sample_thetas):
    """Returns Sample Y and Sample Z 1D arrays, use this function if sample center of rotation not at target point
    At normal incidence, Sample Y direction is Qz, at grazing incidence, Sample Y direction is Qx
    At normal incidence, Sample Z direction is Qx, at grazing incidence, Sample Z direction is Qz
    see p. 119 of notebook

    Args:
        y90: Sample Y at sample_theta = -90 (normal incidence)
        z0: Sample Z at sample_theta = 0
        z180: Sample Z at sample_theta = -180
        sample_thetas: -90 is normal, as in dataframe, txt and fits files
            in rest of python code, sample theta is -(fits sample theta) - 90
        """
    sample_thetas = np.asarray(sample_thetas)
    sample_ys = (y90 + ((z0 - z180) / 2) * (np.sin(sample_thetas * np.pi / 180) + 1)).round(2)
    sample_zs = (z0 + ((z0 - z180) / 2) * (np.cos(sample_thetas * np.pi / 180) - 1)).round(2)
    return sample_ys, sample_zs


def plot_qz_qx(frame, pitch_nm=140, num_slices=13, pixel_um=27, SDD_cm=7.18, beamcenter_xz_px=595,
               detector_theta0=-0.843, detector_thetascale=0.979, qxz_px_start=40, qxz_px_end=990,
               detector_theta_plot='new', sample_theta_plot='new'):
    """Calculates and plots positions of measured points on Qz-Qx, colorbar represents sample_theta or detector_theta"""
    qzs = []
    qxs = []
    detector_thetas = []
    sample_thetas = []
    for energy_ev, detector_theta, sample_theta in zip(frame['Beamline Energy'], frame['CCD Theta'], frame['Sample Theta']):
        sample_theta = - sample_theta - 90
        lambda_nm = EV_NM / energy_ev
        qxz_pixels = np.arange(qxz_px_start, qxz_px_end) - beamcenter_xz_px
        qxzs = diffraction.qxz_pixels_to_qxz(qxz_pixels, lambda_nm, pixel_um, SDD_cm,
                                             detector_thetascale, detector_theta, detector_theta0)
        qzs_array, qxs_array, _, _ = diffraction.qxz_to_qz_qx(qxzs, 0, lambda_nm, sample_theta)
        qzs.extend(qzs_array)
        qxs.extend(qxs_array)
        detector_thetas.extend([detector_theta] * len(qxz_pixels))
        sample_thetas.extend([sample_theta] * len(qxz_pixels))

    peaks_qx = np.arange(1, num_slices + 1) * 0.2 * np.pi / pitch_nm

    if detector_theta_plot is not False:
        if detector_theta_plot == 'new':
            detector_theta_plot = matplotlibwidget.MatplotlibWidget2D()
        detector_theta_plot.scatter(
            qxs, qzs, logz=False, xlabel='$q_x (\AA^{-1})$', ylabel='$q_z (\AA^{-1})$',
            title='detector theta {0} eV'.format(energy_ev), s=10, c=detector_thetas, linewidths=0)
        for peak in peaks_qx:
            detector_theta_plot.figure.axes[0].axvline(peak)
        detector_theta_plot.canvas.draw()

    if sample_theta_plot is not False:
        if sample_theta_plot == 'new':
            sample_theta_plot = matplotlibwidget.MatplotlibWidget2D()
        sample_theta_plot.scatter(
            qxs, qzs, logz=False, xlabel='$q_x (\AA^{-1})$', ylabel='$q_z (\AA^{-1})$',
            title='sample theta {0} eV'.format(energy_ev), s=10, c=sample_thetas, linewidths=0)
        for peak in peaks_qx:
            sample_theta_plot.figure.axes[0].axvline(peak)
        sample_theta_plot.canvas.draw()

    return qzs, qxs


def print_resolution(frame, avoid_peaks, pitch_nm=140, num_slices=13, pixel_um=27, SDD_cm=7.18, beamcenter_xz_px=595,
                     detector_theta0=-0.843, detector_thetascale=0.979, qxz_px_start=40, qxz_px_end=990):
    """Prints vertical (qz) resolution of each slice"""
    assert(len(frame)) == len(avoid_peaks)
    peaks_qx = np.arange(1, num_slices + 1) * 0.2 * np.pi / pitch_nm
    for avoid_peak in set(avoid_peaks):
        print('avoid_peak=', avoid_peak)
        qzs = [[] for _ in range(num_slices)]  # list of qzs for each slice
        for energy_ev, detector_theta, sample_theta, current_avoid_peak in zip(
                frame['Beamline Energy'], frame['CCD Theta'], frame['Sample Theta'], avoid_peaks):
            if current_avoid_peak == avoid_peak:
                sample_theta = - sample_theta - 90
                lambda_nm = EV_NM / energy_ev
                for i in range(num_slices):  # for current slice append qz of current sample_theta
                    qxz = diffraction.sample_theta_qx_to_qxz(sample_theta, peaks_qx[i], lambda_nm)
                    qxz_min_max_pixels = np.array([qxz_px_start, qxz_px_end - 1]) - beamcenter_xz_px
                    qxz_min_max = diffraction.qxz_pixels_to_qxz(
                        qxz_min_max_pixels, lambda_nm, pixel_um, SDD_cm, detector_thetascale, detector_theta, detector_theta0)
                    # only append if qxz is within min and max qxz of current sample_theta
                    if qxz_min_max[0] <= qxz <= qxz_min_max[1]:
                        qzs[i].append(np.sqrt(qxz ** 2 - peaks_qx[i] ** 2))
        for i in range(num_slices):
            print('qx={0}, num_points={1}, qz_resolution={2}'.format(peaks_qx[i], len(qzs[i]), np.mean(np.abs(np.diff(qzs[i])))))
