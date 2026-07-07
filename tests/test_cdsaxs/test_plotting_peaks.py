import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cdsaxs.data.data2d import Data2D
from cdsaxs.data.reduced_data1d import ReducedData1D
from cdsaxs.plotting import plotting


class TestPeakPlotting(unittest.TestCase):

    def setUp(self):
        image = np.array(
            [[44386., 20215., 48676., 20338.],
             [64811., 49702., 45711., 77210.],
             [4701., 9052., 61100., 46325.],
             [60019., 70069., np.nan, 65941.],
             [18439., 82922., 99999., 91001.],
             [10592., 19847., 10893., 41683.],
             [15954., 3109., 35295., 61517.]],
            dtype=float,
        )
        custom_mask = np.isnan(image)
        custom_mask[0, 0] = True
        custom_mask[3, 2] = False
        metadata = {
            'center_px': (4, 2),
            'wavelength_nm': 0.07,
            'pixel_size_um': 172,
            'sdd_cm': 542,
            'sample_phi_deg': 0,
            'sample_phi_offset_deg': 0,
            'sample_chi_deg': 0,
            'sample_chi_offset_deg': 0,
            'sample_omega_deg': 0,
            'sample_omega_offset_deg': 0,
            'detector_phi_deg': 0,
            'detector_phi0_deg': 0,
            'detector_phi_scale': 1,
            'detector_y_mm': 0,
            'detector_y0_mm': 0,
        }
        self.data2d = Data2D(
            image=image,
            name='plot peaks data',
            mask=custom_mask,
            hide_q_warnings=True,
            **metadata,
        )

    def tearDown(self):
        plt.close('all')

    def _make_qslice_with_backgrounds(self):
        background_slice_1 = ReducedData1D(
            q=np.array([0.001, 0.002, 0.003], dtype=float),
            Iq=np.array([2.0, 3.0, 4.0], dtype=float),
            q_axis='qbx',
            mask=np.zeros(3, dtype=bool),
            data2d=self.data2d,
            limits_axis0=(0, 2),
            limits_axis1=(0, 3),
            integration_mode='mean',
            integration_axis=0,
            image_roi=np.ones((2, 3), dtype=float),
            image_mask=np.zeros((2, 3), dtype=bool),
            background=0.0,
        )
        background_slice_2 = ReducedData1D(
            q=np.array([0.001, 0.002, 0.003], dtype=float),
            Iq=np.array([1.5, 2.5, 3.5], dtype=float),
            q_axis='qbx',
            mask=np.zeros(3, dtype=bool),
            data2d=self.data2d,
            limits_axis0=(0, 2),
            limits_axis1=(0, 3),
            integration_mode='mean',
            integration_axis=0,
            image_roi=np.ones((2, 3), dtype=float),
            image_mask=np.zeros((2, 3), dtype=bool),
            background=0.0,
        )
        return ReducedData1D(
            q=np.array([0.001, 0.002, 0.003], dtype=float),
            Iq=np.array([10.0, 20.0, 30.0], dtype=float),
            q_axis='qbx',
            mask=np.zeros(3, dtype=bool),
            data2d=self.data2d,
            limits_axis0=(0, 2),
            limits_axis1=(0, 3),
            integration_mode='mean',
            integration_axis=0,
            image_roi=np.ones((2, 3), dtype=float),
            image_mask=np.zeros((2, 3), dtype=bool),
            background_Iq=np.array([1.0, 2.0, 3.0], dtype=float),
            background_qslices=[background_slice_1, background_slice_2],
        )

    def test_plot_data2d_find_peaks2d_returns_zoomed_figure_with_peak_overlay(self):
        peaks = np.array([[2.5, 1.5], [4.0, 3.0]], dtype=float)

        fig = plotting.plot_data2d_find_peaks2d(
            self.data2d,
            peaks=peaks,
            limits_axis0=(1, 5),
            limits_axis1=(1, 4),
            zoom_plot=True,
            log_scale=False,
        )

        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertGreaterEqual(len(ax.lines), 2)
        self.assertEqual(ax.get_xlim(), (0.0, 4.0))
        self.assertEqual(ax.get_ylim(), (7.0, 0.0))

    def test_plot_data2d_find_peaks2d_handles_empty_peaks_without_overlay(self):
        fig = plotting.plot_data2d_find_peaks2d(
            self.data2d,
            peaks=np.empty((0, 2)),
            limits_axis0=(2, 4),
            limits_axis1=(1, 3),
            zoom_plot=False,
            log_scale=False,
        )

        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 1)

    def test_plot_data2d_find_detector_rotation_correction_adds_fit_line(self):
        peaks = np.array([[2.0, 1.0], [5.0, 2.0]], dtype=float)

        fig = plotting.plot_data2d_find_detector_rotation_correction(
            self.data2d,
            peaks=peaks,
            line=(-63.4349488229, -2.0, 0.0),
            limits_axis0=(0, 7),
            limits_axis1=(0, 4),
            zoom_plot=False,
            log_scale=False,
        )

        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 3)

    def test_plot_data2d_find_detector_rotation_correction_skips_nan_line(self):
        peaks = np.array([[2.0, 1.0], [5.0, 2.0]], dtype=float)

        fig = plotting.plot_data2d_find_detector_rotation_correction(
            self.data2d,
            peaks=peaks,
            line=(np.nan, np.nan, np.nan),
            limits_axis0=(0, 7),
            limits_axis1=(0, 4),
            zoom_plot=False,
            log_scale=False,
        )

        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)

    def test_plot_data2d_find_beam_center_adds_crosshairs_for_tuple_or_metadata_center(self):
        peaks = np.array([[4.0, 1.0], [4.0, 3.0]], dtype=float)

        result = plotting.plot_data2d_find_beam_center(
            self.data2d,
            peaks=peaks,
            limits_axis0=(2, 6),
            limits_axis1=(0, 4),
            show_beam_center=(3.5, 1.5),
            zoom_plot=False,
            log_scale=False,
        )

        self.assertIsNone(result)
        fig = plt.gcf()
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 4)

        result = plotting.plot_data2d_find_beam_center(
            self.data2d,
            peaks=peaks,
            limits_axis0=(2, 6),
            limits_axis1=(0, 4),
            show_beam_center=True,
            zoom_plot=False,
            log_scale=False,
        )

        self.assertIsNone(result)
        fig = plt.gcf()
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 4)

    def test_plot_data2d_find_beam_center_can_hide_crosshairs(self):
        peaks = np.array([[4.0, 1.0], [4.0, 3.0]], dtype=float)

        result = plotting.plot_data2d_find_beam_center(
            self.data2d,
            peaks=peaks,
            limits_axis0=(2, 6),
            limits_axis1=(0, 4),
            show_beam_center=False,
            zoom_plot=False,
            log_scale=False,
        )

        self.assertIsNone(result)
        fig = plt.gcf()
        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 2)

    def test_plot_data2d_find_sdd_sets_title_for_tuple_and_scalar_inputs(self):
        peaks = np.array([[4.0, 1.0], [4.0, 3.0]], dtype=float)

        fig = plotting.plot_data2d_find_sdd(
            self.data2d,
            peaks=peaks,
            limits_axis0=(2, 6),
            limits_axis1=(0, 4),
            sdd_cm=(24.57, 0.0),
            zoom_plot=False,
            log_scale=False,
        )

        self.assertIsNotNone(fig)
        self.assertIn('24.57 cm +/- 0.0 cm', fig.axes[0].get_title())

        fig = plotting.plot_data2d_find_sdd(
            self.data2d,
            peaks=peaks,
            limits_axis0=(2, 6),
            limits_axis1=(0, 4),
            sdd_cm=np.array([24.57]),
            zoom_plot=False,
            log_scale=False,
        )

        self.assertIsNotNone(fig)
        self.assertIn('24.57 cm', fig.axes[0].get_title())

    def test_plot_qslice_background_figure_uses_log_scale_when_requested(self):
        qslice = self._make_qslice_with_backgrounds()

        fig_box, fig_slice, fig_background = plotting.plot_qslice(
            qslice,
            log_scale=True,
            show_backgrounds=True,
        )

        self.assertIsNotNone(fig_box)
        self.assertIsNotNone(fig_slice)
        self.assertIsNotNone(fig_background)
        self.assertEqual(fig_slice.axes[0].get_yscale(), 'log')
        self.assertEqual(fig_background.axes[0].get_yscale(), 'log')

    def test_plot_image_adds_mask_and_invalid_overlays(self):
        image = np.array(
            [[1.0, 2.0],
             [np.inf, np.nan]],
            dtype=float,
        )
        mask = np.array(
            [[False, True],
             [False, False]],
            dtype=bool,
        )

        fig = plotting.plot_image(
            image,
            mask=mask,
            log_scale=False,
            color_mask='blue',
            color_inf='green',
            color_nan='red',
        )

        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.images), 2)
        np.testing.assert_array_equal(
            ax.images[0].get_array(),
            np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=float),
        )
        np.testing.assert_array_equal(
            ax.images[1].get_array(),
            np.array([[np.nan, 0.0], [0.5, 1.0]], dtype=float),
        )

    def test_plot_image_log_scale_handles_single_positive_finite_value(self):
        image = np.array(
            [[1.0, 0.0],
             [np.inf, np.nan]],
            dtype=float,
        )
        mask = np.array(
            [[False, True],
             [False, False]],
            dtype=bool,
        )

        fig = plotting.plot_image(
            image,
            mask=mask,
            log_scale=True,
        )

        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(len(ax.images), 2)
        self.assertEqual(ax.images[0].norm.vmin, 1.0)
        self.assertEqual(ax.images[0].norm.vmax, 10.0)

    def test_plot_image_add_roi_draws_expected_outline(self):
        fig = plotting.plot_image(np.arange(9, dtype=float).reshape(3, 3), mask=np.zeros((3, 3), dtype=bool), log_scale=False)

        fig = plotting.plot_image_add_roi(
            limits_axis0=(1, 3),
            limits_axis1=(0, 2),
            fig=fig,
            show_legend=False,
            color='black',
        )

        ax = fig.axes[0]
        line = ax.lines[-1]
        np.testing.assert_array_equal(line.get_xdata(), np.array([-0.5, -0.5, 1.5, 1.5, -0.5]))
        np.testing.assert_array_equal(line.get_ydata(), np.array([0.5, 2.5, 2.5, 0.5, 0.5]))

    def test_plot_errorbar_forwards_log_scale_flags_to_axes(self):
        fig = plotting.plot_errorbar(
            x=np.array([1.0, 10.0], dtype=float),
            y=np.array([2.0, 20.0], dtype=float),
            log_scale_x=True,
            log_scale_y=True,
            show_legend=False,
        )

        ax = fig.axes[0]
        self.assertEqual(ax.get_xscale(), 'log')
        self.assertEqual(ax.get_yscale(), 'log')
