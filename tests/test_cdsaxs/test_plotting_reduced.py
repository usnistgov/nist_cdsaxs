import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cdsaxs.data.dataset import ReducedDataset, ReducedSlices
from cdsaxs.data.reduced_data1d import ReducedData1D
from cdsaxs.data.reduced_slice import ReducedData1DSlice
from cdsaxs.plotting import plotting


class TestReducedPlotting(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def _make_reduced_data(
            self,
            name,
            qsx,
            qsz,
            iq,
            mask=None,
            sample_group=None):
        if mask is None:
            mask = np.zeros(len(iq), dtype=bool)

        data = ReducedData1D(
            q=np.array(qsz, dtype=float),
            Iq=np.array(iq, dtype=float),
            q_axis='qsz',
            qsx=np.array(qsx, dtype=float),
            qsy=np.zeros(len(iq), dtype=float),
            qsr=np.sqrt(np.array(qsx, dtype=float) ** 2),
            mask=np.array(mask, dtype=bool),
            data2d=name,
            limits_axis0=(0, 1),
            limits_axis1=(0, len(iq)),
            integration_mode='mean',
            integration_axis=0,
            image_roi=np.ones((1, len(iq)), dtype=float),
            image_mask=np.zeros((1, len(iq)), dtype=bool),
        )
        if sample_group is not None:
            data.sample_group = sample_group
        return data

    def _make_reduced_slice(self, q, iq, qsx, qsy):
        return ReducedData1DSlice(
            q=np.array(q, dtype=float),
            Iq=np.array(iq, dtype=float),
            q_axis='qsz',
            integrated_axis='qsx',
            offset_axis='qsy',
            slice_width=0.01,
            qsx=qsx,
            qsy=np.array(qsy, dtype=float),
        )

    def test_plot_reduced_dataset_returns_scatter_figure(self):
        reduced_dataset = ReducedDataset(datas=[
            self._make_reduced_data('data-1', [0.1, 0.2], [0.01, 0.02], [10.0, 20.0]),
            self._make_reduced_data('data-2', [0.3, 0.4], [0.03, 0.04], [30.0, 40.0]),
        ], name='scatter dataset')

        fig = plotting.plot_reduced_dataset(
            reduced_dataset,
            log_scale=False,
            interpolated_data=False,
        )

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig.axes[0].get_title(), 'scatter dataset')
        self.assertIn('q_{s,x}', fig.axes[0].get_xlabel())
        self.assertIn('q_{s,z}', fig.axes[0].get_ylabel())

    def test_plot_reduced_dataset_filters_by_metadata_and_q(self):
        reduced_dataset = ReducedDataset(datas=[
            self._make_reduced_data('keep', [0.1, 0.2], [0.01, 0.02], [10.0, 20.0], sample_group='A'),
            self._make_reduced_data('drop', [0.5, 0.6], [0.05, 0.06], [30.0, 40.0], sample_group='B'),
        ], name='filtered dataset')

        fig = plotting.plot_reduced_dataset(
            reduced_dataset,
            log_scale=False,
            filter_by_metadata={'sample_group': 'A'},
            filter_by_q={'qsx': (0.0, 0.3)},
        )

        offsets = fig.axes[0].collections[0].get_offsets()
        np.testing.assert_allclose(offsets, np.array([[0.1, 0.01], [0.2, 0.02]]))

    def test_plot_reduced_dataset_uses_new_interpolation_branch(self):
        reduced_dataset = ReducedDataset(datas=[
            self._make_reduced_data('data-1', [0.1, 0.2], [0.01, 0.02], [10.0, 20.0]),
            self._make_reduced_data('data-2', [0.3, 0.4], [0.03, 0.04], [30.0, 40.0]),
        ], name='interp dataset')

        x_interp = np.array([[0.1, 0.2], [0.1, 0.2]], dtype=float)
        y_interp = np.array([[0.01, 0.01], [0.02, 0.02]], dtype=float)
        iq_interp = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)

        with patch('cdsaxs.plotting.plotting.plotting_tools.new_interp_func', return_value=(x_interp, y_interp, iq_interp)) as interp_mock:
            fig = plotting.plot_reduced_dataset(
                reduced_dataset,
                interpolated_data=True,
                log_scale=False,
                grid_size=25,
            )

        interp_mock.assert_called_once()
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.axes[0].collections), 0)

    def test_plot_reduced_dataset_uses_legacy_interpolation_branch(self):
        reduced_dataset = ReducedDataset(datas=[
            self._make_reduced_data('data-1', [0.1, 0.2], [0.01, 0.02], [10.0, 20.0]),
        ], name='legacy interp dataset')

        x_interp = np.array([[0.1, 0.2], [0.1, 0.2]], dtype=float)
        y_interp = np.array([[0.01, 0.01], [0.02, 0.02]], dtype=float)
        iq_interp = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)

        with patch('cdsaxs.plotting.plotting.plotting_tools.generate_interpolated_reduced_data', return_value=(x_interp, y_interp, iq_interp)) as interp_mock:
            fig = plotting.plot_reduced_dataset(
                reduced_dataset,
                interpolated_data=True,
                use_legacy_interpolation=True,
                log_scale=False,
                grid_size=10,
                method='linear',
                distance_factor=2,
            )

        interp_mock.assert_called_once()
        self.assertIsNotNone(fig)

    def test_plot_slice_reduced_dataset_adds_slice_markers(self):
        reduced_dataset = ReducedDataset(datas=[
            self._make_reduced_data('data-1', [0.1, 0.2], [0.01, 0.02], [10.0, 20.0]),
        ], name='slice dataset')

        fig = plotting.plot_slice_reduced_dataset(
            reduced_dataset,
            q_bins=[(0.09, 0.1, 0.11), (0.19, 0.2, 0.21)],
            log_scale=False,
            slice_color='cyan',
            slice_lw=3,
        )

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes[0].patches), 2)
        self.assertEqual(len(fig.axes[0].collections), 3)

    def test_plot_reduced_slices_returns_figure_and_axes_with_sorted_labels(self):
        slice1 = self._make_reduced_slice(
            q=[0.3, 0.1, 0.2],
            iq=[3.0, 1.0, 2.0],
            qsx=0.4,
            qsy=[0.03, 0.01, 0.02],
        )
        slice2 = self._make_reduced_slice(
            q=[0.4, 0.2, 0.3],
            iq=[4.0, 2.0, 3.0],
            qsx=0.2,
            qsy=[0.04, 0.02, 0.03],
        )
        reduced_slices = ReducedSlices(slices=[slice1, slice2], name='reduced slices')

        fig, ax = plotting.plot_reduced_slices(
            reduced_slices,
            log_scale=False,
            offset_order=1,
            offset_value=5,
        )

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertEqual(legend_labels, ['0.2', '0.4'])
        np.testing.assert_allclose(ax.lines[0].get_xdata(), np.array([0.2, 0.3, 0.4]))
        np.testing.assert_allclose(ax.lines[1].get_xdata(), np.array([0.1, 0.2, 0.3]))

    def test_plot_reduced_slices_filters_out_non_matching_slices(self):
        keep_slice = self._make_reduced_slice(
            q=[0.1, 0.2],
            iq=[1.0, 2.0],
            qsx=0.3,
            qsy=[0.01, 0.02],
        )
        drop_slice = self._make_reduced_slice(
            q=[0.1, 0.2],
            iq=[3.0, 4.0],
            qsx=0.7,
            qsy=[0.05, 0.06],
        )
        reduced_slices = ReducedSlices(slices=[keep_slice, drop_slice])

        fig, ax = plotting.plot_reduced_slices(
            reduced_slices,
            filter_by_q={'qsy': (0.0, 0.03)},
            log_scale=False,
        )

        self.assertIsNotNone(fig)
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertEqual(legend_labels, ['0.3'])
