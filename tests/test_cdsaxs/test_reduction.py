import unittest

import numpy as np

from cdsaxs.data.dataset import ReducedDataset
from cdsaxs.data.reduced_data1d import ReducedData1D
from cdsaxs.reduction import slice_reduced_dataset


class TestSliceReducedDataset(unittest.TestCase):

    def _make_reduced_data(self, name, qsx_values, iq_values):
        qsz_values = np.array([0.01, 0.02, 0.03], dtype=float)
        qsy_values = np.array([0.0, 0.0, 0.0], dtype=float)
        return ReducedData1D(
            q=qsz_values,
            Iq=np.array(iq_values, dtype=float),
            q_axis='qsz',
            qsx=np.array(qsx_values, dtype=float),
            qsy=qsy_values,
            qsr=np.array(qsz_values, dtype=float),
            qs=np.array(qsz_values, dtype=float),
            mask=np.zeros(3, dtype=bool),
            data2d=name,
            limits_axis0=(0, 1),
            limits_axis1=(0, 3),
            integration_mode='mean',
            integration_axis=0,
            image_roi=np.ones((1, 3), dtype=float),
            image_mask=np.zeros((1, 3), dtype=bool),
            background=0.0,
        )

    def _make_dataset(self):
        data1 = self._make_reduced_data('data-1', [0.0995, 0.1000, 0.1005], [1.0, 2.0, 3.0])
        data2 = self._make_reduced_data('data-2', [0.1995, 0.2000, 0.2005], [4.0, 5.0, 6.0])
        return ReducedDataset(datas=[data1, data2], name='reduced-dataset')

    def _make_multi_selection_dataset(self):
        data1 = self._make_reduced_data('data-1', [0.0995, 0.1000, 0.1005], [1.0, 2.0, 3.0])
        data2 = self._make_reduced_data('data-2', [0.0996, 0.1001, 0.1004], [4.0, 5.0, 6.0])
        return ReducedDataset(datas=[data1, data2], name='multi-selection-dataset')

    def test_scalar_q_width_expands_for_each_q_value(self):
        reduced_slices, fig = slice_reduced_dataset(
            dataset=self._make_dataset(),
            q_values=[0.1, 0.2],
            q_widths=0.002,
            show_plot=False,
        )

        self.assertIsNone(fig)
        self.assertEqual(len(reduced_slices.data), 2)
        self.assertAlmostEqual(reduced_slices.data[0].slice_width, 0.002)
        self.assertAlmostEqual(reduced_slices.data[1].slice_width, 0.002)
        np.testing.assert_allclose(reduced_slices.data[0].Iq, np.array([2.0]))
        np.testing.assert_allclose(reduced_slices.data[1].Iq, np.array([5.0]))

    def test_mismatched_q_widths_length_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'same length as q_values'):
            slice_reduced_dataset(
                dataset=self._make_dataset(),
                q_values=[0.1, 0.2],
                q_widths=[0.002],
                show_plot=False,
            )

    def test_show_plot_false_returns_none_figure(self):
        reduced_slices, fig = slice_reduced_dataset(
            dataset=self._make_dataset(),
            q_values=[0.1],
            q_widths=0.002,
            show_plot=False,
        )

        self.assertIsNone(fig)
        self.assertEqual(len(reduced_slices.data), 1)

    def test_masked_selection_is_skipped(self):
        dataset = self._make_dataset()
        dataset.datas[0].mask[:] = True

        reduced_slices, _ = slice_reduced_dataset(
            dataset=dataset,
            q_values=[0.1],
            q_widths=0.002,
            show_plot=False,
        )

        self.assertEqual(len(reduced_slices.data), 1)
        self.assertEqual(reduced_slices.data[0].Iq.size, 0)
        self.assertEqual(reduced_slices.data[0].q.size, 0)

    def test_nan_intensity_selection_is_skipped(self):
        dataset = self._make_dataset()
        dataset.datas[0].Iq[1] = np.nan

        reduced_slices, _ = slice_reduced_dataset(
            dataset=dataset,
            q_values=[0.1],
            q_widths=0.002,
            show_plot=False,
        )

        self.assertEqual(len(reduced_slices.data), 1)
        self.assertEqual(reduced_slices.data[0].Iq.size, 0)
        self.assertEqual(reduced_slices.data[0].q.size, 0)

    def test_empty_selection_produces_empty_slice_arrays(self):
        reduced_slices, _ = slice_reduced_dataset(
            dataset=self._make_dataset(),
            q_values=[0.5],
            q_widths=0.002,
            show_plot=False,
        )

        self.assertEqual(len(reduced_slices.data), 1)
        self.assertEqual(reduced_slices.data[0].Iq.size, 0)
        self.assertEqual(reduced_slices.data[0].q.size, 0)
        self.assertAlmostEqual(reduced_slices.data[0].qsx, 0.5)

    def test_mode_mean_and_sum_produce_different_intensities(self):
        dataset = self._make_multi_selection_dataset()

        reduced_slices_mean, _ = slice_reduced_dataset(
            dataset=dataset,
            q_values=[0.1],
            q_widths=0.002,
            show_plot=False,
            mode='mean',
        )
        reduced_slices_sum, _ = slice_reduced_dataset(
            dataset=dataset,
            q_values=[0.1],
            q_widths=0.002,
            show_plot=False,
            mode='sum',
        )

        np.testing.assert_allclose(reduced_slices_mean.data[0].Iq, np.array([2.0, 5.0]))
        np.testing.assert_allclose(reduced_slices_sum.data[0].Iq, np.array([6.0, 15.0]))

    def test_slice_metadata_tracks_slice_and_integrated_axes(self):
        reduced_slices, _ = slice_reduced_dataset(
            dataset=self._make_dataset(),
            q_values=[0.1],
            q_widths=0.002,
            q_axis='qsx',
            slice_axis='qsz',
            offset_axis='qsy',
            show_plot=False,
        )

        reduced_slice = reduced_slices.data[0]
        self.assertEqual(reduced_slice.q_axis, 'qsz')
        self.assertEqual(reduced_slice.integrated_axis, 'qsx')
        self.assertEqual(reduced_slice.slice_axis, 'qsz')
        self.assertEqual(reduced_slice.offset_axis, 'qsy')
