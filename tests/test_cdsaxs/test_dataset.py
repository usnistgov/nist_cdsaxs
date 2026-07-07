import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np

from cdsaxs.data.data2d import Data2D
from cdsaxs.data.dataset import Dataset, ReducedDataset, ReducedSlices
from cdsaxs.data.reduced_data1d import ReducedData1D
from cdsaxs.data.reduced_slice import ReducedData1DSlice


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset(name='test dataset')
        self.metadata = {
            'center_px': (1, 1),
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

    def _make_data(self, name, image_value):
        image = np.full((3, 3), image_value, dtype=np.float64)
        return Data2D(
            image=image,
            name=name,
            hide_q_warnings=True,
            **self.metadata,
        )

    def _make_filtered_dataset(self):
        data1 = self._make_data('data-1', 1)
        data2 = self._make_data('data-2', 2)
        data3 = self._make_data('data-3', 3)

        data1.update_metadata({'sample_phi_deg': -5}, hide_q_warnings=True)
        data2.update_metadata({'sample_phi_deg': 0}, hide_q_warnings=True)
        data3.update_metadata({'sample_phi_deg': 5}, hide_q_warnings=True)

        data1.update_user_params({'group': 'A', 'run': 1})
        data2.update_user_params({'group': 'B', 'run': 2})
        data3.update_user_params({'group': 'A', 'run': 3})

        self.dataset.add_data([data1, data2, data3])

        return data1, data2, data3

    def _make_mocked_dataset(self):
        data1 = self._make_data('data-1', 1)
        data2 = self._make_data('data-2', 2)
        data3 = self._make_data('data-3', 3)

        for data in [data1, data2, data3]:
            data.update_metadata = MagicMock()
            data.update_user_params = MagicMock()
            data.normalize_by_metadata = MagicMock()
            data.scale_by_metadata = MagicMock()
            data.normalize_data = MagicMock()
            data.scale_data = MagicMock()
            data.add_to_data = MagicMock()
            data.subtract_from_data = MagicMock()
            data.apply_footprint_correction = MagicMock()
            data.apply_sample_size_correction = MagicMock()
            data.apply_substrate_absorption_correction = MagicMock()
            data.reset_intensity = MagicMock()
            data.rotate_image_ccw = MagicMock()
            data.flip_horizontally = MagicMock()
            data.flip_vertically = MagicMock()
            data.reset_image = MagicMock()

        self.dataset.add_data([data1, data2, data3])

        return data1, data2, data3

    def _make_real_dataset_for_transformations(self):
        data1 = self._make_data('data-1', 1)
        data2 = self._make_data('data-2', 2)

        data1.image = np.array(
            [[1., 2., 3.],
             [4., 5., 6.]],
            dtype=np.float64,
        )
        data1._raw_image = np.copy(data1.image)
        data1.reset_mask(use_raw_image=True)

        data2.image = np.array(
            [[10., 20., 30.],
             [40., 50., 60.]],
            dtype=np.float64,
        )
        data2._raw_image = np.copy(data2.image)
        data2.reset_mask(use_raw_image=True)

        self.dataset.add_data([data1, data2])

        return data1, data2

    def _make_reduced_data1d(self, q_offset=0.0, label='data'):
        q = np.array([0.1, 0.2, 0.3], dtype=float) + q_offset
        image_roi = np.ones((2, 3), dtype=float)
        image_mask = np.zeros((2, 3), dtype=bool)
        return ReducedData1D(
            q=q,
            Iq=np.array([1.0, 2.0, 3.0], dtype=float),
            q_axis='qsx',
            data2d=label,
            limits_axis0=(0, 2),
            limits_axis1=(0, 3),
            integration_mode='sum',
            integration_axis=0,
            image_roi=image_roi,
            image_mask=image_mask,
            qsy=np.zeros_like(q),
        )

    def _make_reduced_slice(self, q_offset=0.0):
        q = np.array([0.1, 0.2, 0.3], dtype=float) + q_offset
        return ReducedData1DSlice(
            q=q,
            Iq=np.array([1.0, 2.0, 3.0], dtype=float),
            q_axis='qsz',
            integrated_axis='qsx',
            offset_axis='qsy',
            slice_width=0.01,
            qsy=np.zeros_like(q),
        )

    def _make_export_slice(self, q, iq, qsx, qsy, qsr=None):
        kwargs = {'qsy': np.array(qsy, dtype=float)}
        if qsr is not None:
            kwargs['qsr'] = np.array(qsr, dtype=float)

        return ReducedData1DSlice(
            q=np.array(q, dtype=float),
            Iq=np.array(iq, dtype=float),
            q_axis='qsz',
            integrated_axis='qsx',
            offset_axis='qsy',
            slice_width=0.01,
            qsx=qsx,
            **kwargs,
        )

    def _make_dataset_with_mocked_integrations(self):
        data1 = self._make_data('data-1', 1)
        data2 = self._make_data('data-2', 2)

        qslice1 = self._make_reduced_data1d(label='data-1')
        qslice2 = self._make_reduced_data1d(q_offset=0.5, label='data-2')

        data1.integrate_box = MagicMock(return_value=(qslice1, 'fig-1'))
        data2.integrate_box = MagicMock(return_value=(qslice2, 'fig-2'))

        self.dataset.add_data([data1, data2])

        return data1, data2, qslice1, qslice2

    def test_add_data_single_instance(self):
        data = self._make_data('data-1', 1)

        self.dataset.add_data(data)

        self.assertDictEqual(self.dataset.datas, {'data-1': data})

    def test_add_data_list_of_instances(self):
        data1 = self._make_data('data-1', 1)
        data2 = self._make_data('data-2', 2)

        self.dataset.add_data([data1, data2])

        self.assertDictEqual(
            self.dataset.datas,
            {'data-1': data1, 'data-2': data2},
        )

    def test_add_data_duplicate_name_raises(self):
        data1 = self._make_data('duplicate', 1)
        data2 = self._make_data('duplicate', 2)
        self.dataset.add_data(data1)

        with self.assertRaises(ValueError):
            self.dataset.add_data(data2)

    def test_remove_data_by_instance(self):
        data1 = self._make_data('data-1', 1)
        data2 = self._make_data('data-2', 2)
        self.dataset.add_data([data1, data2])

        self.dataset.remove_data(data1)

        self.assertDictEqual(self.dataset.datas, {'data-2': data2})

    def test_remove_data_by_key(self):
        data1 = self._make_data('data-1', 1)
        data2 = self._make_data('data-2', 2)
        self.dataset.add_data([data1, data2])

        self.dataset.remove_data('data-2')

        self.assertDictEqual(self.dataset.datas, {'data-1': data1})

    def test_remove_data_list_mixed_instance_and_key(self):
        data1 = self._make_data('data-1', 1)
        data2 = self._make_data('data-2', 2)
        data3 = self._make_data('data-3', 3)
        self.dataset.add_data([data1, data2, data3])

        self.dataset.remove_data([data1, 'data-3'])

        self.assertDictEqual(self.dataset.datas, {'data-2': data2})

    def test_remove_data_missing_entry_warns(self):
        data = self._make_data('data-1', 1)
        self.dataset.add_data(data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.dataset.remove_data('missing-data')

        self.assertEqual(len(caught), 1)
        self.assertIn('Could not delete missing-data data', str(caught[0].message))
        self.assertDictEqual(self.dataset.datas, {'data-1': data})

    def test_filter_data_by_metadata_tuple_range(self):
        self._make_filtered_dataset()

        keys = self.dataset.filter_data_by_metadata(sample_phi_deg=(-1, 5))

        self.assertCountEqual(keys, ['data-2', 'data-3'])

    def test_filter_data_by_metadata_scalar_value(self):
        self._make_filtered_dataset()

        keys = self.dataset.filter_data_by_metadata(sample_phi_deg=0)

        self.assertEqual(keys, ['data-2'])

    def test_filter_data_by_metadata_user_param(self):
        self._make_filtered_dataset()

        keys = self.dataset.filter_data_by_metadata(group='A')

        self.assertCountEqual(keys, ['data-1', 'data-3'])

    def test_filter_data_by_metadata_list_of_values(self):
        self._make_filtered_dataset()

        keys = self.dataset.filter_data_by_metadata(run=[1, 3])

        self.assertCountEqual(keys, ['data-1', 'data-3'])

    def test_filter_data_by_metadata_list_of_ranges(self):
        self._make_filtered_dataset()

        keys = self.dataset.filter_data_by_metadata(sample_phi_deg=[(-6, -4), (4, 6)])

        self.assertCountEqual(keys, ['data-1', 'data-3'])

    def test_filter_data_by_metadata_combines_filters(self):
        self._make_filtered_dataset()

        keys = self.dataset.filter_data_by_metadata(group='A', sample_phi_deg=(0, 10))

        self.assertEqual(keys, ['data-3'])

    def test_filter_data_by_metadata_invalid_filter_type_raises(self):
        self._make_filtered_dataset()

        with self.assertRaises(ValueError):
            self.dataset.filter_data_by_metadata(sample_phi_deg={'min': 0, 'max': 1})

    def test_filter_data_by_metadata_invalid_list_item_raises(self):
        self._make_filtered_dataset()

        with self.assertRaises(ValueError):
            self.dataset.filter_data_by_metadata(sample_phi_deg=[0, {'min': 0}])

    def test_update_all_metadata_applies_to_selected_keys(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.update_all_metadata(
            {'sample_phi_deg': 10},
            overwrite=False,
            keys=['data-1', 'data-3'],
            verbose=False,
        )

        data1.update_metadata.assert_called_once_with(
            metadata={'sample_phi_deg': 10},
            overwrite=False,
        )
        data2.update_metadata.assert_not_called()
        data3.update_metadata.assert_called_once_with(
            metadata={'sample_phi_deg': 10},
            overwrite=False,
        )

    def test_update_all_user_params_applies_to_all_by_default(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.update_all_user_params({'group': 'A'}, overwrite=False)

        for data in [data1, data2, data3]:
            data.update_user_params.assert_called_once_with(
                params={'group': 'A'},
                overwrite=False,
            )

    def test_normalize_all_data_by_metadata_applies_to_selected_keys(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.normalize_all_data_by_metadata(['I0'], keys=['data-2'])

        data1.normalize_by_metadata.assert_not_called()
        data2.normalize_by_metadata.assert_called_once_with(['I0'])
        data3.normalize_by_metadata.assert_not_called()

    def test_scale_all_data_by_metadata_applies_to_all_by_default(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.scale_all_data_by_metadata(['exposure_time_s'])

        for data in [data1, data2, data3]:
            data.scale_by_metadata.assert_called_once_with(['exposure_time_s'])

    def test_normalize_all_data_applies_scalar_to_selected_keys(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.normalize_all_data(5, keys=['data-1', 'data-2'])

        data1.normalize_data.assert_called_once_with(5)
        data2.normalize_data.assert_called_once_with(5)
        data3.normalize_data.assert_not_called()

    def test_scale_all_data_applies_array_to_all_by_default(self):
        data1, data2, data3 = self._make_mocked_dataset()
        scale = np.ones((3, 3), dtype=np.float64)

        self.dataset.scale_all_data(scale)

        for data in [data1, data2, data3]:
            data.scale_data.assert_called_once_with(scale)

    def test_add_to_all_data_applies_to_selected_keys(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.add_to_all_data(2, keys=['data-3'])

        data1.add_to_data.assert_not_called()
        data2.add_to_data.assert_not_called()
        data3.add_to_data.assert_called_once_with(2)

    def test_subtract_from_all_data_applies_to_all_by_default(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.subtract_from_all_data(3)

        for data in [data1, data2, data3]:
            data.subtract_from_data.assert_called_once_with(3)

    def test_apply_corrections_and_reset_intensity_respect_selected_keys(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.apply_footprint_correction(keys=['data-1'])
        self.dataset.apply_sample_size_correction(keys=['data-2'])
        self.dataset.apply_substrate_absorption_correction(keys=['data-3'])
        self.dataset.reset_all_data_intensity(keys=['data-1', 'data-3'])

        data1.apply_footprint_correction.assert_called_once_with()
        data2.apply_footprint_correction.assert_not_called()
        data3.apply_footprint_correction.assert_not_called()

        data1.apply_sample_size_correction.assert_not_called()
        data2.apply_sample_size_correction.assert_called_once_with()
        data3.apply_sample_size_correction.assert_not_called()

        data1.apply_substrate_absorption_correction.assert_not_called()
        data2.apply_substrate_absorption_correction.assert_not_called()
        data3.apply_substrate_absorption_correction.assert_called_once_with()

        data1.reset_intensity.assert_called_once_with()
        data2.reset_intensity.assert_not_called()
        data3.reset_intensity.assert_called_once_with()

    def test_rotate_flip_and_reset_apply_to_selected_keys(self):
        data1, data2, data3 = self._make_mocked_dataset()

        self.dataset.rotate_all_data_ccw(2, keys=['data-2'])
        self.dataset.flip_all_data_horizontally(keys=['data-1', 'data-3'])
        self.dataset.flip_all_data_vertically(keys=['data-3'])
        self.dataset.reset_all_data(keys=['data-1'])

        data1.rotate_image_ccw.assert_not_called()
        data2.rotate_image_ccw.assert_called_once_with(2)
        data3.rotate_image_ccw.assert_not_called()

        data1.flip_horizontally.assert_called_once_with()
        data2.flip_horizontally.assert_not_called()
        data3.flip_horizontally.assert_called_once_with()

        data1.flip_vertically.assert_not_called()
        data2.flip_vertically.assert_not_called()
        data3.flip_vertically.assert_called_once_with()

        data1.reset_image.assert_called_once_with()
        data2.reset_image.assert_not_called()
        data3.reset_image.assert_not_called()

    def test_scale_all_data_mutates_only_selected_real_data(self):
        data1, data2 = self._make_real_dataset_for_transformations()

        self.dataset.scale_all_data(2, keys=['data-1'])

        np.testing.assert_array_equal(
            data1.image,
            np.array([[2., 4., 6.], [8., 10., 12.]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            data2.image,
            np.array([[10., 20., 30.], [40., 50., 60.]], dtype=np.float64),
        )

    def test_rotate_all_data_ccw_mutates_only_selected_real_data(self):
        data1, data2 = self._make_real_dataset_for_transformations()

        self.dataset.rotate_all_data_ccw(1, keys=['data-2'])

        np.testing.assert_array_equal(
            data1.image,
            np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            data2.image,
            np.array([[30., 60.], [20., 50.], [10., 40.]], dtype=np.float64),
        )

    def test_reset_all_data_restores_only_selected_real_data(self):
        data1, data2 = self._make_real_dataset_for_transformations()

        self.dataset.flip_all_data_horizontally(keys=['data-1', 'data-2'])
        self.dataset.reset_all_data(keys=['data-1'])

        np.testing.assert_array_equal(
            data1.image,
            np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            data2.image,
            np.array([[30., 20., 10.], [60., 50., 40.]], dtype=np.float64),
        )

    def test_integrate_dataset_returns_reduced_dataset_for_selected_string_key(self):
        data1, data2, qslice1, _ = self._make_dataset_with_mocked_integrations()

        reduced_dataset = self.dataset.integrate_dataset(keys='data-1', axis='qdx')

        self.assertIsInstance(reduced_dataset, ReducedDataset)
        self.assertEqual(reduced_dataset.datas, [qslice1])
        data1.integrate_box.assert_called_once_with(
            mode='sum',
            axis='qdx',
            show_plot=False,
            subtract_background_offset=None,
            width_qdy_px=None,
            width_qdx_px=None,
            range_qdy_px=None,
            range_qdx_px=None,
            center_qdy=None,
            center_qdx=None,
            shift_box_qdy_px=0,
            shift_box_qdx_px=0,
        )
        data2.integrate_box.assert_not_called()

    def test_integrate_dataset_dispatches_dict_parameters_per_key(self):
        data1, data2, qslice1, qslice2 = self._make_dataset_with_mocked_integrations()

        reduced_dataset = self.dataset.integrate_dataset(
            keys=['data-1', 'data-2'],
            mode='mean',
            axis=0,
            subtract_background_offset=3,
            width_qdy_px={'data-1': 10, 'data-2': 20},
            width_qdx_px={'data-1': 11, 'data-2': 21},
            range_qdy_px={'data-1': (0, 2), 'data-2': (1, 3)},
            range_qdx_px={'data-1': (4, 6), 'data-2': (5, 7)},
            center_qdy={'data-1': ('qsy', 0.1), 'data-2': ('qsy', 0.2)},
            center_qdx={'data-1': ('qsx', 0.3), 'data-2': ('qsx', 0.4)},
            shift_box_qdy_px={'data-1': 1, 'data-2': 2},
            shift_box_qdx_px={'data-1': 3, 'data-2': 4},
        )

        self.assertEqual(reduced_dataset.datas, [qslice1, qslice2])
        data1.integrate_box.assert_called_once_with(
            mode='mean',
            axis=0,
            show_plot=False,
            subtract_background_offset=3,
            width_qdy_px=10,
            width_qdx_px=11,
            range_qdy_px=(0, 2),
            range_qdx_px=(4, 6),
            center_qdy=('qsy', 0.1),
            center_qdx=('qsx', 0.3),
            shift_box_qdy_px=1,
            shift_box_qdx_px=3,
        )
        data2.integrate_box.assert_called_once_with(
            mode='mean',
            axis=0,
            show_plot=False,
            subtract_background_offset=3,
            width_qdy_px=20,
            width_qdx_px=21,
            range_qdy_px=(1, 3),
            range_qdx_px=(5, 7),
            center_qdy=('qsy', 0.2),
            center_qdx=('qsx', 0.4),
            shift_box_qdy_px=2,
            shift_box_qdx_px=4,
        )

    def test_reduced_dataset_add_data_accepts_single_list_and_nested_dataset(self):
        qslice1 = self._make_reduced_data1d(label='data-1')
        qslice2 = self._make_reduced_data1d(q_offset=0.5, label='data-2')
        nested = ReducedDataset(datas=qslice2)
        reduced_dataset = ReducedDataset()

        reduced_dataset.add_data(qslice1)
        reduced_dataset.add_data([nested])

        self.assertEqual(reduced_dataset.datas, [qslice1, qslice2])

    def test_reduced_dataset_add_data_invalid_type_raises(self):
        reduced_dataset = ReducedDataset()

        with self.assertRaises(ValueError):
            reduced_dataset.add_data('not reduced data')

    def test_reduced_slices_add_slices_accepts_single_list_and_nested_container(self):
        slice1 = self._make_reduced_slice()
        slice2 = self._make_reduced_slice(q_offset=0.5)
        nested = ReducedSlices(slices=slice2)
        reduced_slices = ReducedSlices()

        reduced_slices.add_slices(slice1)
        reduced_slices.add_slices([nested])

        self.assertEqual(reduced_slices.data, [slice1, slice2])

    def test_reduced_slices_add_slices_invalid_type_raises(self):
        reduced_slices = ReducedSlices()

        with self.assertRaises(ValueError):
            reduced_slices.add_slices('not reduced slices')

    def test_reduced_slices_export_returns_sorted_padded_columns(self):
        slice1 = self._make_export_slice(
            q=[0.3, 0.1, 0.2],
            iq=[3.0, 1.0, 2.0],
            qsx=0.4,
            qsy=[0.03, 0.01, 0.02],
        )
        slice2 = self._make_export_slice(
            q=[0.4, 0.2, 0.3],
            iq=[4.0, np.nan, -1.0],
            qsx=0.8,
            qsy=[0.04, 0.02, 0.03],
        )
        reduced_slices = ReducedSlices(slices=[slice1, slice2])

        with TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / 'reduced_slices.csv'
            datas = reduced_slices.export_reduced_slices(filepath=export_path)

        expected = np.array([
            [r'$q_x (\AA^{-1})$', r'$q_y (\AA^{-1})$', r'$q_z (\AA^{-1})$', r'$I (A.U.)$', r'$q_x (\AA^{-1})$', r'$q_y (\AA^{-1})$', r'$q_z (\AA^{-1})$', r'$I (A.U.)$'],
            ['0.4', '0.01', '0.1', '1.0', '0.8', '0.04', '0.4', '4.0'],
            ['0.4', '0.02', '0.2', '2.0', '', '', '', ''],
            ['0.4', '0.03', '0.3', '3.0', '', '', '', ''],
        ], dtype=str)

        np.testing.assert_array_equal(datas, expected)

    def test_reduced_slices_export_includes_qr_when_requested(self):
        slice1 = self._make_export_slice(
            q=[0.2, 0.1],
            iq=[2.0, 1.0],
            qsx=0.5,
            qsy=[0.02, 0.01],
            qsr=[0.22, 0.11],
        )
        reduced_slices = ReducedSlices(slices=[slice1])

        with TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / 'reduced_slices_qr.csv'
            datas = reduced_slices.export_reduced_slices(
                filepath=export_path,
                export_qr=True,
            )

        expected = np.array([
            [r'$q_x (\AA^{-1})$', r'$q_y (\AA^{-1})$', r'$q_z (\AA^{-1})$', r'$q_r (\AA^{-1})$', r'$I (A.U.)$'],
            ['0.5', '0.01', '0.1', '0.11', '1.0'],
            ['0.5', '0.02', '0.2', '0.22', '2.0'],
        ], dtype=str)

        np.testing.assert_array_equal(datas, expected)

    def test_reduced_slices_export_filters_by_q_range(self):
        keep_slice = self._make_export_slice(
            q=[0.1, 0.2],
            iq=[1.0, 2.0],
            qsx=0.3,
            qsy=[0.01, 0.02],
        )
        drop_slice = self._make_export_slice(
            q=[0.1, 0.2],
            iq=[1.0, 2.0],
            qsx=0.7,
            qsy=[0.05, 0.06],
        )
        reduced_slices = ReducedSlices(slices=[keep_slice, drop_slice])

        with TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / 'filtered_reduced_slices.csv'
            datas = reduced_slices.export_reduced_slices(
                filepath=export_path,
                filter_by_q={'qsy': (0.0, 0.03)},
            )

        expected = np.array([
            [r'$q_x (\AA^{-1})$', r'$q_y (\AA^{-1})$', r'$q_z (\AA^{-1})$', r'$I (A.U.)$'],
            ['0.3', '0.01', '0.1', '1.0'],
            ['0.3', '0.02', '0.2', '2.0'],
        ], dtype=str)

        np.testing.assert_array_equal(datas, expected)

    def test_reduced_slices_export_handles_empty_and_partially_empty_slices(self):
        empty_slice = self._make_export_slice(
            q=[0.2, 0.1],
            iq=[0.0, np.nan],
            qsx=0.2,
            qsy=[0.02, 0.01],
        )
        partial_slice = self._make_export_slice(
            q=[0.3, 0.1, 0.2],
            iq=[-1.0, 5.0, np.nan],
            qsx=0.6,
            qsy=[0.03, 0.01, 0.02],
        )
        reduced_slices = ReducedSlices(slices=[empty_slice, partial_slice])

        with TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / 'empty_reduced_slices.csv'
            datas = reduced_slices.export_reduced_slices(filepath=export_path)

        expected = np.array([
            [r'$q_x (\AA^{-1})$', r'$q_y (\AA^{-1})$', r'$q_z (\AA^{-1})$', r'$I (A.U.)$', r'$q_x (\AA^{-1})$', r'$q_y (\AA^{-1})$', r'$q_z (\AA^{-1})$', r'$I (A.U.)$'],
            ['', '', '', '', '0.6', '0.01', '0.1', '5.0'],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
        ], dtype=str)

        np.testing.assert_array_equal(datas, expected)
