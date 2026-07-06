import unittest
import warnings

import numpy as np

from cdsaxs.data.data2d import Data2D
from cdsaxs.data.dataset import Dataset


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
