import unittest

from cdsaxs.metadata import correct_metadata_dtype, check_metadata
from cdsaxs.metadata import METADATA_KEYWORDS, FLOATS, INTEGERS, STRINGS
from cdsaxs.metadata import SAMPLE_METADATA_KEYWORDS


class TestCorrectMetadataDtype(unittest.TestCase):

    def test_floats(self):
        string_input = 20.2
        for key in FLOATS:
            self.assertAlmostEqual(
                correct_metadata_dtype(key, string_input),
                20.2
            )

    def test_integers(self):

        float_input = 20.2
        for key in INTEGERS:
            self.assertEqual(
                correct_metadata_dtype(key, float_input),
                20
            )

    def test_strings(self):
        float_input = 20.2
        for key in STRINGS:
            self.assertEqual(
                correct_metadata_dtype(key, float_input),
                "20.2"
            )

    def test_no_dtype(self):

        typed_keywords = FLOATS + STRINGS + INTEGERS
        value = (1, 2, 3)
        for key in METADATA_KEYWORDS:
            if key not in typed_keywords:
                self.assertTupleEqual(
                    correct_metadata_dtype(key, value),
                    value
                )


class TestCheckMetadata(unittest.TestCase):

    def test_operational(self):

        self.assertTrue(
            check_metadata({
                'sample_phi_deg': -40,
                'energy_ev': 16100,
            })
        )

    def test_unaccepted_keyswords(self):

        with self.assertRaises(ValueError):
            check_metadata({'not a keyword': 2})

    def test_overspecified_source(self):
        """Check energy OR wavelength is provided, not both."""

        with self.assertRaises(ValueError):
            check_metadata({
                'energy_ev': 16100,
                'wavelength_nm': 1.2
            })

    def test_operational_sample(self):

        self.assertTrue(
            check_metadata({
                'q_peak_positions_Ang-1': [0.001],
            }, sample_mode=True)
        )

    def test_unaccepted_keyswords_sample(self):

        with self.assertRaises(ValueError):
            check_metadata({'sample_phi_deg': 2}, sample_mode=True)
