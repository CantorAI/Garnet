import os
import unittest
from unittest.mock import patch

from pipeline import _dynamic_batch_profile


class DynamicBatchProfileTests(unittest.TestCase):
    def test_standard_and_hybrid_page_axes_normalize_to_request_batch(self):
        shapes = [
            [8, 32], [8, 32], [4, 1024, 16, 8, 64], [4, 1024, 16, 8, 64],
            [8, 256], [8], [8], [8], [4, 512, 16, 8, 64],
            [4, 512, 16, 8, 64], [8, 256],
        ]
        axes = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
        with patch.dict(os.environ, {
            'GARNET_GPT_OSS_BATCH_PROFILE_MIN': '2',
            'GARNET_GPT_OSS_BATCH_PROFILE_MAX': '16',
        }, clear=False):
            result = _dynamic_batch_profile(shapes, axes, 8)['input_shape_profiles']
        self.assertEqual(result['batch_units'], [1, 1, 128, 128, 1, 1, 1, 1, 64, 64, 1])
        for shape, axis, units, minimum, optimum, maximum in zip(
                shapes, axes, result['batch_units'], result['min'], result['opt'], result['max']):
            self.assertEqual(optimum, shape)
            self.assertEqual(minimum[axis] // units, 2)
            self.assertEqual(optimum[axis] // units, 8)
            self.assertEqual(maximum[axis] // units, 16)

    def test_profile_is_absent_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_dynamic_batch_profile([[8, 4]], [0], 8), {})

    def test_rejects_bad_bounds_and_axis_cardinality(self):
        with patch.dict(os.environ, {'GARNET_GPT_OSS_BATCH_PROFILE_MIN': '9'}):
            with self.assertRaises(ValueError):
                _dynamic_batch_profile([[8, 4]], [0], 8)
        with patch.dict(os.environ, {'GARNET_GPT_OSS_BATCH_PROFILE_MIN': '2'}):
            with self.assertRaises(ValueError):
                _dynamic_batch_profile([[8, 4]], [-2], 8)
            with self.assertRaises(ValueError):
                _dynamic_batch_profile([[8, 4]], [0, 1], 8)


if __name__ == '__main__':
    unittest.main()
