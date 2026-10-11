import os
import ast
import unittest
from unittest.mock import patch
from pathlib import Path

from pipeline import _dynamic_batch_profile, shared_kv_handoff


class DynamicBatchProfileTests(unittest.TestCase):
    def test_batch_runner_uses_full_handoff_at_shape_and_decode_boundaries(self):
        runner = Path(__file__).with_name('run_tp_batch_throughput.py')
        tree = ast.parse(runner.read_text(encoding='utf-8'))
        handoff_assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'kv'
                    for target in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == 'shared_kv_handoff'
        ]
        self.assertEqual(len(handoff_assignments), 2)
        self.assertFalse(any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'kv'
                    for target in node.targets)
            and isinstance(node.value, ast.ListComp)
            for node in ast.walk(tree)
        ))

    def test_hybrid_kv_handoff_preserves_every_bank_and_table_reference(self):
        layout = {'extra_tensor_arguments': ['window_keys', 'window_values', 'window_table']}
        keys, values = object(), object()
        resources = {name: object() for name in layout['extra_tensor_arguments']}
        stage = {
            'keys': keys, 'values': values, 'shared_resources': resources,
            'shared_resource_layout': dict(layout),
            'extra_input_names': tuple(layout['extra_tensor_arguments']),
        }
        [handoff] = shared_kv_handoff([stage], layout)
        self.assertIs(handoff['keys'], keys)
        self.assertIs(handoff['values'], values)
        self.assertIs(handoff['shared_resources'], resources)
        self.assertIs(handoff['shared_resource_layout'], stage['shared_resource_layout'])
        self.assertIs(handoff['extra_input_names'], stage['extra_input_names'])

    def test_hybrid_kv_handoff_rejects_missing_or_mismatched_resources(self):
        layout = {'extra_tensor_arguments': ['window_keys', 'window_values', 'window_table']}
        stage = {
            'keys': object(), 'values': object(),
            'shared_resources': {'window_keys': object(), 'window_values': object()},
            'shared_resource_layout': layout,
            'extra_input_names': tuple(layout['extra_tensor_arguments']),
        }
        with self.assertRaisesRegex(ValueError, 'every bank/table'):
            shared_kv_handoff([stage], layout)
        stage['shared_resources']['window_table'] = object()
        stage['shared_resource_layout'] = {'extra_tensor_arguments': []}
        with self.assertRaisesRegex(ValueError, 'every bank/table'):
            shared_kv_handoff([stage], layout)

    def test_non_hybrid_handoff_keeps_legacy_key_value_tuples(self):
        keys, values = object(), object()
        self.assertEqual(shared_kv_handoff([{'keys': keys, 'values': values}], None),
                         [(keys, values)])

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
