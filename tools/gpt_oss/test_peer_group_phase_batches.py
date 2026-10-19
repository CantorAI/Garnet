import os
import unittest
from unittest.mock import patch

from peer_group_layout import peer_group_layout, validate_peer_group_layout


class PeerGroupPhaseBatchTests(unittest.TestCase):
    @patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP': '1'}, clear=False)
    def test_split_prefill_batch_preserves_pool_and_decode_capacity(self):
        layout = peer_group_layout(224, 36, 2880,
            prefill_batch=112, decode_batch=224)
        self.assertEqual(layout['phase_elements'], [112 * 36 * 2880, 224 * 2880])
        self.assertEqual(layout['prefill_batch'], 112)
        self.assertEqual(layout['decode_batch'], 224)
        self.assertEqual(layout['capacity_elements'], 112 * 36 * 2880)
        self.assertEqual(validate_peer_group_layout(layout, 224, 36, 2880),
                         layout['owned_bytes_per_rank'] + layout['mapped_host_bytes'])

    @patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP': '1'}, clear=False)
    def test_rejects_cohort_over_native_row_bound_and_tampering(self):
        with self.assertRaises(ValueError):
            peer_group_layout(224, 37, 2880, prefill_batch=112, decode_batch=224)
        layout = peer_group_layout(224, 36, 2880,
            prefill_batch=112, decode_batch=224)
        layout['prefill_batch'] = 113
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            validate_peer_group_layout(layout, 224, 36, 2880)

    @patch.dict(os.environ, {'GARNET_GPT_OSS_BF16_PEER_GROUP': '1'}, clear=False)
    def test_default_layout_remains_legacy_identical(self):
        old = peer_group_layout(32, 16, 2880)
        explicit = peer_group_layout(32, 16, 2880,
            prefill_batch=32, decode_batch=32)
        self.assertEqual(old, explicit)
        self.assertNotIn('prefill_batch', old)


if __name__ == '__main__':
    unittest.main()
