import json
import unittest

from engine_profile_shape import candidate_shape, validate_candidate_shape


class EngineProfileShapeTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(batch=224, input_tokens_per_request=2005,
            output_tokens_per_request=1024, max_context_tokens_per_request=3072,
            prefill_chunk_tokens=16, input_token_ids=list(range(2005)))
        self.raw = json.dumps(self.reference, separators=(',', ':')).encode()

    def test_defaults_keep_phase_engine_batches_equal_to_kv_pool(self):
        shape = candidate_shape(self.reference, self.raw)
        self.assertEqual(shape['batch'], 224)
        self.assertEqual(shape['prefill_engine_batch'], 224)
        self.assertEqual(shape['decode_engine_batch'], 224)
        self.assertEqual(validate_candidate_shape(shape, self.reference, self.raw), shape)

    def test_separates_half_batch_prefill_from_full_kv_and_decode_capacity(self):
        shape = candidate_shape(self.reference, self.raw, prefill_chunk=36,
            prefill_engine_batch=112, decode_engine_batch=224)
        self.assertEqual(shape['batch'], 224)
        self.assertEqual(shape['prefill_engine_batch'] * shape['prefill_chunk_tokens'], 4032)
        self.assertEqual(shape['decode_engine_batch'], 224)
        self.assertEqual(validate_candidate_shape(shape, self.reference, self.raw), shape)

    def test_rejects_phase_batch_larger_than_shared_pool_or_tampered_shape(self):
        with self.assertRaisesRegex(ValueError, 'prefill_engine_batch'):
            candidate_shape(self.reference, self.raw, prefill_engine_batch=225)
        shape = candidate_shape(self.reference, self.raw, prefill_engine_batch=112,
            decode_engine_batch=224)
        shape['prefill_engine_batch'] = 225
        with self.assertRaisesRegex(ValueError, 'prefill_engine_batch'):
            validate_candidate_shape(shape, self.reference, self.raw)


if __name__ == '__main__':
    unittest.main()
