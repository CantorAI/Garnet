import unittest

from resident_budget import validate_runtime_candidate_shape


class ResidentCandidateShapeTests(unittest.TestCase):
    def setUp(self):
        self.profile = {'candidate_shape': dict(batch=224,
            input_tokens_per_request=2005, output_tokens_per_request=1024,
            max_context_tokens_per_request=3072, prefill_chunk_tokens=36,
            prefill_engine_batch=112, decode_engine_batch=224)}
        self.arguments = dict(batch=224, input_tokens=2005, output_tokens=1024,
            context=3072, prefill_chunk=36,
            phase_engine_batches=dict(prefill=112, decode=224))

    def test_exact_profile_shape_is_admitted(self):
        validate_runtime_candidate_shape(self.profile, **self.arguments)

    def test_request_or_phase_shape_mismatch_is_rejected(self):
        for key, value in (('input_tokens', 2004), ('output_tokens', 1023),
                           ('context', 3071), ('prefill_chunk', 32),
                           ('batch', 112),
                           ('phase_engine_batches', dict(prefill=224, decode=224))):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs'):
                validate_runtime_candidate_shape(self.profile,
                    **(self.arguments | {key: value}))

    def test_reference_profiles_without_candidate_shape_keep_existing_path(self):
        validate_runtime_candidate_shape({}, **self.arguments)

    def test_malformed_candidate_shape_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'malformed'):
            validate_runtime_candidate_shape({'candidate_shape': []}, **self.arguments)


if __name__ == '__main__':
    unittest.main()
