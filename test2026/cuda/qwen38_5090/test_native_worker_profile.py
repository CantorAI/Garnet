import unittest
from test2026.cuda.qwen38_5090.native_worker_profile import resolve_profile

class WorkerProfileTests(unittest.TestCase):
    def test_legacy_defaults(self):
        self.assertEqual(resolve_profile({})['decode_chunk'],16)
        self.assertEqual(resolve_profile({'GARNET_STUDY_PACKED':'1','GARNET_STUDY_MTP_DEPTH':'7'})['draft_chunk'],16)

    def test_shared_mtp3(self):
        p=resolve_profile({'GARNET_STUDY_PACKED':'1','GARNET_STUDY_SHARED_COMPRESSED':'1','GARNET_STUDY_MTP_DEPTH':'3'})
        self.assertEqual((p['prefill_chunk'],p['decode_chunk'],p['draft_chunk']),(128,4,4))

    def test_invalid_combinations(self):
        base={'GARNET_STUDY_PACKED':'1','GARNET_STUDY_SHARED_COMPRESSED':'1','GARNET_STUDY_MTP_DEPTH':'3'}
        for override in ({'GARNET_STUDY_PACKED':'0'},{'GARNET_STUDY_MTP_DEPTH':'7'},{'GARNET_STUDY_MTP_CHUNK':'16'},{'GARNET_STUDY_RESIDENT_VISION':'yes'}):
            with self.subTest(override=override),self.assertRaises(ValueError):resolve_profile(dict(base,**override))

if __name__=='__main__':unittest.main()
