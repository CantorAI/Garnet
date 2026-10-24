import unittest
from pathlib import Path
from test2026.cuda.qwen38_5090.native_worker_profile import resolve_profile

class WorkerProfileTests(unittest.TestCase):
    def test_launchers_pin_shared_policy(self):
        root=Path(__file__).parent
        for path in root.glob('serve_native_mtp*_worker.bat'):
            values=dict(line[4:].split('=',1) for line in path.read_text().splitlines() if line.startswith('set GARNET_STUDY_'))
            with self.subTest(launcher=path.name):
                self.assertIn('GARNET_STUDY_SHARED_COMPRESSED',values)
                profile=resolve_profile(values)
                self.assertIn('GARNET_STUDY_REUSE_VERIFIED_HEAD',values)
                self.assertEqual(profile['shared'],path.name in ('serve_native_mtp3_shared_worker.bat','serve_native_mtp3_shared_headreuse_worker.bat','serve_native_mtp3_shared_joined_worker.bat'))
                self.assertIn('GARNET_STUDY_JOINED_MLP',values)
                self.assertEqual(profile['joined_mlp'],path.name=='serve_native_mtp3_shared_joined_worker.bat')
                self.assertEqual(profile['reuse_head'],path.name=='serve_native_mtp3_shared_headreuse_worker.bat')

    def test_joined_mlp_requires_the_explicit_single_variable_profile(self):
        base={'GARNET_STUDY_PACKED':'1','GARNET_STUDY_SHARED_COMPRESSED':'1','GARNET_STUDY_MTP_DEPTH':'3'}
        self.assertFalse(resolve_profile(base)['joined_mlp'])
        self.assertTrue(resolve_profile(dict(base,GARNET_STUDY_JOINED_MLP='1'))['joined_mlp'])
        for override in ({'GARNET_STUDY_MTP_DEPTH':'0'},{'GARNET_STUDY_MTP_DEPTH':'7'},{'GARNET_STUDY_SHARED_COMPRESSED':'0'},{'GARNET_STUDY_REUSE_VERIFIED_HEAD':'1'},{'GARNET_STUDY_JOINED_MLP':'yes'}):
            values=dict(base,GARNET_STUDY_JOINED_MLP='1');values.update(override)
            with self.subTest(override=override),self.assertRaises(ValueError):resolve_profile(values)

    def test_legacy_defaults(self):
        self.assertEqual(resolve_profile({})['decode_chunk'],16)
        self.assertEqual(resolve_profile({'GARNET_STUDY_PACKED':'1','GARNET_STUDY_MTP_DEPTH':'7'})['draft_chunk'],16)

    def test_shared_mtp3(self):
        p=resolve_profile({'GARNET_STUDY_PACKED':'1','GARNET_STUDY_SHARED_COMPRESSED':'1','GARNET_STUDY_MTP_DEPTH':'3'})
        self.assertEqual((p['prefill_chunk'],p['decode_chunk'],p['draft_chunk']),(128,4,4))

    def test_head_reuse_is_explicit_and_bounded_to_qualified_profile(self):
        base={'GARNET_STUDY_PACKED':'1','GARNET_STUDY_SHARED_COMPRESSED':'1','GARNET_STUDY_MTP_DEPTH':'3'}
        self.assertFalse(resolve_profile(base)['reuse_head'])
        self.assertTrue(resolve_profile(dict(base,GARNET_STUDY_REUSE_VERIFIED_HEAD='1'))['reuse_head'])
        for override in ({'GARNET_STUDY_MTP_DEPTH':'0'},{'GARNET_STUDY_MTP_DEPTH':'1'},{'GARNET_STUDY_SHARED_COMPRESSED':'0'},{'GARNET_STUDY_REUSE_VERIFIED_HEAD':'yes'}):
            with self.subTest(override=override),self.assertRaises(ValueError):resolve_profile(dict(base,**dict({'GARNET_STUDY_REUSE_VERIFIED_HEAD':'1'},**override)))

    def test_invalid_combinations(self):
        base={'GARNET_STUDY_PACKED':'1','GARNET_STUDY_SHARED_COMPRESSED':'1','GARNET_STUDY_MTP_DEPTH':'3'}
        for override in ({'GARNET_STUDY_PACKED':'0'},{'GARNET_STUDY_MTP_DEPTH':'7'},{'GARNET_STUDY_MTP_CHUNK':'16'},{'GARNET_STUDY_RESIDENT_VISION':'yes'}):
            with self.subTest(override=override),self.assertRaises(ValueError):resolve_profile(dict(base,**override))

if __name__=='__main__':unittest.main()
