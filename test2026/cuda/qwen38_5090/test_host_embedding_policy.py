"""CPU constructor admission/lifecycle gates; not native embedding qualification."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

class BuildReached(RuntimeError):pass

class HostEmbeddingPolicyTests(unittest.TestCase):
    def load(self, failure=None):
        events=[]
        class IO:
            def open(self,checkpoint):events.append('open')
            def enable_nvfp4_head(self,value):
                events.append(('head',value))
                if failure=='head':raise RuntimeError('unsupported head')
            def enable_host_embedding(self,value):
                events.append(('host',value))
                if failure=='host':raise RuntimeError('mapping failed')
            def close(self):events.append('close')
        def program(*args):
            events.append('build')
            raise BuildReached('stop before actual engine work')
        def io():events.append('create_io');return IO()
        path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_packed_decoder.py'
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CUDAPackedDecoder')
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','close')]
        namespace={'garnet':SimpleNamespace(cuda_tensor_io=io),
            'validate_joined_mlp_profile':lambda *args:None,'CUDAPackedDecoderProgram':program}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),namespace)
        return namespace['CUDAPackedDecoder'],events
    def test_default_keeps_existing_gpu_policy(self):
        cls,events=self.load()
        with self.assertRaises(BuildReached):cls('fixture',{})
        self.assertEqual(events,['create_io','open','build','close'])
    def test_opt_in_orders_head_and_mapping_before_any_build(self):
        cls,events=self.load()
        with self.assertRaises(BuildReached):cls('fixture',{},host_embedding_lookup=True)
        self.assertEqual(events,['create_io','open',('head',True),('host',True),'build','close'])
    def test_native_admission_failures_close_io_without_build(self):
        for stage,message in [('head','unsupported head'),('host','mapping failed')]:
            cls,events=self.load(stage)
            with self.assertRaisesRegex(RuntimeError,message):cls('fixture',{},host_embedding_lookup=True)
            self.assertEqual(events[-1],'close');self.assertNotIn('build',events)
    def test_ambiguous_flag_rejected_before_open(self):
        cls,events=self.load()
        for value in (1,0,None,'true',[],{}):
            with self.assertRaises(ValueError):cls('fixture',{},host_embedding_lookup=value)
        self.assertEqual(events,[])

if __name__=='__main__':unittest.main()
