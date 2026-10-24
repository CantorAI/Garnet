"""CPU builder-policy/lifecycle test; no numerical or GPU qualification claim."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

class WorkspaceTests(unittest.TestCase):
    def load(self, fail=False):
        graphs=[]
        class Graph:
            def __init__(self):self.workspace=[];self.closed=False;graphs.append(self)
            def set_builder_workspace_bytes(self,value):
                self.workspace.append(value)
                if fail:raise RuntimeError('setter failure')
            def build(self,*args):self.built=True
            def close(self):self.closed=True
        source=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_mtp.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CUDAMTPProgram')
        namespace={'garnet':SimpleNamespace(compiled_graph=Graph),
            'tensor':SimpleNamespace(input=lambda *args,**kwargs:None,float32='f32'),
            'Qwen35MTPChunk':lambda *args:None}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
        return namespace['CUDAMTPProgram'],graphs
    def make(self,klass,**kwargs):
        return klass('fixture',{'hidden_size':8,'num_key_value_heads':1,'head_dim':4},**kwargs)
    def test_default_does_not_override_existing_builder_policy(self):
        klass,graphs=self.load();program=self.make(klass)
        self.assertEqual(graphs[0].workspace,[]);program.close();self.assertTrue(graphs[0].closed)
    def test_explicit_bounded_opt_in_and_rejects(self):
        klass,graphs=self.load();program=self.make(klass,builder_workspace_bytes=128<<20)
        self.assertEqual(graphs[0].workspace,[128<<20]);program.close()
        for value in (True,None,64<<20,256<<20,-1):
            with self.assertRaises(ValueError):self.make(klass,builder_workspace_bytes=value)
        self.assertEqual(len(graphs),1)
    def test_setter_failure_retires_graph(self):
        klass,graphs=self.load(fail=True)
        with self.assertRaisesRegex(RuntimeError,'setter failure'):self.make(klass,builder_workspace_bytes=128<<20)
        self.assertTrue(graphs[0].closed)

if __name__=='__main__':unittest.main()
