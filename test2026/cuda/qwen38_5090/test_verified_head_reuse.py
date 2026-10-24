"""CPU transaction/cache controls; no simulated GPU arithmetic or benchmark."""
import ast,types,unittest
from pathlib import Path

def method(file,cls,name):
    path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b'/file
    tree=ast.parse(path.read_text(encoding='utf8'))
    node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace);return namespace[name]

class Fixture:
    commit_prefix=method('cuda_packed_replay.py','CUDAPackedDeltaReplayVerifier','commit_prefix')
    _read=method('cuda_verify.py','CUDATargetVerifier','_read')
    def __init__(self,submitted,reuse=True):
        b=len(submitted);self.pending=True;self.submitted=submitted;self.saved_slots=[5]*b;self.deferred=False;self.native_prefix=None
        self.reuse_verified_head=reuse;self.max_tokens=4;self.head_evaluations=0;self.head_reuses=0;self._reuse_accepted_head=False
        self.decoder=types.SimpleNamespace(batch=b,chunk_size=4,active=object(),slot_tensor=object(),state=object(),hidden=object(),last_hidden=object(),last_packed=object(),slots=[9]*b)
        self.backups=[object() for _ in range(6)];self.cache_spans=object();self.head_input=object();self.predictions=list(range(10,10+b*4));self.calls=[]
        self.graph=types.SimpleNamespace(run_device=lambda args:self.calls.append('norm') or object())
        self.replay_graph=types.SimpleNamespace(run_device=lambda args:self.calls.append('replay') or object())
        def predict(value):self.calls.append('head');return [100+i for i in range(b*4)]
        self.io=types.SimpleNamespace(update=lambda *args:None,mask_chunk=lambda *args:None,copy_into=lambda *args:None,copy_last_chunk=lambda *args:None,slice_into=lambda *args:None,predict_tokens=predict)
    def rollback(self):self.pending=False;self.calls.append('rollback')

class VerifiedHeadReuseTests(unittest.TestCase):
    def test_partial_prefix_preserves_authoritative_ids_and_refreshes_norm(self):
        f=Fixture([4]);self.assertEqual(f.commit_prefix([2]),[11])
        self.assertEqual(f.calls,['replay','norm']);self.assertEqual((f.head_evaluations,f.head_reuses),(0,1));self.assertFalse(f.pending)
    def test_zero_active_prefix_forces_new_head_and_never_uses_old_prediction(self):
        f=Fixture([4,4]);self.assertEqual(f.commit_prefix([2,0]),[101,None]);self.assertIn('head',f.calls);self.assertEqual(f.head_reuses,0)
    def test_inactive_empty_row_does_not_disable_valid_reuse(self):
        f=Fixture([4,0]);self.assertEqual(f.commit_prefix([2,0]),[11,None]);self.assertNotIn('head',f.calls)
    def test_default_path_recomputes_and_missing_cache_rolls_back(self):
        f=Fixture([4],False);self.assertEqual(f.commit_prefix([2]),[101]);self.assertIn('head',f.calls)
        f=Fixture([4]);f.predictions=[]
        with self.assertRaises(ValueError):f.commit_prefix([2])
        self.assertIn('rollback',f.calls);self.assertFalse(f._reuse_accepted_head)
    def test_full_or_invalid_prefix_never_triggers_partial_replay(self):
        f=Fixture([4]);self.assertEqual(f.commit_prefix([4]),[13]);self.assertEqual(f.calls,[])
        f=Fixture([4])
        with self.assertRaises(ValueError):f.commit_prefix([5])
        self.assertTrue(f.pending);self.assertEqual(f.calls,[])

if __name__=='__main__':unittest.main()
