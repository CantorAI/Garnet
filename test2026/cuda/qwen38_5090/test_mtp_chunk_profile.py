"""CPU shape/control checks; actual GPU greedy equivalence remains required."""
import ast,json,types,unittest
from pathlib import Path
from test_speculative_events import production_method

path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_speculative.py'
tree=ast.parse(path.read_text(encoding='utf8'))
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='resolve_mtp_chunk')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
resolve=namespace['resolve_mtp_chunk']


class MTPChunkProfileTests(unittest.TestCase):
    def test_constructor_separates_target_norm_and_draft_profile(self):
        shapes=[];program_sizes=[]
        graph=types.SimpleNamespace(enable_cuda_graph=lambda value:None)
        def program(checkpoint,config,batch,capacity,size,workspace):
            program_sizes.append(size);return types.SimpleNamespace(graph=graph)
        io=types.SimpleNamespace(memory_json=lambda:json.dumps({'vocab_size':100}),
            zeros=lambda shape:shapes.append(shape) or object())
        decoder=types.SimpleNamespace(chunk_size=16,prefill_chunk_size=16,batch=1,capacity=32,io=io,
            config={'hidden_size':2,'num_key_value_heads':1,'head_dim':2},projection_workspace_bytes=0)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CUDASpeculativeSession')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        ns={'json':json,'resolve_mtp_chunk':resolve,'CUDAMTPProgram':program,
            'CUDATargetVerifier':lambda *args:object()}
        exec(compile(ast.Module(body=[init],type_ignores=[]),str(path),'exec'),ns)
        session=types.SimpleNamespace(close=lambda:None)
        ns['__init__'](session,decoder,'fixture',[99],3,draft_chunk_size=4)
        self.assertEqual(program_sizes,[4]);self.assertEqual(session.size,4)
        self.assertEqual(session.prefill_width,16);self.assertIsNone(session.prefill_norm)
        self.assertEqual(shapes[:2],[[1,4,2],[1,4,2]])

    def test_profile_bounds_and_default(self):
        self.assertEqual(resolve(16,3),16)
        self.assertEqual(resolve(16,3,4),4)
        self.assertEqual(resolve(16,7,8),8)
        for depth,size in [(3,1),(7,4),(3,32),(3,True),(3,4.0)]:
            with self.assertRaises(ValueError):resolve(16,depth,size)

    def test_wide_prefill_is_split_into_small_draft_chunks_with_shifted_positions(self):
        offsets=[];calls=[];writes=[]
        io=types.SimpleNamespace(slice_into=lambda source,offset,dst:offsets.append(offset),
            zero=lambda value:None,write_chunk_token=lambda value,dst,t:writes.append((value,t)))
        s=types.SimpleNamespace(_original_chunk=lambda *args:[9],_prefill_emit=None,_prefill_index=0,
            prefill_norm=None,decoder=types.SimpleNamespace(hidden=object()),
            verifier=types.SimpleNamespace(graph=types.SimpleNamespace(run_device=lambda inputs:object())),
            io=io,size=4,hidden_size=2,batch=1,target=object(),embedding=object(),
            _prefill_hidden=list(range(20)),_prefill_lengths=[20],_prefill_positions=[[[i]*3 for i in range(20)]],
            _call=lambda flags,positions,predict,stage:calls.append((flags,positions,predict,stage)))
        fn=production_method('_prefill_hook')
        self.assertEqual(fn(s,[0]*16,[[1]]*16,[[[i]*3 for i in range(16)]],False),[9])
        self.assertEqual(offsets,[0,8,16,24]);self.assertEqual(s._prefill_index,16)
        self.assertEqual(len(calls),4)
        self.assertEqual([p for c in calls for p in c[1][0]],[[i]*3 for i in range(1,17)])
        self.assertEqual([v for v,t in writes],list(range(1,17)))
        calls.clear();writes.clear()
        fn(s,[0]*4,[[1]]*4,[[[i]*3 for i in range(16,20)]],True)
        self.assertEqual(calls[0][0],[[1],[1],[1],[0]])
        self.assertEqual(writes,[(17,0),(18,1),(19,2)])
        self.assertEqual(s._prefill_index,20)


if __name__=='__main__':unittest.main()
