"""Actual Model control flow with deterministic CPU decoder; no GPU speed claim."""
import ast,hashlib,threading,time,types,unittest
from pathlib import Path

def fixture():
 path=Path(__file__).resolve().parents[3]/'runtime/cuda_serving.py'
 tree=ast.parse(path.read_text(encoding='utf8'))
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
 methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('infer_batch','infer')]
 scope={'time':time,'hashlib':hashlib,'__file__':str(path)}
 exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),scope)
 class Model:pass
 for name in ('infer_batch','infer'):setattr(Model,name,scope[name])
 m=Model();m.lock=threading.RLock();m.batch=2;m.max_output=4;m.speculative=None;m.pack_prompt=False
 m.config={'text_config':{'hidden_size':1}};m.stops=[99];m.name='fixture';m.profile={}
 m.prefill_chunk=16;m.decode_chunk=16;m.shared_projection=False;m.packed_program=False
 for name in ('resident_controls','register_delta_128','flat_packed_concat','serial_target_graph_reuse','deferred_state_commit','native_prefix_commit','shared_output_storage','reuse_verified_head','serial_graph_reuse','speculative_replay'):setattr(m,name,False)
 m.speculative_depth=0;m.speculative_candidates=0
 m.tokenizer=types.SimpleNamespace(decode=lambda ids:','.join(map(str,ids)))
 m._prepare=lambda prompt,image:([1,2],[[0,0,0],[1,1,1]],[11,12],2,0.,0,None,-1)
 trace=[];predictions=iter(([20,99],[99,99]))
 def step(*args):trace.append('step');return next(predictions)
 m.decoder=types.SimpleNamespace(reset=lambda:None,prefill_sequence=lambda *a:[10,30],step=step,io=types.SimpleNamespace(zeros=lambda *a:0,stack_rows=lambda x:x))
 return m,trace

class ModelTokenEventsTests(unittest.TestCase):
 def test_live_events_precede_next_decode_and_stop_once(self):
  m,trace=fixture();events=[]
  def callback(*event):events.append(event);trace.append(event)
  r=m.infer_batch(['a','b'],['',''],4,on_token=callback)
  self.assertEqual(events,[(0,10,1,2),(1,30,1,2),(0,20,2,2),(1,99,2,2),(0,99,3,2)])
  self.assertEqual(trace[:3],[events[0],events[1],'step'])
  self.assertEqual(r['generated_token_ids'],[[10,20],[30]])
  self.assertEqual(r['sampled_tokens_including_stop'],[3,2]);self.assertEqual(r['finish_reasons'],['eos','eos'])
 def test_default_results_and_length_boundary(self):
  m,_=fixture();r=m.infer_batch(['a','b'],['',''],1)
  self.assertEqual(r['generated_token_ids'],[[10],[30]]);self.assertEqual(r['decode_steps'],0)
  self.assertEqual(r['finish_reasons'],['length','length'])
 def test_callback_failure_propagates_without_fabricated_completion(self):
  m,trace=fixture()
  def fail(*args):raise RuntimeError('consumer failed')
  with self.assertRaisesRegex(RuntimeError,'consumer failed'):m.infer_batch(['a','b'],['',''],4,on_token=fail)
  self.assertEqual(trace,[])
 def test_unsupported_callbacks_fail_before_decoder(self):
  m,_=fixture()
  with self.assertRaises(TypeError):m.infer_batch(['a','b'],['',''],1,on_token=True)
  m.speculative=object()
  with self.assertRaisesRegex(ValueError,'target-only'):m.infer_batch(['a','b'],['',''],1,on_token=lambda *a:None)
if __name__=='__main__':unittest.main()
