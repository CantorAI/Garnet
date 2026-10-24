"""Exercise the production MTP control method with deterministic verifier fixtures.

AST-loading isolates its CPU control flow from unavailable native GPU imports;
no model/kernel implementation or performance result is simulated here.
"""
import ast,types,unittest
from pathlib import Path

def production_method(name):
 path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_speculative.py'
 tree=ast.parse(path.read_text(encoding='utf-8'));cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CUDASpeculativeSession')
 method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)
 module=ast.Module(body=[method],type_ignores=[]);namespace={};exec(compile(module,str(path),'exec'),namespace)
 return namespace[name]

class Verifier:
 def __init__(self,session,predictions):self.session=session;self.predictions=predictions;self.pending=False;self.head_evaluations=0;self.head_reuses=0
 def verify(self,proposals,flags,positions):self.submitted=[sum(r[0] for r in flags)];self.pending=True;self.head_evaluations+=1;self.session.timeline.append('verified');return [self.predictions]
 def commit_prefix(self,counts):self.session.decoder.slots=[self.session.decoder.slots[0]+counts[0]];self.pending=False;return [self.predictions[counts[0]-1]]
 def rollback(self):self.pending=False;self.session.rolled_back=True

class Fixture:
 generate=production_method('generate')
 def __init__(self,proposals,predictions):
  self.ready=True;self.finished=False;self.poisoned=False;self.capacity=4096;self.batch=1;self.stops={99};self.current=[10];self.current_positions=[5];self.decoder=types.SimpleNamespace(slots=[5])
  self.proposals=proposals;self.timeline=[];self.rolled_back=False;self.slots=[5];self.head_candidates=0;self.prefix_replay=True
  self.metrics={'target_verification_calls':0,'accepted_draft_tokens':0,'target_prefix_replays':0,'target_prefix_recomputes':0,'iterations':[]}
  self.verifier=Verifier(self,predictions)
 def _draft(self,*args):return [[t] for t in self.proposals]
 def _correct(self,*args):self.timeline.append('correction');self.slots=list(self.decoder.slots);return [0]

class SpeculativeEventTests(unittest.TestCase):
 def test_prefill_first_token_is_visible_before_mtp_work_without_mutating_target(self):
  timeline=[];target=[10]
  def original(*args):timeline.append('target');return target
  def emit(value):timeline.append('first');value[0]=999
  graph=types.SimpleNamespace(run_device=lambda inputs:timeline.append('norm') or object())
  io=types.SimpleNamespace(slice_into=lambda *args:None,zero=lambda *args:None,write_chunk_token=lambda *args:None)
  session=types.SimpleNamespace(_original_chunk=original,_prefill_emit=emit,_prefill_index=0,prefill_norm=graph,decoder=types.SimpleNamespace(hidden=object()),io=io,size=1,hidden_size=4,batch=1,target=object(),embedding=object(),_prefill_hidden=[object(),object()],_prefill_lengths=[2],_prefill_positions=[[[0]*3,[1]*3]],_call=lambda *args:timeline.append('mtp'))
  result=production_method('_prefill_hook')(session,[object()],[[1]],[[[0]*3]],True)
  self.assertEqual(result,[10]);self.assertEqual(timeline,['target','first','norm','mtp'])
 def test_rejected_draft_never_emitted_and_accepted_tokens_arrive_before_correction(self):
  session=Fixture([10,20,77],[20,30,40]);events=[]
  def emit(row,token,count):events.append((row,token,count));session.timeline.append('emit'+str(token))
  result=session.generate(3,emit)
  self.assertEqual(result['generated_token_ids'],[[10,20,30]]);self.assertEqual(events,[(0,10,1),(0,20,2),(0,30,3)])
  self.assertLess(session.timeline.index('verified'),session.timeline.index('emit20'));self.assertLess(session.timeline.index('emit20'),session.timeline.index('correction'))
  self.assertNotIn('emit77',session.timeline)
 def test_verified_eos_counted_once_and_excluded_from_text_ids(self):
  session=Fixture([10,99],[99,0]);events=[];result=session.generate(8,lambda row,token,count:events.append((token,count)))
  self.assertEqual(events,[(10,1),(99,2)]);self.assertEqual(result['generated_token_ids'],[[10]])
  self.assertEqual(result['sampled_tokens_including_stop'],[2]);self.assertEqual(result['stop_token_ids'],[99])
 def test_callback_failure_poison_and_pending_state_rollback(self):
  session=Fixture([10,20],[20,30])
  def fail(*args):raise OSError('fixture closed stream')
  with self.assertRaises(OSError):session.generate(4,fail)
  self.assertTrue(session.poisoned);self.assertTrue(session.rolled_back);self.assertFalse(session.verifier.pending)
 def test_legacy_no_callback_keeps_same_greedy_output(self):
  self.assertEqual(Fixture([10,20,77],[20,30,40]).generate(3)['generated_token_ids'],[[10,20,30]])
if __name__=='__main__':unittest.main()
