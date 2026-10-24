"""Production native notification control, without loading model/GPU modules."""
import ast,types,unittest
from pathlib import Path

def notification_fixture():
 path=Path(__file__).with_name('native_stream_worker.py');tree=ast.parse(path.read_text(encoding='utf-8'))
 method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='notify')
 generated=[];frames=[];state={'ids':generated,'text':''}
 scope={'generated':generated,'stream_state':state,'stops':[99],'ids':[1,2,3],'key':'fixture','time':types.SimpleNamespace(monotonic=lambda:1.),'tokenizer':types.SimpleNamespace(decode=lambda tokens:','.join(str(t) for t in tokens)),'emit':frames.append}
 exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
 return scope['notify'],generated,frames

class NativeMTPNotificationTests(unittest.TestCase):
 def test_first_prefill_token_repeated_by_generate_is_not_counted_twice(self):
  notify,generated,frames=notification_fixture();notify(0,10,1);notify(0,10,1);notify(0,20,2);notify(0,99,3)
  self.assertEqual(generated,[10,20,99]);self.assertEqual([f['sampled_tokens'] for f in frames],[1,2,3]);self.assertEqual(frames[-1]['text'],'10,20');self.assertTrue(frames[-1]['stop'])
 def test_wrong_duplicate_gap_or_row_rejected(self):
  notify,_,_=notification_fixture();notify(0,10,1)
  with self.assertRaises(AssertionError):notify(0,11,1)
  with self.assertRaises(AssertionError):notify(0,20,3)
  with self.assertRaises(AssertionError):notify(1,20,2)
if __name__=='__main__':unittest.main()
