"""CPU fixtures for the strict pre-benchmark MTP equality gate, not GPU evidence."""
import copy,unittest
from validate_native_mtp_http import compare_greedy_pair,validate_worker_profile
class MTPGateTests(unittest.TestCase):
 def test_actual_profile_required_before_candidate_timing(self):
  h={'native_worker':{'mtp_depth':3,'mtp_chunk':4,'vision_input_policy':'direct'}}
  self.assertEqual(validate_worker_profile(h,3,4,'direct'),h['native_worker'])
  for depth,chunk,vision in [(7,4,'direct'),(3,16,'direct'),(3,4,'resident_copy')]:
   with self.assertRaises(ValueError):validate_worker_profile(h,depth,chunk,vision)
  with self.assertRaises(ValueError):validate_worker_profile({'native_worker':{'mtp_depth':3}},3,4)
  with self.assertRaises(ValueError):validate_worker_profile({})
 def test_shared_profile_missing_or_stale_fields_rejected(self):
  worker={'mtp_depth':3,'mtp_chunk':4,'vision_input_policy':'direct','chunk':4,'prefill_chunk':128,'shared_compressed_weights':True}
  validate_worker_profile({'native_worker':worker},3,4,'direct',4,128,True)
  for key in ('chunk','prefill_chunk','shared_compressed_weights'):
   for value in (None,False,16):
    bad=dict(worker);bad[key]=value
    with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_worker_profile({'native_worker':bad},3,4,'direct',4,128,True)
 def test_matching_ids_require_actual_mtp_mode_and_metrics(self):
  teacher={'choices':[{'prompt_token_ids':[1,2],'response_token_ids':[10,99]}],'study':{'generation_mode':'greedy_target'}}
  trial=copy.deepcopy(teacher);trial['study']={'generation_mode':'verified_mtp','speculative_metrics':{'target_verification_calls':1}}
  self.assertTrue(compare_greedy_pair('fixture',teacher,trial)['passed'])
  for change in ('token','prompt','mode','metrics'):
   bad=copy.deepcopy(trial)
   if change=='token':bad['choices'][0]['response_token_ids']=[10,20,99]
   elif change=='prompt':bad['choices'][0]['prompt_token_ids']=[1,3]
   elif change=='mode':bad['study']['generation_mode']='greedy_target'
   else:bad['study']['speculative_metrics']=None
   self.assertFalse(compare_greedy_pair('fixture',teacher,bad)['passed'])
if __name__=='__main__':unittest.main()
