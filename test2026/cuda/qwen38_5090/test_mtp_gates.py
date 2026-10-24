"""CPU fixtures for the strict pre-benchmark MTP equality gate, not GPU evidence."""
import copy,unittest
from validate_native_mtp_http import compare_greedy_pair
class MTPGateTests(unittest.TestCase):
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
