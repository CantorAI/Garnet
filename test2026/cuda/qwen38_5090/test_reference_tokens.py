"""Reference API layout/invalid-ID fixtures, not actual model outputs."""
import unittest
from prepare_vlm_reference import reference_tokens
class ReferenceTokenLayoutTests(unittest.TestCase):
 def test_installed_engine_fields_are_distinct_and_explicit(self):
  sg={'choices':[{'prompt_token_ids':[1,248056],'response_token_ids':[5844,248046]}]}
  vl={'prompt_token_ids':[1,248056],'choices':[{'token_ids':[5844,248046]}]}
  self.assertEqual(reference_tokens(sg,'sglang'),reference_tokens(vl,'vllm'))
  with self.assertRaises(ValueError):reference_tokens(sg,'vllm')
  with self.assertRaises(ValueError):reference_tokens(vl,'sglang')
 def test_no_missing_ids_boolean_or_text_inference(self):
  for values in (None,[],[True],[-1],[248320]):
   with self.assertRaises(ValueError):reference_tokens({'prompt_token_ids':[1],'choices':[{'token_ids':values,'message':{'content':'Red'}}]},'vllm')
  with self.assertRaises(ValueError):reference_tokens({'prompt_token_ids':[1]*4096,'choices':[{'token_ids':[1]}]},'vllm')
if __name__=='__main__':unittest.main()
