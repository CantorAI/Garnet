import copy,json,unittest
from pathlib import Path
from xModel.qwen38.vl_27b.model import validate_cuda_profile,validate_cuda_batch_study_profile
class BatchStudyProfileTests(unittest.TestCase):
 def setUp(self):
  self.config=json.loads((Path(__file__).parent/'evidence/base-config.json').read_text(encoding='utf-8-sig'))
  self.config['quantization_config']={'quant_method':'modelopt','quant_algo':'MIXED_PRECISION'}
 def test_research_opt_in_does_not_open_general_batch_admission(self):
  for b in (2,4):
   self.assertTrue(validate_cuda_batch_study_profile(self.config,4096,b))
   with self.assertRaises(ValueError):validate_cuda_profile(self.config,4096,b)
 def test_rejects_unqualified_shapes_recipe_and_model(self):
  for b in (0,1,3,8,True,2.0):
   with self.assertRaises(ValueError):validate_cuda_batch_study_profile(self.config,4096,b)
  for cap in (0,4097,4096.0,True):
   with self.assertRaises(ValueError):validate_cuda_batch_study_profile(self.config,cap,2)
  for field,value in [('quantization_config',{}),('text_config',{})]:
   config=copy.deepcopy(self.config);config[field]=value
   with self.assertRaises((ValueError,KeyError)):validate_cuda_batch_study_profile(config,4096,2)
if __name__=='__main__':unittest.main()
