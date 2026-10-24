import copy,json,unittest
from pathlib import Path
from xModel.qwen38.vl_27b.model import validate_cuda_int4_profile
class CTAdmissionTests(unittest.TestCase):
 def setUp(self):self.config=json.loads((Path(__file__).parent/'evidence/int4-config.json').read_text())
 def test_actual_pinned_recipe_and_bounds(self):
  self.assertTrue(validate_cuda_int4_profile(self.config,4096))
  for capacity,batch in [(4097,1),(31,1),(4096,2)]:
   with self.assertRaises(ValueError):validate_cuda_int4_profile(self.config,capacity,batch)
 def test_rejects_different_quantization_arithmetic(self):
  for field,value in [('num_bits',8),('group_size',64),('symmetric',False),('dynamic',True),('type','float'),('strategy','tensor')]:
   config=copy.deepcopy(self.config);config['quantization_config']['config_groups']['group_0']['weights'][field]=value
   with self.assertRaises(ValueError):validate_cuda_int4_profile(config,4096)
  config=copy.deepcopy(self.config);config['quantization_config']['config_groups']['group_0']['input_activations']={'num_bits':8}
  with self.assertRaises(ValueError):validate_cuda_int4_profile(config,4096)
if __name__=='__main__':unittest.main()
