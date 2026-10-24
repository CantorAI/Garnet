"""Validate draft tensor contract against pinned checkpoint headers, not GPU math."""
import copy,json,unittest
from pathlib import Path
from xModel.qwen38.vl_27b.model import mtp_shapes,validate_mtp_metadata
class MTPLayoutTests(unittest.TestCase):
 def setUp(self):
  root=Path(__file__).parent/'evidence';self.config=json.loads((root/'nvfp4-config.json').read_bytes());self.metadata={}
  for path in (root/'nvfp4-headers').glob('*.header.json'):
   self.metadata.update({k:v for k,v in json.loads(path.read_bytes()).items() if k.startswith('mtp.')})
 def test_all_fifteen_pinned_bf16_tensors_match(self):
  self.assertEqual(len(mtp_shapes(self.config)),15);self.assertEqual(set(mtp_shapes(self.config)),set(self.metadata));self.assertTrue(validate_mtp_metadata(self.config,self.metadata))
 def test_wrong_dtype_shape_or_layer_count_rejected(self):
  for field,value in [('dtype','U8'),('shape',[5120,5120])]:
   changed=copy.deepcopy(self.metadata);changed['mtp.fc.weight'][field]=value
   with self.assertRaises(ValueError):validate_mtp_metadata(self.config,changed)
  config=copy.deepcopy(self.config);config['text_config']['mtp_num_hidden_layers']=2
  with self.assertRaises(ValueError):mtp_shapes(config)
if __name__=='__main__':unittest.main()
