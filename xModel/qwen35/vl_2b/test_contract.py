# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('qwen35_2b_contract',Path(__file__).with_name('model.py'))
contract=importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)

def config():
    return {'text_config':{'hidden_size':2048,'num_hidden_layers':24,'intermediate_size':6144,
        'num_attention_heads':8,'num_key_value_heads':2,'head_dim':256,
        'linear_num_key_heads':16,'linear_num_value_heads':16,'linear_key_head_dim':128,
        'linear_value_head_dim':128,'linear_conv_kernel_dim':4,
        'layer_types':['linear_attention']*3+['full_attention']},
        'vision_config':{'depth':24,'hidden_size':1024,'num_heads':16,
            'out_hidden_size':2048,'patch_size':16,'temporal_patch_size':2,'spatial_merge_size':2}}

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.c=config();self.c['text_config']['layer_types']*=6
    def test_validation_does_not_import_a_gpu_backend(self):
        self.assertTrue(contract.validate_config(self.c))
    def test_rejects_4b_grouped_value_heads_before_allocation(self):
        self.c['text_config']['linear_num_value_heads']=32
        with self.assertRaisesRegex(ValueError,'linear_num_value_heads'):
            contract.create_decoder('unused',self.c,256)
    def test_rejects_wrong_attention_layout(self):
        self.c['text_config']['layer_types'][0]='full_attention'
        with self.assertRaisesRegex(ValueError,'hybrid attention layout'):
            contract.validate_config(self.c)
    def test_rejects_wrong_vision_projection(self):
        self.c['vision_config']['out_hidden_size']=2560
        with self.assertRaisesRegex(ValueError,'out_hidden_size'):
            contract.create_vision('unused',self.c,16,16)

if __name__=='__main__':unittest.main()
