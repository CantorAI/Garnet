# SPDX-License-Identifier: Apache-2.0
"""Configuration validation runs without loading GPU weights."""
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib.util

class ContractTests(unittest.TestCase):
    def test_4b_dimensions_are_distinct_from_08b(self):
        path=Path(__file__).with_name('model.py')
        fake=types.ModuleType('fake');fake.MetalDecoder=object;fake.MetalImageEncoder=object
        with patch.dict(sys.modules,{'xModel.qwen35.vl_0_8b.metal_decoder':fake,'xModel.qwen35.vl_0_8b.metal_vision':fake}):
            spec=importlib.util.spec_from_file_location('contract',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config={'text_config':{'hidden_size':2560,'num_hidden_layers':32,'intermediate_size':9216,'num_attention_heads':16,'num_key_value_heads':4,'linear_num_key_heads':16,'linear_num_value_heads':32},'vision_config':{'depth':24,'hidden_size':1024,'num_heads':16,'out_hidden_size':2560}}
        self.assertTrue(module.validate_config(config))
        config['text_config']['linear_num_value_heads']=16
        with self.assertRaisesRegex(ValueError,'linear_num_value_heads'):module.validate_config(config)

if __name__=='__main__':unittest.main()
