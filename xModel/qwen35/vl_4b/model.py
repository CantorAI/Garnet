# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-4B Metal model contract using the shared Qwen3.5 graph engine.

4B differs from 0.8B in both decoder and vision dimensions. Validate those
before allocating graphs so a wrong checkpoint fails with a useful error.
"""
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder


def validate_config(config):
    text, vision = config['text_config'], config['vision_config']
    expected = {'hidden_size': 2560, 'num_hidden_layers': 32, 'intermediate_size': 9216,
                'num_attention_heads': 16, 'num_key_value_heads': 4,
                'linear_num_key_heads': 16, 'linear_num_value_heads': 32}
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError('Qwen3.5-4B requires text '+key+'='+str(value))
    for key, value in {'depth': 24, 'hidden_size': 1024, 'num_heads': 16, 'out_hidden_size': 2560}.items():
        if vision.get(key) != value:
            raise ValueError('Qwen3.5-4B requires vision '+key+'='+str(value))
    return True


def create_decoder(checkpoint, config, capacity):
    validate_config(config)
    return MetalDecoder(checkpoint, config['text_config'], 1, capacity)


def create_vision(checkpoint, config, gh, gw):
    validate_config(config)
    return MetalImageEncoder(checkpoint, config['vision_config'], gh, gw, 1)
