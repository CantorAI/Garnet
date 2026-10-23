# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-4B model contract and explicit Metal/CUDA factories.

4B differs from 0.8B in both decoder and vision dimensions. Validate those
before allocating graphs so a wrong checkpoint fails with a useful error.
"""


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
    from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
    return MetalDecoder(checkpoint, config['text_config'], 1, capacity)


def create_vision(checkpoint, config, gh, gw, shared_encoder=None):
    validate_config(config)
    from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
    return MetalImageEncoder(checkpoint, config['vision_config'], gh, gw, 1, shared_encoder)

def create_cuda_decoder(checkpoint,config,capacity,batch=1,chunk_size=1,fused_delta=False,fused_attention=False,projection_workspace_bytes=0,retain_linear_inputs=False):
    validate_config(config)
    from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
    return CUDADecoder(checkpoint,config['text_config'],batch,capacity,chunk_size,fused_delta,fused_attention,projection_workspace_bytes,retain_linear_inputs)

def create_cuda_vision(checkpoint,config,gh,gw):
    validate_config(config)
    from xModel.qwen35.vl_0_8b.cuda_decoder import CUDAImageEncoder
    return CUDAImageEncoder(checkpoint,config['vision_config'],gh,gw,1)
