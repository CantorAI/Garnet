# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-2B contract over the shared config-driven Qwen3.5 graph.

Factories import the backend lazily so checkpoint validation needs no GPU.
CUDA execution is validated separately from this architecture contract.
"""

def validate_config(config):
    text,vision=config['text_config'],config['vision_config']
    expected={'hidden_size':2048,'num_hidden_layers':24,'intermediate_size':6144,
        'num_attention_heads':8,'num_key_value_heads':2,'head_dim':256,
        'linear_num_key_heads':16,'linear_num_value_heads':16,
        'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4}
    for key,value in expected.items():
        if text.get(key)!=value:
            raise ValueError('Qwen3.5-2B requires text '+key+'='+str(value))
    kinds=['linear_attention','linear_attention','linear_attention','full_attention']*6
    if text.get('layer_types')!=kinds:
        raise ValueError('Qwen3.5-2B requires the 24-layer hybrid attention layout')
    for key,value in {'depth':24,'hidden_size':1024,'num_heads':16,
        'out_hidden_size':2048,'patch_size':16,'temporal_patch_size':2,'spatial_merge_size':2}.items():
        if vision.get(key)!=value:
            raise ValueError('Qwen3.5-2B requires vision '+key+'='+str(value))
    return True

def create_decoder(checkpoint,config,capacity):
    validate_config(config)
    from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
    return MetalDecoder(checkpoint,config['text_config'],1,capacity)

def create_vision(checkpoint,config,gh,gw,shared_encoder=None):
    validate_config(config)
    from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
    return MetalImageEncoder(checkpoint,config['vision_config'],gh,gw,1,shared_encoder)

def create_cuda_decoder(checkpoint,config,capacity,batch=1,chunk_size=1,fused_delta=False,fused_attention=False,projection_workspace_bytes=0,retain_linear_inputs=False):
    validate_config(config)
    from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
    return CUDADecoder(checkpoint,config['text_config'],batch,capacity,chunk_size,fused_delta,fused_attention,projection_workspace_bytes,retain_linear_inputs)

def create_cuda_vision(checkpoint,config,gh,gw):
    validate_config(config)
    from xModel.qwen35.vl_0_8b.cuda_decoder import CUDAImageEncoder
    return CUDAImageEncoder(checkpoint,config['vision_config'],gh,gw,1)
