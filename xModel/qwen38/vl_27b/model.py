# SPDX-License-Identifier: Apache-2.0
"""Exact Qwen3.8-27B VLM contract. Runtime admission requires separate GPU gates."""


def validate_config(config):
    if config.get('architectures') != ['Qwen3_5ForConditionalGeneration'] or config.get('model_type') != 'qwen3_5':
        raise ValueError('Qwen3.8-27B requires the Qwen3.5 conditional-generation architecture')
    if config.get('language_model_only', False):
        raise ValueError('Qwen3.8 VLM cannot use a language-only checkpoint')
    text, vision = config['text_config'], config['vision_config']
    expected = {'hidden_size':5120, 'num_hidden_layers':64, 'intermediate_size':17408,
                'num_attention_heads':24, 'num_key_value_heads':4, 'head_dim':256,
                'linear_num_key_heads':16, 'linear_num_value_heads':48,
                'linear_key_head_dim':128, 'linear_value_head_dim':128,
                'linear_conv_kernel_dim':4, 'vocab_size':248320,
                'tie_word_embeddings':False, 'attn_output_gate':True}
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError('Qwen3.8-27B text contract mismatch: '+key)
    for key, value in {'depth':27, 'hidden_size':1152, 'intermediate_size':4304,
                       'num_heads':16, 'out_hidden_size':5120, 'patch_size':16,
                       'spatial_merge_size':2, 'temporal_patch_size':2}.items():
        if vision.get(key) != value:
            raise ValueError('Qwen3.8-27B vision contract mismatch: '+key)
    if text.get('layer_types') != (['linear_attention']*3+['full_attention'])*16:
        raise ValueError('Qwen3.8-27B requires the verified 48 linear/16 full schedule')
    rope = text.get('rope_parameters', {})
    if rope.get('mrope_section') != [11,11,10] or not rope.get('mrope_interleaved'):
        raise ValueError('Qwen3.8-27B requires verified multimodal rotary positions')
    for key, value in {'image_token_id':248056, 'video_token_id':248057,
                       'vision_start_token_id':248053, 'vision_end_token_id':248054}.items():
        if config.get(key) != value:
            raise ValueError('Qwen3.8-27B token contract mismatch: '+key)
    return True


def projection_shapes(config):
    """Logical shapes, before quantized packing, for every text projection."""
    validate_config(config)
    text = config['text_config']
    hidden, middle = text['hidden_size'], text['intermediate_size']
    result = {'lm_head':(text['vocab_size'],hidden)}
    for i, kind in enumerate(text['layer_types']):
        prefix = 'model.language_model.layers.'+str(i)+'.'
        for name, shape in {'gate_proj':(middle,hidden),'up_proj':(middle,hidden),'down_proj':(hidden,middle)}.items():
            result[prefix+'mlp.'+name] = shape
        if kind == 'linear_attention':
            qk = text['linear_num_key_heads']*text['linear_key_head_dim']
            v = text['linear_num_value_heads']*text['linear_value_head_dim']
            shapes = {'in_proj_qkv':(2*qk+v,hidden), 'in_proj_z':(v,hidden),
                      'in_proj_a':(text['linear_num_value_heads'],hidden),
                      'in_proj_b':(text['linear_num_value_heads'],hidden), 'out_proj':(hidden,v)}
            family = 'linear_attn.'
        else:
            q = text['num_attention_heads']*text['head_dim']
            kv = text['num_key_value_heads']*text['head_dim']
            shapes = {'q_proj':(2*q,hidden),'k_proj':(kv,hidden),'v_proj':(kv,hidden),'o_proj':(hidden,q)}
            family = 'self_attn.'
        for name, shape in shapes.items(): result[prefix+family+name] = shape
    return result


def validate_cuda_profile(config,capacity,batch=1):
    """Initial native study admission; measured full-model gates remain separate."""
    validate_config(config)
    q=config.get('quantization_config',{})
    if q.get('quant_method')!='modelopt' or q.get('quant_algo')!='MIXED_PRECISION':
        raise ValueError('Initial RTX5090 profile requires verified ModelOpt mixed NVFP4/FP8 weights')
    if batch!=1 or capacity<32 or capacity>4096:
        raise ValueError('Initial Qwen3.8 CUDA profile requires batch1 and capacity32..4096')
    return True


def create_cuda_decoder(checkpoint,config,capacity,batch=1,chunk_size=1,fused_delta=False,fused_attention=False,projection_workspace_bytes=0,retain_linear_inputs=False):
    validate_cuda_profile(config,capacity,batch)
    from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
    decoder=CUDADecoder(checkpoint,config['text_config'],batch,capacity,chunk_size,fused_delta,fused_attention,projection_workspace_bytes,retain_linear_inputs)
    try:
        decoder.io.enable_nvfp4_head(True)
        return decoder
    except:
        decoder.close();raise


def create_cuda_vision(checkpoint,config,gh,gw):
    validate_cuda_profile(config,4096,1)
    from xModel.qwen35.vl_0_8b.cuda_decoder import CUDAImageEncoder
    return CUDAImageEncoder(checkpoint,config['vision_config'],gh,gw,1)
