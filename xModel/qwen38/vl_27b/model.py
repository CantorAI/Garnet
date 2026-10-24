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
                       'spatial_merge_size':2, 'temporal_patch_size':2,
                       'in_channels':3,'num_position_embeddings':2304,
                       'hidden_act':'gelu_pytorch_tanh'}.items():
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


def create_cuda_packed_decoder(checkpoint,config,capacity,batch=1,retain_linear_inputs=False):
    """Research candidate: one embedded mixed program, one16-token profile."""
    validate_cuda_profile(config,capacity,batch)
    return _create_cuda_packed_embedded(checkpoint,config,capacity,batch,retain_linear_inputs)


def create_cuda_shared_compressed_decoder(checkpoint,config,capacity,batch=1,retain_linear_inputs=False,prefill_chunk=128,decode_chunk=4):
    """Unqualified opt-in: two static profiles borrow one mixed projection pool."""
    validate_cuda_profile(config,capacity,batch)
    if type(prefill_chunk) is not int or prefill_chunk not in (32,64,128) or type(decode_chunk) is not int or decode_chunk not in (4,8,16) or prefill_chunk<decode_chunk:
        raise ValueError('Shared compressed candidate requires bounded prefill32/64/128 and decode4/8/16')
    import json
    from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
    decoder=CUDAPackedDecoder(checkpoint,config['text_config'],batch,capacity,prefill_chunk,decode_chunk,
        flat_concat=True,external_compressed_weights=True,shared_output_storage=True,retain_linear_inputs=retain_linear_inputs)
    try:
        decoder.io.enable_nvfp4_head(True)
        pool=json.loads(decoder.io.memory_json())
        if pool['compressed_projection_pool_entries']!=400 or pool['compressed_nvfp4_entries']!=192 or pool['compressed_fp8_entries']!=208:
            raise ValueError('Shared mixed pool must contain exactly192NVFP4+208FP8 projection matrices')
        physical=pool['compressed_projection_weight_bytes']+pool['compressed_projection_scale_bytes']
        for program in decoder.programs.values():
            stats=json.loads(program.graph.stats_json())
            if stats['native_nvfp4_projection_weights']!=192 or stats['native_fp8_projection_weights']!=208 or stats['external_bf16_weight_inputs']!=0 or stats['external_compressed_weight_inputs']!=400 or stats['external_compressed_weight_bytes']!=physical:
                raise ValueError('Shared whole-program compressed projection contract mismatch')
        return decoder
    except:
        decoder.close();raise


def validate_cuda_batch_study_profile(config,capacity,batch):
    """Opt-in backend qualification only; not general serving admission."""
    validate_cuda_profile(config,capacity,1)
    if type(batch) is not int or batch not in (2,4) or type(capacity) is not int:
        raise ValueError('Batch study requires batch2/4; numerical/memory gates remain pending')
    return True


def create_cuda_batch_study_decoder(checkpoint,config,capacity,batch=2,retain_linear_inputs=False):
    validate_cuda_batch_study_profile(config,capacity,batch)
    return _create_cuda_packed_embedded(checkpoint,config,capacity,batch,retain_linear_inputs)


def _create_cuda_packed_embedded(checkpoint,config,capacity,batch,retain_linear_inputs):
    import json
    from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
    decoder=CUDAPackedDecoder(checkpoint,config['text_config'],batch,capacity,16,16,
        flat_concat=True,embedded_projection_weights=True,retain_linear_inputs=retain_linear_inputs)
    try:
        decoder.io.enable_nvfp4_head(True)
        stats=json.loads(decoder.programs[16].graph.stats_json())
        if stats['native_nvfp4_projection_weights']!=192 or stats['native_fp8_projection_weights']!=208 or stats['external_bf16_weight_inputs']!=0:
            raise ValueError('Embedded whole-program mixed projection contract mismatch')
        return decoder
    except:
        decoder.close();raise


def create_cuda_vision(checkpoint,config,gh,gw):
    validate_cuda_profile(config,4096,1)
    from xModel.qwen35.vl_0_8b.cuda_decoder import CUDAImageEncoder
    return CUDAImageEncoder(checkpoint,config['vision_config'],gh,gw,1)


def mtp_shapes(config):
    """Pinned one-layer BF16 draft contract; runtime/GPU equivalence is separate."""
    validate_config(config)
    c=config['text_config'];h=c['hidden_size'];m=c['intermediate_size'];q=c['num_attention_heads']*c['head_dim'];kv=c['num_key_value_heads']*c['head_dim']
    if c.get('mtp_num_hidden_layers',1)!=1:raise ValueError('One checkpoint MTP layer required')
    result={'mtp.fc.weight':[h,2*h]}
    for name in ('pre_fc_norm_embedding','pre_fc_norm_hidden','norm'):result['mtp.'+name+'.weight']=[h]
    for name in ('input_layernorm','post_attention_layernorm'):result['mtp.layers.0.'+name+'.weight']=[h]
    for name in ('q_norm','k_norm'):result['mtp.layers.0.self_attn.'+name+'.weight']=[c['head_dim']]
    for name,shape in {'q_proj':[2*q,h],'k_proj':[kv,h],'v_proj':[kv,h],'o_proj':[h,q]}.items():result['mtp.layers.0.self_attn.'+name+'.weight']=shape
    for name,shape in {'gate_proj':[m,h],'up_proj':[m,h],'down_proj':[h,m]}.items():result['mtp.layers.0.mlp.'+name+'.weight']=shape
    return result


def validate_mtp_metadata(config,metadata):
    for name,shape in mtp_shapes(config).items():
        value=metadata.get(name,{})
        if value.get('dtype')!='BF16' or value.get('shape')!=shape:raise ValueError('MTP BF16 tensor contract mismatch: '+name)
    return True


def create_cuda_mtp_session(decoder,checkpoint,config,stops,depth=3,draft_chunk_size=None):
    """Research-only verified draft; no indexed NVFP4 head or second target copy."""
    validate_cuda_profile(config,decoder.capacity,decoder.batch)
    if depth not in (1,3,7) or not getattr(decoder,'packed_state_program',False) or not decoder.retain_linear_inputs:
        raise ValueError('MTP requires depth1/3/7 and retained packed target projections')
    import json
    metadata={name:json.loads(decoder.io.weight_info(name)) for name in mtp_shapes(config)}
    validate_mtp_metadata(config,metadata)
    if json.loads(decoder.io.memory_json())['head_precision']!='modelopt_nvfp4_w4a4':raise ValueError('MTP requires the target native NVFP4 head')
    from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
    return CUDASpeculativeSession(decoder,checkpoint,stops,depth,head_candidates=0,prefix_replay=True,draft_chunk_size=draft_chunk_size)


def validate_image_processor(processor,max_pixels=1048576):
    """Initial still-image JPEG profile; reject incompatible preprocessing."""
    expected={'patch_size':16,'temporal_patch_size':2,'merge_size':2,
              'image_mean':[.5,.5,.5],'image_std':[.5,.5,.5]}
    for key,value in expected.items():
        if processor.get(key)!=value:
            raise ValueError('Qwen3.8 image processor mismatch: '+key)
    size=processor.get('size',{})
    if size.get('shortest_edge')!=65536 or size.get('longest_edge')!=16777216:
        raise ValueError('Qwen3.8 image processor pixel bounds mismatch')
    if max_pixels<65536 or max_pixels>1048576:
        raise ValueError('Initial Qwen3.8 JPEG profile requires65536..1048576 max pixels')
    return 65536,max_pixels


def prepare_cuda_image(io,checkpoint,path,max_pixels=1048576):
    import json,os
    with open(os.path.join(checkpoint,'preprocessor_config.json')) as f:
        processor=json.load(f)
    low,high=validate_image_processor(processor,max_pixels)
    return io.prepare_image_profile(path,low,high)


def validate_cuda_int4_profile(config,capacity,batch=1):
    """Research-only compressed-tensors recipe; no general serving admission."""
    validate_config(config)
    q=config.get('quantization_config',{});groups=q.get('config_groups',{});g=groups.get('group_0',{});w=g.get('weights',{})
    if (q.get('quant_method')!='compressed-tensors' or q.get('format')!='pack-quantized' or set(groups)!={'group_0'} or
        g.get('format')!='pack-quantized' or g.get('input_activations') is not None or g.get('output_activations') is not None or
        w.get('num_bits')!=4 or w.get('group_size')!=128 or w.get('symmetric') is not True or w.get('dynamic') is not False or
        w.get('type')!='int' or w.get('strategy')!='group'):
        raise ValueError('Research INT4 requires verified symmetric packed G128 W4A16 recipe')
    if batch!=1 or not 32<=capacity<=4096:raise ValueError('Initial INT4 study requires batch1/capacity32..4096')
    return True


def create_cuda_int4_decoder(checkpoint,config,capacity=4096):
    """Explicit index preserves BF16 vision/MTP sidecars; all496 text projections stay packed."""
    import json
    from pathlib import Path
    validate_cuda_int4_profile(config,capacity)
    index=Path(checkpoint)/'model.safetensors.index.json'
    if not index.is_file():raise ValueError('INT4 study requires explicit verified shard index')
    from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
    decoder=CUDADecoder(str(index),config['text_config'],1,capacity,16,True,True)
    try:
        graphs=[json.loads(layer['graph'].stats_json()) for layer in decoder.layers]
        if (sum(g.get('native_int4_g128_projection_weights',0) for g in graphs)!=496 or
            any(g.get('external_bf16_weight_inputs',0) or g.get('checkpoint_weight_precision')!='ct_int4_g128' for g in graphs)):
            raise ValueError('INT4 decoder must retain all496 packed projections without BF16 input fallback')
        if json.loads(decoder.io.memory_json())['head_precision']!='bf16':raise ValueError('Pinned INT4 head must remain BF16')
        return decoder
    except:
        decoder.close();raise


def create_cuda_int4_packed_decoder(checkpoint,config,capacity=4096,chunk_size=16):
    """Unqualified whole-graph alternative; same packed G128/BF16-head recipe."""
    import json
    from pathlib import Path
    validate_cuda_int4_profile(config,capacity)
    if type(chunk_size) is not int or chunk_size not in (4,8,16):
        raise ValueError('Packed INT4 study requires static chunk4/8/16')
    index=Path(checkpoint)/'model.safetensors.index.json'
    if not index.is_file():raise ValueError('Packed INT4 study requires explicit verified shard index')
    from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
    decoder=CUDAPackedDecoder(str(index),config['text_config'],1,capacity,chunk_size,chunk_size,
        flat_concat=True,embedded_projection_weights=True)
    try:
        stats=json.loads(decoder.programs[chunk_size].graph.stats_json())
        if (stats.get('native_int4_g128_projection_weights')!=496 or
            stats.get('native_nvfp4_projection_weights',0) or stats.get('native_fp8_projection_weights',0) or
            stats.get('external_bf16_weight_inputs',0) or stats.get('checkpoint_weight_precision')!='ct_int4_g128'):
            raise ValueError('Whole INT4 program must retain exactly496 G128 projections')
        if json.loads(decoder.io.memory_json())['head_precision']!='bf16':raise ValueError('Pinned INT4 head must remain BF16')
        return decoder
    except:
        decoder.close();raise
