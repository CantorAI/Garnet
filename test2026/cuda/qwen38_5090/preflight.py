"""Offline checkpoint contract audit. Does not load weights or claim runtime support."""
import argparse
import json
from pathlib import Path


def audit(config, batch=1, context=4096, cache_bytes=4):
    if batch not in (1, 2, 4, 8) or not 1 <= context <= 4096:
        raise ValueError('Initial study requires batch 1/2/4/8 and context <=4096')
    if cache_bytes not in (2, 4):
        raise ValueError('Cache estimate supports 2 or 4 byte elements')
    if config.get('architectures') != ['Qwen3_5ForConditionalGeneration'] or config.get('model_type') != 'qwen3_5':
        raise ValueError('Unverified model architecture')
    t, v = config['text_config'], config['vision_config']
    expected = {'hidden_size':5120, 'num_hidden_layers':64, 'intermediate_size':17408,
                'num_attention_heads':24, 'num_key_value_heads':4, 'head_dim':256,
                'linear_num_key_heads':16, 'linear_num_value_heads':48,
                'linear_key_head_dim':128, 'linear_value_head_dim':128,
                'linear_conv_kernel_dim':4, 'vocab_size':248320, 'tie_word_embeddings':False}
    for key, value in expected.items():
        if t.get(key) != value:
            raise ValueError('27B text contract mismatch: '+key)
    for key, value in {'depth':27, 'hidden_size':1152, 'intermediate_size':4304,
                       'num_heads':16, 'out_hidden_size':5120, 'patch_size':16,
                       'spatial_merge_size':2, 'temporal_patch_size':2}.items():
        if v.get(key) != value:
            raise ValueError('27B vision contract mismatch: '+key)
    if t.get('layer_types') != (['linear_attention']*3+['full_attention'])*16:
        raise ValueError('Unexpected hybrid layer schedule')
    if len(t['layer_types']) != 64:
        raise ValueError('Expected 64 layers')
    quant = config.get('quantization_config')
    # Only diagnose formats: never treat format recognition as runnable support.
    fmt = 'bf16' if not quant else quant.get('quant_method', 'unknown')
    full = t['layer_types'].count('full_attention')
    linear = t['layer_types'].count('linear_attention')
    kv = batch*full*2*t['num_key_value_heads']*context*t['head_dim']*cache_bytes
    recurrent = batch*linear*t['linear_num_value_heads']*t['linear_key_head_dim']*t['linear_value_head_dim']*4
    conv_width = 2*t['linear_num_key_heads']*t['linear_key_head_dim'] + t['linear_num_value_heads']*t['linear_value_head_dim']
    conv = batch*linear*conv_width*t['linear_conv_kernel_dim']*4
    return {'architecture_verified':True, 'format':fmt, 'batch':batch, 'context_capacity':context,
            'garnet_27b_runnable':False, 'status':'integration_required',
            'cache_estimate_bytes':{'attention_kv':kv, 'recurrent_fp32':recurrent, 'convolution_fp32':conv},
            'estimate_scope':'Single canonical state only; excludes weights, duplicate states, graphs, workspace, activations and vision. Not peak VRAM.',
            'quantization':quant and {'method':fmt, 'algorithm':quant.get('quant_algo'), 'producer':quant.get('producer')},
            'blockers':['CUDA serving model admission and 27B factory missing',
                        'NVFP4/FP8 mixed loader and projection kernels unverified',
                        'INT4 captured-weight path does not establish external BF16 projection compatibility',
                        'Single RTX 5090 numerical, memory and performance validation pending']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--context', type=int, default=4096)
    a = p.parse_args()
    print(json.dumps(audit(json.loads(a.config.read_text(encoding='utf-8-sig')), a.batch, a.context), indent=2))
