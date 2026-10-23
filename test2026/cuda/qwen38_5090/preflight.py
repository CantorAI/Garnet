"""Offline checkpoint contract audit. Does not load weights or claim runtime support."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from xModel.qwen38.vl_27b.model import validate_config


def audit(config, batch=1, context=4096, cache_bytes=4):
    if batch not in (1, 2, 4, 8) or not 1 <= context <= 4096:
        raise ValueError('Initial study requires batch 1/2/4/8 and context <=4096')
    if cache_bytes not in (2, 4):
        raise ValueError('Cache estimate supports 2 or 4 byte elements')
    validate_config(config)
    t, v = config['text_config'], config['vision_config']
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
            'blockers':['CUDA serving model admission and 27B GPU validation pending',
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
