"""Attribute retained allocation growth; never sum per-engine peaks."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--model',choices=['2b','4b'],required=True)
a=p.parse_args();root=Path(a.root);profiles=[]
config=json.loads((root/'models'/('Qwen3.5-'+a.model.upper())/'config.json').read_text())['text_config']
for batch in (1,2,4,8):
    path=root/f'matched512-garnet-{a.model}-b{batch}.json'
    if not path.exists():continue
    rows=json.loads(path.read_text())
    if len(rows)!=28:continue
    measured=[r for r in rows if not r['warmup']]
    memories=[r['owned_memory'] for r in measured]
    assert len({m['owned_model_bytes'] for m in memories})==1
    m=memories[-1];d=m['decoder'];graphs=d['graphs']
    assert len(graphs)==config['num_hidden_layers']+1
    values={'owned_model_bytes':m['owned_model_bytes'],'full_attention_engine_live_bytes':0,'linear_attention_engine_live_bytes':0}
    for kind,graph in zip(config['layer_types'],graphs):
        values['full_attention_engine_live_bytes' if kind=='full_attention' else 'linear_attention_engine_live_bytes']+=graph['trt_allocator_live_bytes']
    for field in ('state_bytes','engine_output_bytes','request_buffer_bytes','embedding_head_shared_bytes','head_input_bytes','head_output_bytes','trt_runtime_allocator_live_bytes'):values[field]=d[field]
    values['vision_live_bytes']=m['vision']['trt_allocator_live_bytes']
    profiles.append({'batch':batch,'context':512,'retained_after_warmup_stable':True,'bytes':values,'source':path.name})
assert profiles and profiles[0]['batch']==1
base=profiles[0]['bytes']
for profile in profiles:profile['delta_from_batch1_bytes']={k:v-base[k] for k,v in profile['bytes'].items()}
result={'model':a.model,'profiles':profiles,
    'scope':'Retained logical tensor ownership and TensorRT live allocation counters. Engine allocations include weights and runtime buffers; not pure workspace. Per-engine peaks are not added. Not process VRAM or peak residency.',
    'limitation':'Only completed current512 profiles included; allocator deltas attribute where storage grows, not exact temporary lifetimes or bandwidth/time costs.'}
(root/f'matched512-{a.model}-allocation-growth.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
