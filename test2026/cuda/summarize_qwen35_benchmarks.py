"""Audit the initial sequential batch1 cohort and write qualified statistics."""
import json,sys,statistics
from pathlib import Path
root=Path(sys.argv[1])
model=sys.argv[2] if len(sys.argv)>2 else 'Qwen3.5-2B'
assert model in ('Qwen3.5-2B','Qwen3.5-4B')
size=model.split('-')[-1].lower()
garnet_file=sys.argv[3] if len(sys.argv)>3 else 'benchmark-garnet-'+size+'-b1-variants.json'
output_file=sys.argv[4] if len(sys.argv)>4 else 'initial-batch1-comparison.json'
files={'garnet':garnet_file,
    'vllm':'benchmark-vllm-'+size+'-b1-variants.json',
    'lmstudio':'benchmark-lmstudio-'+size+'-b1-rest-variants.json'}
cohorts={engine:json.loads((root/name).read_text()) for engine,name in files.items()}
def key(row):return row['trial'],tuple(row['images'])
for engine,rows in cohorts.items():
    assert len(rows)==28 and len({key(r) for r in rows})==28,(engine,len(rows))
    assert len([r for r in rows if not r['warmup']])==20
    assert all(r.get('variant_round')==r['trial'] for r in rows)
g={key(r):r for r in cohorts['garnet']};v={key(r):r for r in cohorts['vllm']}
assert set(g)==set(v)=={key(r) for r in cohorts['lmstudio']}
assert all(g[k]['prompt_token_ids'][0]==v[k]['records'][0]['prompt_token_ids'] for k in g)
assert all(len(v[k]['records'][0]['prompt_token_ids'])==85 for k in v)
assert all(not v[k]['records'][0]['num_cached_tokens'] for k in v)
assert all(r['result']['stats']['input_tokens']==85 and r['result']['stats']['reasoning_output_tokens']==0 for r in cohorts['lmstudio'])
stats={}
work_by_engine={}
for engine,rows in cohorts.items():
    rows=[r for r in rows if not r['warmup']]
    walls=[r['wall_seconds'] for r in rows]
    if engine=='garnet':
        ttft=[r['ttft_seconds'] for r in rows]
        # Older retained results omit sampled-stop count. For batch1 only,
        # infer EOS when generation ends below limit (the only early stop).
        work=[r.get('sampled_tokens_including_stop',[len(r['generated_token_ids'][0])+int(len(r['generated_token_ids'][0])<r['request_output_limit'])])[0] for r in rows]
        decode=[(n-1)/r['decode_seconds'] for r,n in zip(rows,work)]
    elif engine=='vllm':
        ttft=[r['records'][0]['metrics']['first_token_latency'] for r in rows]
        work=[len(r['records'][0]['token_ids']) for r in rows]
        decode=[(n-1)/(r['records'][0]['metrics']['last_token_ts']-r['records'][0]['metrics']['first_token_ts']) for r,n in zip(rows,work)]
    else:
        ttft=[r['result']['stats']['time_to_first_token_seconds'] for r in rows]
        work=[r['result']['stats']['total_output_tokens'] for r in rows]
        decode=[r['result']['stats']['tokens_per_second'] for r in rows]
    stats[engine]={'measured_requests':20,'median_wall_ms':statistics.median(walls)*1000,
        'median_ttft_ms':statistics.median(ttft)*1000,'median_decode_tokens_per_second':statistics.median(decode),
        'sampled_output_tokens_min':min(work),'sampled_output_tokens_max':max(work),
        'source':files[engine]}
    work_by_engine[engine]={key(r):n for r,n in zip(rows,work)}
paired_keys=[k for k in work_by_engine['garnet'] if len({work_by_engine[e][k] for e in cohorts})==1]
paired={engine:{'requests':len(paired_keys),
    'sampled_token_counts':sorted({work_by_engine[engine][k] for k in paired_keys}),
    'median_wall_ms':statistics.median(r['wall_seconds'] for r in rows if key(r) in paired_keys)*1000 if paired_keys else None}
    for engine,rows in cohorts.items()}
result={'model':model,'batch':1,'prompt_tokens':85,'visual_tokens':60,'output_limit':32,
    'garnet_projection_reduced_precision_reduction':cohorts['garnet'][0].get('arithmetic',{}).get('projection_reduced_precision_reduction'),
    'weights':'Official checkpoint BF16 / derived BF16 GGUF; engine-specific intermediate arithmetic differs.',
    'stats':stats,'equal_sampled_token_work_subset':paired,
    'all_garnet_vllm_prompt_ids_equal':True,'lmstudio_prompt_token_count_matches':True,
    'limitations':['LM Studio actual prompt IDs unavailable; count equality is weaker evidence.',
        'LM Studio rounds requested384 context to512; Garnet/vLLM384 logical capacity.',
        'LM Studio fresh images are cache-busted, but short text-only prefix reuse is not fully quantified.',
        'JPEG decode/interpolation and internal precision differ by engine.',
        ('All20 measured requests match sampled output-token work across engines.' if len(paired_keys)==20 else 'Early EOS yields unequal output work in part of the cohort; only the paired subset matches sampled token counts across all three engines.'),
        'TTFT/decode definitions differ between direct API and server-reported engine metrics.',
        'This is an initial performance cohort; isolated comparable steady/peak memory and broad quality acceptance are pending.',
        ('Garnet sampled-stop counts are explicitly recorded.' if all('sampled_tokens_including_stop' in r for r in cohorts['garnet']) else 'Garnet older sampled-stop counts are inferred for batch1 from its EOS-only early-stop behavior.')],
    'goal_2x_achieved':False}
owned=[r['owned_memory']['owned_model_bytes'] for r in cohorts['garnet'] if not r['warmup'] and 'owned_memory' in r]
if owned:result['garnet_retained_owned_memory']={'min_bytes':min(owned),'max_bytes':max(owned),'stable':len(set(owned))==1,
    'scope':'Retained tensors + tracked TRT runtime; excludes driver/external libraries, graph executables, transient requests/build workspace. Not process peak VRAM.'}
(root/output_file).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
