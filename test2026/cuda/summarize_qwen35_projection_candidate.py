"""Compare candidate to current strict baseline, preserving unequal-work cases."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--model',choices=('2b','4b'),default='2b')
a=p.parse_args();root=Path(a.root)
files={
    'candidate':f'benchmark-garnet-{a.model}-b1-full-projection-candidate.json',
    'prior_strict':f'benchmark-garnet-{a.model}-b1-fused-chunk4-strict.json',
    'lmstudio':f'benchmark-lmstudio-{a.model}-b1-rest-variants.json',
    'vllm':f'benchmark-vllm-{a.model}-b1-variants.json'}
cohorts={k:json.loads((root/name).read_text()) for k,name in files.items()}
def key(r):return r['trial'],tuple(r['images'])
indices={k:{key(r):r for r in rows} for k,rows in cohorts.items()}
assert all(len(rows)==28 for rows in cohorts.values())
assert all(set(index)==set(indices['candidate']) for index in indices.values())
def samples(engine,r):
    if engine=='lmstudio':return r['result']['stats']['total_output_tokens']
    if engine=='vllm':return len(r['records'][0]['token_ids'])
    return r['sampled_tokens_including_stop'][0]
keys=[key(r) for r in cohorts['candidate'] if not r['warmup']]
assert len(keys)==20
paired=[k for k in keys if all(samples(e,index[k])==32 for e,index in indices.items())]
for k,r in indices['candidate'].items():
    assert r['arithmetic']['projection_reduced_precision_reduction'] is False
    assert r['arithmetic']['full_attention_projection_layout']=='chunk_batched'
    assert r['vision_compile_ms']==0
    assert r['prompt_token_ids']==indices['prior_strict'][k]['prompt_token_ids']
    assert r['prompt_token_ids'][0]==indices['vllm'][k]['records'][0]['prompt_token_ids']
def median_ms(index,chosen):return statistics.median(index[k]['wall_seconds'] for k in chosen)*1000
candidate=cohorts['candidate'];measured=[r for r in candidate if not r['warmup']]
result={'model':a.model,'candidate_status':'experimental; not quality accepted',
    'raw_sha256':{e:hashlib.sha256((root/name).read_bytes()).hexdigest() for e,name in files.items()},
    'measured_requests':len(keys),'equal_32_sample_cases':len(paired),
    'all_measured_wall_ms':{e:median_ms(index,keys) for e,index in indices.items()},
    'equal_work_wall_ms':{e:median_ms(index,paired) for e,index in indices.items()} if paired else {},
    'candidate_median_ttft_ms':statistics.median(r['ttft_seconds'] for r in measured)*1000,
    'candidate_median_decode_tokens_per_second':statistics.median(r['decode_steps']/r['decode_seconds'] for r in measured),
    'exact_prior_generated_token_sequences':sum(r['generated_token_ids']==indices['prior_strict'][key(r)]['generated_token_ids'] for r in candidate),
    'recorded_requests':len(candidate),
    'retained_owned_model_bytes':sorted({r['owned_memory']['owned_model_bytes'] for r in measured}),
    'broad_quality_accepted':False,
    'limitations':'Candidate batch8 synthetic numeric check unresolved. BF16 projection GEMM geometry changed. Native API timing differs from external engines; LM context512 vs384. Batch ownership is not isolated process VRAM. Prior external cohorts reused, not contemporaneous.'}
if paired:
    walls=result['equal_work_wall_ms'];result['prior_strict_over_candidate']=walls['prior_strict']/walls['candidate']
    result['two_x_best_external_target_ms']=min(walls['lmstudio'],walls['vllm'])/2
references=[]
for path in sorted(root.glob(f'full-projection-{a.model}-changed-reference-*.json')):
    ref=json.loads(path.read_text());q=ref['quantitative_hidden_logit_comparison']
    assert ref['full_attention_projection_layout']=='chunk_batched' and ref['native_input_embeddings']
    references.append({'source':path.name,'decisions':len(q),
        'exact_ids':sum(r['native_selected_token']==r['reference_top1'] for r in q),
        'maximum_set':sum(r['native_selected_reference_rank']==1 for r in q),
        'head_reconstructions':sum(r['native_selected_token']==r['native_hidden_reconstructed_top1'] for r in q),
        'max_hidden_relative_l2':max(r['hidden_relative_l2'] for r in q)})
result['changed_trajectory_decoder_references']=references
old=root/'test-old-projection-b8-real-attention.log';new=root/'test-full-projection-b8-real-attention.log'
if old.exists() and new.exists():
    signature="AssertionError: ('max_abs_error', 0.00787353515625)"
    result['synthetic_batch8_failure_reproduced_on_prior_build']=all(signature in path.read_text(errors='replace') for path in (old,new))
(root/f'full-projection-{a.model}-candidate-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
