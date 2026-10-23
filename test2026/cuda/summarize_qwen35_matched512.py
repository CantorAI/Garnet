"""Audit actual matched workload/config evidence before computing comparisons."""
import argparse,json,statistics
from pathlib import Path
from qwen35_comparator_speculation import saved_comparator_speculation
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--model',choices=['2b','4b'],required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--lmstudio-suffix',default='',help='Explicit separate corrected-cohort suffix, e.g. -partitioned')
p.add_argument('--garnet-suffix',default='',help='Candidate suffix; baseline input/output files remain unchanged')
p.add_argument('--vllm-suffix',default='',help='Separate comparator cohort, e.g. -mtp3-cache128')
a=p.parse_args();root=Path(a.root);prefix=f'matched512-';indices={};configs={}
for engine in ('garnet','vllm','lmstudio'):
    suffix=a.lmstudio_suffix if engine=='lmstudio' else a.garnet_suffix if engine=='garnet' else a.vllm_suffix
    path=root/f'{prefix}{engine}-{a.model}-b{a.batch}{suffix}.json'
    if not path.exists():continue
    rows=json.loads(path.read_text());assert len(rows)==28,'Incomplete cohort'
    assert all(r['batch']==a.batch and r['matched_batch_inputs'] for r in rows)
    index={(r['trial'],r['workload_group']):r for r in rows};assert len(index)==28
    indices[engine]=index
    if engine=='garnet':assert all(r['profile']['maxInputTokens']+r['profile']['maxOutputTokens']==512 for r in rows)
    else:
        config=json.loads(path.with_suffix('.config.json').read_text());configs[engine]=config
        if engine=='vllm':
            assert config['options']['max_model_len']==512
            assert not config['options']['enable_prefix_caching'] and config['options']['mm_processor_cache_gb']==0
            assert all(not q['num_cached_tokens'] for r in rows for q in r['records'])
        else:
            audit=config.get('actual_load_audit')
            if audit:assert audit['context_per_sequence_verified'] and audit['configured_parallel_slots_verified']
            else:assert a.batch==1 and config['load_config']['contextLength']==512
assert 'garnet' in indices and len(indices)>=2
g=indices['garnet'];measured=[key for key,r in g.items() if not r['warmup']];assert len(measured)==20
def counts(engine,r):
    if engine=='garnet':return r['sampled_tokens_including_stop']
    if engine=='vllm':return [len(q['token_ids']) for q in r['records']]
    return [q['result']['stats']['total_output_tokens'] for q in r['records']]
for key,base in g.items():
    for engine,index in indices.items():
        r=index[key];assert r['input_sha256']==base['input_sha256']
        if engine=='vllm':assert [q['prompt_token_ids'] for q in r['records']]==base['prompt_token_ids']
        if engine=='lmstudio':assert [q['result']['stats']['input_tokens'] for q in r['records']]==base['prompt_tokens']
equal=[key for key in measured if all(counts(e,index[key])==[32]*a.batch for e,index in indices.items())]
result={'model':a.model,'batch':a.batch,'context_capacity':512,'cohorts_present':list(indices),
    'garnet_candidate_suffix':a.garnet_suffix,
    'all_three_engines_present':len(indices)==3,'all_input_hashes_equal':True,
    'garnet_vllm_prompt_ids_equal':'vllm' in indices,'equal_32_sample_groups':len(equal),
    'equal_work_median_group_completion_ms':{e:statistics.median(index[k]['wall_seconds'] for k in equal)*1000 for e,index in indices.items()} if equal else {},
    'all_measured_aggregate_sample_throughput':{e:sum(sum(counts(e,index[k])) for k in measured)/sum(index[k]['wall_seconds'] for k in measured) for e,index in indices.items()},
    'broad_quality_accepted':False,'lmstudio_runtime_slot_cache_audit_complete':False,
    'limitation':'Group completion latency and aggregate throughput are different metrics. Intermediate arithmetic/API timing differ. LM prompt IDs unavailable and runtime slot/cache audit pending; no overall acceptance.'}
mtp_depth=next(iter(g.values()))['profile'].get('speculativeMTP',0)
result['garnet_speculative_mtp_depth']=mtp_depth
result['vllm_baseline_suffix']=a.vllm_suffix
result['saved_comparator_speculation']=saved_comparator_speculation(root,a.model,a.batch,a.lmstudio_suffix,a.vllm_suffix)
result['algorithm_configuration_status']='LM Studio drafting is runtime-audited; vLLM configuration is recorded separately; exact cross-engine proposal settings differ or are unresolved'
result['speculation_configuration_equivalent']=False
result['overall_goal_accepted']=False
load_audit=configs.get('lmstudio',{}).get('actual_load_audit')
result['lmstudio_loaded_slot_context_verified']=bool(load_audit)
result['lmstudio_independent_capacity_verified']=bool(load_audit and load_audit.get('independent_slot_capacity_verified'))
result['comparison_capacity_status']='matched independent capacity' if result['lmstudio_independent_capacity_verified'] else 'provisional: LM independent capacity unverified/shared'
runtime_audit=root/f'matched512-lmstudio-{a.model}-b{a.batch}{a.lmstudio_suffix}-audit.json'
if runtime_audit.exists():
    audit=json.loads(runtime_audit.read_text())
    result['lmstudio_all_full_prompt_evals_observed']=audit['all_expected_prompt_evals_seen']
    result['lmstudio_observed_timed_slots']=audit['observed_timed_slots']
(root/f'matched512-{a.model}-b{a.batch}{a.garnet_suffix}{a.lmstudio_suffix}{a.vllm_suffix}-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
