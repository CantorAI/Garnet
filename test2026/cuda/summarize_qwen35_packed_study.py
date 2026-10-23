"""Same-build ABBA/ABCCBA real-image profile comparisons, no favorable subsets."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--model',choices=('2b','4b'),required=True)
p.add_argument('--batch',type=int,choices=(1,2,4,8),required=True)
feature=p.add_mutually_exclusive_group();feature.add_argument('--reuse-head',action='store_true');feature.add_argument('--joined-decode',action='store_true');a=p.parse_args()
study='joined-decode' if a.joined_decode else 'promptpack-reusehead' if a.reuse_head else 'promptpack';root=a.root
roles=('a1','b1','c1','c2','b2','a2') if a.reuse_head else ('a1','b1','b2','a2')
output=root/f'packed-study-{a.model}-b{a.batch}-{study}-summary.json';assert not output.exists(),'Preserve study summary'
data={};sources={};binary=set()
for role in roles:
    name=f'matched512-garnet-{a.model}-b{a.batch}-study-{study}-{role}'
    source=root/(name+'.json');exit_data=json.loads((root/(name+'-exit.json')).read_text())
    assert exit_data['exit_code']==0;binary.add(exit_data['garnet_dll_sha256'])
    rows=json.loads(source.read_text());assert len(rows)==28
    data[role]={(r['trial'],r['workload_group']):r for r in rows};assert len(data[role])==28
    sources[role]={'file':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
assert len(binary)==1
base=data['a1'];keys=[k for k,r in base.items() if not r['warmup']];assert len(keys)==20
sig=lambda r:{k:r[k] for k in ('generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids')}
for role,rows in data.items():
    assert rows.keys()==base.keys()
    expected_pack=a.joined_decode or not role.startswith('a');expected_reuse=a.joined_decode or role.startswith('c')
    expected_join=a.joined_decode and role.startswith('b')
    for k,r in rows.items():
        assert r['input_sha256']==base[k]['input_sha256'] and r['prompt_token_ids']==base[k]['prompt_token_ids']
        assert sig(r)==sig(base[k]),('Changed output/stop signature',role,k)
        assert r['batch']==a.batch and r['request_output_limit']==32
        assert r['profile'].get('packPrompt',False)==expected_pack
        assert r['profile'].get('reuseVerifiedHead',False)==expected_reuse
        assert r['profile'].get('joinDecodeInputProjections',False)==expected_join
        assert r['warmup']==base[k]['warmup']
        original=dict(base[k]['profile']);current=dict(r['profile'])
        transitions=('joinDecodeInputProjections',) if a.joined_decode else ('packPrompt','reuseVerifiedHead')
        for feature in transitions:original.pop(feature,None);current.pop(feature,None)
        assert original==current,('Other profile parameters differ',role,k)
        assert r['cache_policy']=={'prefix':False,'image':False,'answer':False}
        if a.joined_decode:
            expected_groups=(18 if a.model=='2b' else 24) if expected_join else 0
            assert r['io_memory']['joined_projection_group_count']==expected_groups
            assert sum(g['joined_projection_inputs'] for g in r['owned_memory']['decoder']['graphs'])==expected_groups
    if expected_reuse:assert sum(rows[k]['speculative_metrics'].get('target_head_reuses',0) for k in keys)>0

def med(values):return statistics.median(values)
run_metrics={}
for role,rows in data.items():
    run_metrics[role]={'median_group_ms':med(rows[k]['wall_seconds'] for k in keys)*1000,
        'median_full_request_output_tokens_per_second':med(sum(rows[k]['generated_tokens'])/rows[k]['wall_seconds'] for k in keys),
        'median_derived_post_first_sample_tokens_per_second':med((sum(rows[k]['sampled_tokens_including_stop'])-a.batch)/rows[k]['decode_seconds'] for k in keys),
        'phase_medians_ms_not_additive':{name:med(rows[k][name] for k in keys)*1000 for name in ('input_seconds','prefill_seconds','decode_seconds')},
        'kept_gpu_allocation_bytes':sorted({rows[k]['owned_memory']['owned_model_bytes'] for k in keys})}
groups={letter:{k:med(data[letter+repeat][k]['wall_seconds'] for repeat in ('1','2'))*1000 for k in keys}
    for letter in ('a','b','c') if letter+'1' in data}
aggregates={letter:med(values.values()) for letter,values in groups.items()}
comparisons={}
for before,after in (('a','b'),('b','c'),('a','c')):
    if before in groups and after in groups:
        comparisons[before+'_to_'+after]={'ratio_of_aggregate_medians':aggregates[before]/aggregates[after],
            'reduction_fraction':1-aggregates[after]/aggregates[before],
            'median_per_group_speed_ratio':med(groups[before][k]/groups[after][k] for k in keys)}
result={'sources':sources,'garnet_dll_sha256':next(iter(binary)),'order':roles,'model':a.model,'batch':a.batch,
    'profiles':({'a':'native-prefix/arena D4; prompt packing and verified-head reuse on; joined decode projections off',
        'b':'same with joined decode input projections on'} if a.joined_decode else
        {'a':'native-prefix/arena D4; prompt packing off; head reuse off',
        'b':'same with prompt packing on','c':'same with prompt packing and verified-head reuse on'}),
    'warm_groups':keys,'same_all_28_input_prompt_output_stop_signatures':True,'per_run':run_metrics,
    'per_group_repeated_median_ms':{letter:[{'trial':k[0],'workload_group':k[1],'ms':v} for k,v in values.items()] for letter,values in groups.items()},
    'aggregate_group_median_ms':aggregates,'comparisons':comparisons,
    'independent_quality_accepted':False,'goal_2x_achieved':False,
    'scope':'Fresh processes in ABBA or ABCCBA order, same DLL and fixed profiles, full28 requests per run, two warmup rounds excluded. All20 warm groups retained; equal input/output signatures required. Aggregates first median the two repeats per group, then median across groups. No significance/causal claim from two repeats, comparator rerun, peak/residency proof or independent quality acceptance. Derived decode rate subtracts one sampled token per row and includes host/stop overhead; phase medians do not add.'}
output.write_text(json.dumps(result,indent=2));print(json.dumps({'aggregate_group_median_ms':aggregates,'comparisons':comparisons},indent=2))
