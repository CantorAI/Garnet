"""Repeated Garnet profiles versus historical engines on one equal-work subset."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
p.add_argument('--model',choices=('2b','4b'),required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--vllm-extra-suffix',action='append',default=[]);p.add_argument('--joined-decode',action='store_true');a=p.parse_args()
assert not a.output.exists(),'Preserve existing summary'
root=a.root;study_name='joined-decode' if a.joined_decode else 'promptpack-reusehead'
study_path=root/f'packed-study-{a.model}-b{a.batch}-{study_name}-summary.json'
study=json.loads(study_path.read_text());assert study['same_all_28_input_prompt_output_stop_signatures']
rows={};sources={}
for role,record in study['sources'].items():
    path=root/record['file'];blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
    assert digest==record['sha256'],'Raw study changed after preservation summary'
    rows[role]={(r['trial'],r['workload_group']):r for r in json.loads(blob)}
    assert len(rows[role])==28;sources[role]={'file':path.name,'sha256':digest}
external={'lmstudio':f'matched512-lmstudio-{a.model}-b{a.batch}-partitioned.json',
    'vllm_serial':f'matched512-vllm-{a.model}-b{a.batch}.json'}
for suffix in a.vllm_extra_suffix:external['vllm'+suffix]=f'matched512-vllm-{a.model}-b{a.batch}{suffix}.json'
for name,filename in external.items():
    blob=(root/filename).read_bytes();rows[name]={(r['trial'],r['workload_group']):r for r in json.loads(blob)}
    assert len(rows[name])==28;sources[name]={'file':filename,'sha256':hashlib.sha256(blob).hexdigest()}
base=rows['a1']
for name,data in rows.items():
    assert data.keys()==base.keys()
    for key,row in data.items():
        assert row['input_sha256']==base[key]['input_sha256'] and row['batch']==a.batch
        assert row['warmup']==base[key]['warmup']
        if name.startswith('vllm'):assert [q['prompt_token_ids'] for q in row['records']]==base[key]['prompt_token_ids']
def counts(name,row):
    if name in study['sources']:return row['sampled_tokens_including_stop']
    if name=='lmstudio':return [q['result']['stats']['total_output_tokens'] for q in row['records']]
    return [len(q['token_ids']) for q in row['records']]
keys=[key for key,row in base.items() if not row['warmup'] and all(counts(name,data[key])==[32]*a.batch for name,data in rows.items())]
assert keys,'No common equal32 groups'
med=statistics.median
letters='ab' if a.joined_decode else 'abc';selected='garnet_b' if a.joined_decode else 'garnet_c'
latencies={'garnet_'+letter:med(med(rows[letter+repeat][key]['wall_seconds'] for repeat in ('1','2')) for key in keys)*1000 for letter in letters}
latencies.update({name:med(rows[name][key]['wall_seconds'] for key in keys)*1000 for name in external})
fastest=min(external,key=lambda name:latencies[name]);threshold=latencies[fastest]/2
result={'model':a.model,'batch':a.batch,'sources':sources,'study_summary':study_path.name,'common_equal32_groups':len(keys),
    'group_keys':keys,'median_group_ms':latencies,'fastest_tested_baseline':fastest,'half_fastest_baseline_ms':threshold,
    selected+'_speed_ratio_vs_fastest':latencies[fastest]/latencies[selected],
    'latency_threshold_met':latencies[selected]<=threshold,'independent_quality_accepted':False,'goal_2x_achieved':False,
    'scope':f'One common equal32 subset across all {len(study["sources"])} Garnet runs and every listed historical competitor. Garnet first medians two repeats per group, then groups; competitors are single saved cohorts and were not rerun. APIs, intermediate arithmetic and speculative settings differ. LM2B drafting observed, LM4B unresolved. No significance, causal, independent-quality, or exhaustive-baseline claim.'}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('sources','group_keys')},indent=2))
