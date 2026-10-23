"""Rank completed comparator profiles on one common equal-work subset."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output')
p.add_argument('--model',required=True,choices=['2b','4b']);p.add_argument('--batch',required=True,type=int)
p.add_argument('--garnet-suffix',required=True);p.add_argument('--control-suffix',required=True)
p.add_argument('--vllm-extra-suffix',action='append',default=[])
a=p.parse_args();root=Path(a.root);base=f'matched512-{{}}-{a.model}-b{a.batch}{{}}.json'
names={'garnet':base.format('garnet',a.garnet_suffix),'control':base.format('garnet',a.control_suffix),
       'lmstudio':base.format('lmstudio','-partitioned'),'vllm_serial':base.format('vllm','')}
for suffix in a.vllm_extra_suffix:names['vllm'+suffix]=base.format('vllm',suffix)
data={};identities={}
for label,name in names.items():
    blob=(root/name).read_bytes();rows=json.loads(blob);assert len(rows)==28
    data[label]={(r['trial'],r['workload_group']):r for r in rows}
    assert len(data[label])==28
    identities[label]={'file':name,'sha256':hashlib.sha256(blob).hexdigest()}
def count(label,r):
    if label in ('garnet','control'):return r['sampled_tokens_including_stop']
    if label=='lmstudio':return [q['result']['stats']['total_output_tokens'] for q in r['records']]
    return [len(q['token_ids']) for q in r['records']]
for key,g in data['garnet'].items():
    for label,rows in data.items():
        r=rows[key];assert r['input_sha256']==g['input_sha256'] and r['batch']==a.batch
        assert r['warmup']==g['warmup']
        if label.startswith('vllm'):assert [q['prompt_token_ids'] for q in r['records']]==g['prompt_token_ids']
        if label=='control':assert r['prompt_token_ids']==g['prompt_token_ids']
equal=[k for k,g in data['garnet'].items() if not g['warmup'] and all(count(label,rows[k])==[32]*a.batch for label,rows in data.items())]
assert equal,'No common equal-output groups'
medians={label:statistics.median(rows[k]['wall_seconds'] for k in equal)*1000 for label,rows in data.items()}
baselines={label:value for label,value in medians.items() if label not in ('garnet','control')}
fastest=min(baselines,key=baselines.get)
result={'model':a.model,'batch':a.batch,'sources':identities,'common_equal32_groups':len(equal),'group_keys':equal,
    'median_group_ms':medians,'fastest_baseline':fastest,'half_fastest_baseline_ms':baselines[fastest]/2,
    'latency_threshold_met':medians['garnet']<=baselines[fastest]/2,
    'same_control_sequences':sum(g['generated_token_ids']==data['control'][k]['generated_token_ids'] for k,g in data['garnet'].items()),
    'quality_accepted':False,'goal_met':False,
    'scope':'All profiles use this one common equal32 subset. API timing and intermediate arithmetic/proposal algorithms differ. LM raw prompt IDs unavailable. Retained historical baselines; no independent quality acceptance.'}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
