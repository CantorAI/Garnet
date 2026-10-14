"""Recompute profiler overhead; do not infer decode roles from missing samples."""
import collections,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();jobs=root/'logs/counter-trace-sparse-profiles-20261015/jobs'
profiles=json.loads((jobs/'profiles.json').read_text())['profiles'];results=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for batch in (1,2):
        warm={policy:[r for p in profiles if p['model']==model and p['batch']==batch and p['tracing']==policy for r in p['records'] if r['request']>0] for policy in (False,True)}
        assert all(len(v)==4 for v in warm.values())
        observed={name:dict(decode_tps=statistics.mean(r['decode_wall_steps_per_second'] for r in warm[policy]),ttft_ms=statistics.mean(r['ttft_seconds'] for r in warm[policy])*1000) for name,policy in (('quiet',False),('traced',True))}
        groups=collections.defaultdict(lambda:dict(count=0,raw_ticks=0));samples=0;commands=0;head_samples=0
        for p in profiles:
            if p['model']!=model or p['batch']!=batch or not p['tracing']:continue
            name=model+'-b'+str(batch)+'-traced-'+str(p['repeat'])
            for line in (jobs/name/'gpu-counters.jsonl').read_text().splitlines():
                prefix,data=line.split(' ',1);value=json.loads(data)
                if prefix=='metal_counter_command':commands+=1;continue
                assert prefix=='metal_counter_profile';samples+=1
                g=groups[value['op']];g['count']+=1;g['raw_ticks']+=value['end_raw']-value['start_raw']
                head_samples+=int(value['sampling_weight']==1)
        ranks=sorted([dict(op=op,**g,mean_raw_ticks=g['raw_ticks']/g['count']) for op,g in groups.items()],key=lambda g:g['raw_ticks'],reverse=True)
        results.append(dict(model=model,batch=batch,warm_observations=observed,traced_tps_fraction=observed['traced']['decode_tps']/observed['quiet']['decode_tps'],traced_ttft_fraction=observed['traced']['ttft_ms']/observed['quiet']['ttft_ms'],samples=samples,commands=commands,always_sampled_head_events=head_samples,raw_sample_ranking=ranks))
result=dict(scope='Sparse profiling overhead and sampled shapes across first and warm requests, prefill and decode together. Raw ticks are not milliseconds. Heads sampled every time, other operations one in 31; ranks are not unbiased serving-time shares. No fresh comparator evidence.',profiles=results,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps([{k:v for k,v in p.items() if k!='raw_sample_ranking'} for p in results],indent=2))
