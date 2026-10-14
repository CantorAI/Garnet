"""Shape ranks in raw counter units; decode-width includes final prefill heads."""
import collections,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();jobs=root/'logs/counter-trace-file-profiles-20261015/jobs'
profiles=json.loads((jobs/'profiles.json').read_text())['profiles'];ranks=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for batch in (1,2):
        groups={};commands=0;overlaps=0
        for p in profiles:
            if p['model']!=model or p['batch']!=batch or not p['tracing']:continue
            name=model+'-b'+str(batch)+'-traced-'+str(p['repeat'])
            events=[json.loads(line.split(' ',1)[1]) for line in (jobs/name/'gpu-counters.jsonl').read_text().splitlines() if line.startswith('metal_counter_profile ')]
            by=collections.defaultdict(list)
            for e in events:by[(e['context'],e['command'])].append(e)
            for values in by.values():
                # Select observed decode-width head commands. This also admits
                # a final prefill head and any one/two-token physical tail.
                shapes=[tuple(map(int,e['op'].split(':')[1:])) for e in values if e['op'].startswith('bf16:')]
                if not any(rows==batch and columns>100000 for rows,inner,columns in shapes):continue
                if any(rows>batch for rows,inner,columns in shapes):continue
                if any(e['op'].startswith('pointwise:') and int(e['op'].split(':')[2])>batch for e in values):continue
                commands+=1;last=0
                for e in sorted(values,key=lambda e:e['start_raw']):
                    if e['start_raw']<last:overlaps+=1
                    last=max(last,e['end_raw'])
                    g=groups.setdefault(e['op'],dict(count=0,raw_ticks=0));g['count']+=1;g['raw_ticks']+=e['end_raw']-e['start_raw']
        ranking=[dict(op=op,**g,mean_raw_ticks=g['raw_ticks']/g['count']) for op,g in groups.items()]
        ranking.sort(key=lambda g:g['raw_ticks'],reverse=True)
        total=sum(g['raw_ticks'] for g in ranking)
        assert total>0
        projection=sum(g['raw_ticks'] for g in ranking if g['op'].startswith('bf16:'))
        head=sum(g['raw_ticks'] for g in ranking if g['op'].startswith('bf16:') and int(g['op'].split(':')[-1])>100000)
        warm={policy:[r for p in profiles if p['model']==model and p['batch']==batch and p['tracing']==policy for r in p['records'] if r['request']>0] for policy in (False,True)}
        assert all(len(v)==4 for v in warm.values())
        observed={str(policy):dict(decode_tps=statistics.mean(r['decode_wall_steps_per_second'] for r in values),ttft_ms=statistics.mean(r['ttft_seconds'] for r in values)*1000) for policy,values in warm.items()}
        ranks.append(dict(model=model,batch=batch,decode_width_head_commands=commands,overlapping_intervals=overlaps,
            projection_fraction_of_summed_sample_intervals=projection/total,head_fraction_of_summed_sample_intervals=head/total,
            observed_instrumentation_clocks=observed,ranking=ranking))
result=dict(scope='Diagnostic raw counter units; summed samples are not a full-command clock or serving throughput attribution; decode-width includes final prefill heads and small physical tails; captures first and warm requests',profiles=ranks)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps([{k:p[k] for k in ('model','batch','decode_width_head_commands','overlapping_intervals','projection_fraction_of_summed_sample_intervals','head_fraction_of_summed_sample_intervals','observed_instrumentation_clocks')} for p in ranks],indent=2))
