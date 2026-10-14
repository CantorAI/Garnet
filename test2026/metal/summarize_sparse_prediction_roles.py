"""Align always-sampled heads with complete request records; no missing-shape inference."""
import collections,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();jobs=root/'logs/counter-trace-sparse-profiles-20261015/jobs'
profiles=json.loads((jobs/'profiles.json').read_text())['profiles'];results=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    config=json.loads((root/'models'/model/'config.json').read_text())
    config=config.get('text_config',config)
    for batch in (1,2):
        cohorts={name:dict(commands=0,head_events=0,samples=0,ranking=collections.defaultdict(lambda:dict(count=0,raw_ticks=0))) for name in ('first_decode','warm_decode','first_final_prefill','warm_final_prefill')}
        workers=[]
        for p in profiles:
            if p['model']!=model or p['batch']!=batch or not p['tracing']:continue
            name=model+'-b'+str(batch)+'-traced-'+str(p['repeat'])
            events=[json.loads(line.split(' ',1)[1]) for line in (jobs/name/'gpu-counters.jsonl').read_text().splitlines() if line.startswith('metal_counter_profile ')]
            heads=[e for e in events if e['op'].startswith('bf16:') and int(e['op'].split(':')[-1])>100000]
            assert heads and len({e['context'] for e in heads})==1
            assert all(e['sampling_weight']==1 and int(e['op'].split(':')[1])==batch for e in heads)
            assert all(e['op']=='bf16:'+str(batch)+':'+str(config['hidden_size'])+':'+str(config['vocab_size']) for e in heads)
            assert len({(e['context'],e['command']) for e in heads})==len(heads)
            heads.sort(key=lambda e:e['command']);records=sorted(p['records'],key=lambda r:r['request'])
            assert [r['request'] for r in records]==[0,1,2]
            assert len(heads)==sum(r['decode_steps_after_first']+1 for r in records)
            by_command=collections.defaultdict(list)
            for e in events:by_command[(e['context'],e['command'])].append(e)
            offset=0;assignments=[]
            for r in records:
                count=r['decode_steps_after_first']+1;block=heads[offset:offset+count];offset+=count
                assert len(block)==count
                for prediction,e in enumerate(block):
                    role=('warm_' if r['request']>0 else 'first_')+('decode' if prediction else 'final_prefill')
                    cohort=cohorts[role];cohort['commands']+=1;cohort['head_events']+=1
                    selected=by_command[(e['context'],e['command'])];cohort['samples']+=len(selected)
                    # Decode single-token calls have rows at most batch. Unlike
                    # the old classifier, missing large shapes never select a role.
                    if prediction:
                        for sample in selected:
                            if sample['op'].startswith('bf16:'):assert int(sample['op'].split(':')[1])==batch
                    for sample in selected:
                        g=cohort['ranking'][sample['op']];g['count']+=1;g['raw_ticks']+=sample['end_raw']-sample['start_raw']
                assignments.append(dict(request=r['request'],prediction_calls=count,first_head_command=block[0]['command'],last_head_command=block[-1]['command']))
            workers.append(dict(name=name,head_context=heads[0]['context'],assignments=assignments))
        assert len(workers)==2
        for cohort in cohorts.values():
            cohort['ranking']=sorted([dict(op=op,**g,mean_raw_ticks=g['raw_ticks']/g['count']) for op,g in cohort['ranking'].items()],key=lambda g:g['raw_ticks'],reverse=True)
        results.append(dict(model=model,batch=batch,workers=workers,cohorts=cohorts))
result=dict(scope='Inference from source-guaranteed one BF16 vocabulary head per prediction and exact complete-request call counts, verified against all forced-head records. No explicit runtime phase marker. Captures selected-global-alpha diagnostic, not the new fused route. Raw ticks and sampled ranks are not milliseconds or unbiased end-to-end time shares; intervals may overlap and heads are oversampled. Earlier prefill chunks and vision commands without a vocabulary head are excluded.',profiles=results,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps([dict(model=p['model'],batch=p['batch'],warm_decode_commands=p['cohorts']['warm_decode']['commands'],warm_final_prefill_commands=p['cohorts']['warm_final_prefill']['commands'],top_warm_decode_samples=p['cohorts']['warm_decode']['ranking'][:4]) for p in results],indent=2))
