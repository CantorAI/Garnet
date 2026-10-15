"""Recompute all projection admissions, raw ABBA means and component ratios."""
import hashlib,json,math,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());profiles=[];checks=[];skipped=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,v in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text());snap=out/'source-snapshot'
    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==4 and {v['name'] for v in stages}=={'compile'}|{m+'-gpu' for m in meta['assets']}
    for v in stages:
        assert v['exit_code']==0;monitor=json.loads((out/(v['name']+'-monitor.json')).read_text());assert monitor and all(s['guard_passed'] for s in monitor)
    assert proof['omitted']['logs/'+trial+'/jobs/padded_mps_body_gpu']['sha256']==meta['binary_sha256']
    assert set(meta['assets'])=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
    observed=[];trialchecks=[];trialskipped=[]
    for model,assets in meta['assets'].items():
        values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        for c in assets['cases']:
            omitted=[v for v in values if v['kind']=='skipped' and v['matrix']==c['name']]
            if omitted:
                assert len(omitted)==1 and not any(v['kind'] in ('check','samples') and v['matrix']==c['name'] for v in values)
                trialskipped.extend(dict(model=model,**v) for v in omitted);continue
            for batch in (1,2):
                for physical in (4,8,16):
                    selected=[v for v in values if v['kind']=='check' and (v['matrix'],v['batch'],v['physical_rows'])==(c['name'],batch,physical)]
                    assert len(selected)==4 and {v['pattern'] for v in selected}==set(range(4));trialchecks.extend(dict(model=model,**v) for v in selected)
                    for v in selected:
                        assert v['global_half_dispatches']==1 and math.isfinite(v['control_fp64_worst']) and 0<=v['control_fp64_worst']<=1
                        assert v['passed']==(v['finite'] and v['padded_rows_zero'] and 0<=v['fp64_worst']<=1 and 0<=v['comparison_worst']<=1)
                    windows=[v for v in values if v['kind']=='samples' and (v['matrix'],v['batch'],v['physical_rows'])==(c['name'],batch,physical)]
                    if not all(v['passed'] for v in selected):assert not windows;continue
                    assert len(windows)==8
                    for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
                    assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
                    means={policy:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for policy,candidate in (('control',False),('candidate',True))}
                    record=next(v for v in summary['profiles'] if (v['model'],v['matrix'],v['batch'],v['physical_rows'])==(model,c['name'],batch,physical))
                    assert record['tensor']==c['tensor'] and record['means']==means
                    for metric in ('gpu','wall'):assert abs(record[metric+'_fraction']-means['candidate'][metric+'_ms']/means['control'][metric+'_ms'])<1e-12
                    observed.append(record)
    def identity(v):return (v['model'],v['matrix'],v['batch'],v['physical_rows'],v['pattern'])
    assert len({identity(v) for v in trialchecks})==len(trialchecks)==len(summary['checks'])
    assert observed==summary['profiles'] and {identity(v):v for v in trialchecks}=={identity(v):v for v in summary['checks']} and trialskipped==summary['skipped']
    profiles.extend(dict(trial=trial,**v) for v in observed);checks.extend(dict(trial=trial,**v) for v in trialchecks);skipped.extend(dict(trial=trial,**v) for v in trialskipped)
aggregates=[]
for key in sorted({(v['trial'],v['model'],v['batch'],v['physical_rows']) for v in profiles}):
    rows=[v for v in profiles if (v['trial'],v['model'],v['batch'],v['physical_rows'])==key]
    aggregates.append(dict(trial=key[0],model=key[1],batch=key[2],physical_rows=key[3],matrices=len(rows),**{metric+'_sum_fraction':sum(v['means']['candidate'][metric+'_ms'] for v in rows)/sum(v['means']['control'][metric+'_ms'] for v in rows) for metric in ('gpu','wall')}))
result=dict(verified_files=len(proof['files']),numeric_checks=len(checks),failed_numeric_checks=sum(not v['passed'] for v in checks),admitted_profiles=len(profiles),timed_executions=len(profiles)*64,skipped=skipped,aggregates=aggregates,profiles=profiles,scope='Resident individual native-context body components; input padding outside timing. Sum fractions are sums of individual measured means, not full-model performance. No serving or fresh vLLM result.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','aggregates')}))
