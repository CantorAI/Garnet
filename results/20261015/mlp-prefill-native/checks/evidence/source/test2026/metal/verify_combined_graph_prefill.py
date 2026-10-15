"""Audit all-matrix combined graph bounds, first-use separation and raw wall windows."""
import hashlib,json,math,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    path=root/name;assert path.resolve().is_relative_to(root)
    assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256']
profiles=[];all_checks=[];rejected=[];first_count=0
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0
    out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text());snap=out/'source-snapshot'
    assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30
    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
    assert meta['reference_static_library_sha256']==proof['omitted']['build-coalesced-bf16-body/libgarnet_metal.a']['sha256']
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==4 and {s['name'] for s in stages}=={'compile'}|{m+'-gpu' for m in meta['assets']}
    command=next(s['command'] for s in stages if s['name']=='compile');binary=Path(command[command.index('-o')+1]).name
    assert 'MetalPerformanceShadersGraph' in command and meta['binary_sha256']==proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']
    for stage in stages:
        assert stage['exit_code']==0;monitor=json.loads((out/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
    observed=[];checks=[];first=[]
    assert set(meta['assets'])=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
    for model,assets in meta['assets'].items():
        assert assets['checkpoint_sha256']==proof['omitted']['models/'+model+'/model.safetensors']['sha256']
        assert len(assets['cases'])==(15 if model=='Qwen3.5-0.8B' else 7)
        for c in assets['cases']:assert c['sha256']['raw.bin']==proof['omitted']['logs/decoder-codec-gpu-20261013/jobs/'+model+'-assets/'+c['name']+'/raw.bin']['sha256']
        values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        assert all(v['kind'] in ('check','first_graph_run','samples') for v in values)
        quality=[v for v in values if v['kind']=='check'];setups=[v for v in values if v['kind']=='first_graph_run']
        assert len(quality)==16 and len(setups)==4
        checks.extend(dict(model=model,**v) for v in quality);first.extend(dict(model=model,**v) for v in setups)
        for batch in (1,2):
            rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch
            for policy in (1,2):
                match=lambda v:(v['batch'],v['policy'])==(batch,policy)
                relevant=[v for v in quality if match(v)];assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
                setup=[v for v in setups if match(v)];assert len(setup)==1 and setup[0]['rows']==rows and math.isfinite(setup[0]['wall_ms']) and setup[0]['wall_ms']>0
                for check in relevant:
                    assert check['rows']==rows and check['values']==rows*sum(c['rows'] for c in assets['cases']) and check['projections']==len(assets['cases'])
                    assert check['fp32_activation_weight_product_output']
                    for key in ('control_fp64_worst','fp64_worst','comparison_worst'):assert math.isfinite(check[key]) and check[key]>=0
                    assert check['control_fp64_worst']<=1 and check['passed']==(check['finite'] and check['fp64_worst']<=1 and check['comparison_worst']<=1)
                windows=[v for v in values if v['kind']=='samples' and match(v)]
                if not all(v['passed'] for v in relevant):
                    assert not windows;rejected.append(dict(trial=trial,model=model,batch=batch,policy=policy,checks=relevant));continue
                assert len(windows)==8 and all(v['rows']==rows and 'gpu_ms' not in v for v in windows)
                for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
                assert all(len(v['wall_ms'])==8 and all(math.isfinite(x) and x>0 for x in v['wall_ms']) for v in windows)
                means={name:statistics.mean(x for v in windows if v['candidate']==candidate for x in v['wall_ms']) for name,candidate in (('control',False),('candidate',True))}
                record=next(v for v in summary['profiles'] if (v['model'],v['batch'],v['policy'])==(model,batch,policy))
                assert record['wall_ms']==means and record['checks']==relevant and record['rows']==rows and abs(record['wall_fraction']-means['candidate']/means['control'])<1e-12
                observed.append(record)
    assert observed==summary['profiles'] and checks==summary['checks'] and first==summary['first_graph_runs'] and not summary['full_goal_met']
    profiles.extend(dict(trial=trial,**v) for v in observed);all_checks.extend(checks);first_count+=len(first)
result=dict(verified_files=len(proof['files']),numeric_checks=len(all_checks),first_graph_runs=first_count,rejected_profiles=rejected,timed_executions=len(profiles)*64,profiles=profiles,scope='All independent first-layer projections in one native submission or one graph call. Original FP32 tensor values; placeholder/constant weight policies. Wall clocks only, first graph use separate; no model TPS/TTFT or vLLM result.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','rejected_profiles')}))
