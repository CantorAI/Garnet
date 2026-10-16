"""Audit frozen native NDArray-prefill generation, original bounds and raw ABBA samples."""
import hashlib,json,math,runpy,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
proof=json.loads((root/'evidence/manifest.json').read_text())
profiles=[];all_checks=[];rejected=[]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    path=root/name
    assert path.resolve().is_relative_to(root)
    assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256']
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0
    out=root/'evidence/logs'/trial/'jobs'
    meta=json.loads((out/'metadata.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    snap=out/'source-snapshot'
    assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string']
    assert int(meta['hardware']['hw.memsize'])==16*2**30
    for name,sha in meta['source_sha256'].items():
        assert digest(snap/name)==sha
    sys.path.insert(0,str(snap/'test2026/metal'))
    sys.modules.pop('make_prefill_tile_shader',None)
    factory=runpy.run_path(str(snap/'test2026/metal/make_ndarray_prefill.py'))
    for file,key,operation,inputfile in [
        ('metal_context_ndarray_prefill.mm','context_generated_sha256','context','src/backends/metal/metal_context.mm'),
        ('metal_context.h','private_header_sha256','header','src/backends/metal/metal_context.h')]:
        assert (out/file).read_bytes()==factory[operation]((snap/inputfile).read_text()).encode()
        assert digest(out/file)==meta[key]
    sys.path.pop(0)
    harness=out/'ndarray_prefill_gpu.mm'
    assert harness.read_bytes()==factory['harness']((snap/'test2026/metal/packed_body_prefill_gpu.mm').read_text()).encode()
    assert digest(harness)==meta['harness_generated_sha256']
    for name in ('EXACT_HALF_PREFILL','GLOBAL_HALF_ALPHA','INTERLEAVED_BF16_BODY'):
        assert '-DGARNET_METAL_'+name+'=1' in meta['context_macro_defines']
    for name,sha in meta['sdk_headers'].items():assert digest(out/name)==sha
    stages=json.loads((out/'statuses.json').read_text())
    assert len(stages)==4 and {s['name'] for s in stages}=={'compile'}|{m+'-gpu' for m in meta['assets']}
    command=next(s['command'] for s in stages if s['name']=='compile')
    binary=Path(command[command.index('-o')+1]).name
    assert proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']==meta['binary_sha256']
    for stage in stages:
        assert stage['exit_code']==0
        monitor=json.loads((out/(stage['name']+'-monitor.json')).read_text())
        assert monitor and all(v['guard_passed'] for v in monitor)
    observed=[];checks=[]
    assert set(meta['assets'])=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
    for model,assets in meta['assets'].items():
        assert proof['omitted']['models/'+model+'/model.safetensors']['sha256']==assets['checkpoint_sha256']
        assert len(assets['cases'])==(15 if model=='Qwen3.5-0.8B' else 7)
        for case in assets['cases']:
            assert proof['omitted']['logs/decoder-codec-gpu-20261013/jobs/'+model+'-assets/'+case['name']+'/raw.bin']['sha256']==case['sha256']['raw.bin']
        values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        selected=[v for v in values if v['kind']=='check']
        assert len(selected)==24 and all(v['kind'] in ('check','samples') for v in values)
        checks.extend(dict(model=model,**v) for v in selected)
        suffix='.self_attn.k_proj.weight'
        admissions=proof['global_half_cases'][trial][model]
        assert len(admissions)==len(assets['cases'])
        for c,a in zip(assets['cases'],admissions):
            assert a['name']==c['name'] and a['raw_sha256']==c['sha256']['raw.bin']
        half_eligible=sum(v['global_half_ready'] for v in admissions)
        float_eligible=sum(not(c['tensor'].startswith('model.layers.') and c['tensor'].endswith(suffix)) for c in assets['cases'])
        assert half_eligible>0 and float_eligible>0
        for batch in (1,2):
            rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch
            for policy in (1,2,3):
                eligible={1:half_eligible,2:float_eligible,3:0}[policy]
                relevant=[v for v in selected if (v['batch'],v['policy'])==(batch,policy)]
                assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
                for check in relevant:
                    assert check['rows']==rows and check['projections']==len(assets['cases'])
                    assert check['selected_projections']==check['actual_ndarray_dispatches']==eligible and check['values']==rows*sum(c['rows'] for c in assets['cases'])
                    for key in ('control_fp64_worst','fp64_worst','comparison_worst'):
                        assert math.isfinite(check[key]) and check[key]>=0
                    assert check['control_fp64_worst']<=1
                    assert check['passed']==(check['finite'] and check['fp64_worst']<=1 and check['comparison_worst']<=1 and (policy!=3 or check['bitwise_equal']))
                    if policy==3:assert check['bitwise_equal'] and check['comparison_worst']==0
                windows=[v for v in values if v['kind']=='samples' and (v['batch'],v['policy'])==(batch,policy)]
                if not all(v['passed'] for v in relevant):
                    assert not windows
                    rejected.append(dict(trial=trial,model=model,batch=batch,policy=policy,checks=relevant))
                    continue
                assert len(windows)==8 and all(v['rows']==rows for v in windows)
                for rnd in (0,1):
                    assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
                assert all(len(v[key])==8 and all(math.isfinite(x) and x>0 for x in v[key]) for v in windows for key in ('gpu_ms','wall_ms'))
                means={name:{key:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[key]) for key in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
                record=next(v for v in summary['profiles'] if (v['model'],v['batch'],v['policy'])==(model,batch,policy))
                assert record['means']==means and record['checks']==relevant and record['physical_rows']==rows
                for metric in ('gpu','wall'):
                    assert abs(record[metric+'_fraction']-means['candidate'][metric+'_ms']/means['control'][metric+'_ms'])<1e-12
                observed.append(record)
    assert observed==summary['profiles'] and checks==summary['checks'] and not summary['full_goal_met']
    profiles.extend(dict(trial=trial,**v) for v in observed)
    all_checks.extend(dict(trial=trial,**v) for v in checks)
result=dict(verified_files=len(proof['files']),numeric_checks=len(all_checks),rejected_profiles=rejected,timed_executions=len(profiles)*64,profiles=profiles,scope='Private independent first-layer projections; original FP32 activation and accumulation, original BF16 values, unchanged numerical bounds. Mixed three-source alpha/zero addend or exact F32 weight expansion; virtual transpose and typed aliases regenerated. Cache ownership implementation is captured, not a synthetic ownership-suite proof. F32 cache adds storage. No synthetic ownership suite, complete-model or vLLM speed claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({key:value for key,value in result.items() if key not in ('profiles','rejected_profiles')}))
