"""Independent original-bits, FP64, actual route and raw timed-window audit."""
import hashlib,json,math,runpy,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());profiles=[];all_checks=[];controls=0
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text());snap=out/'source-snapshot'
    assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30
    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
    factory=runpy.run_path(str(snap/'test2026/metal/make_interleaved_bf16_body.py'))
    for file,key,operation,inputfile in [('metal_context_interleaved_body.mm','context_generated_sha256','context','src/backends/metal/metal_context.mm'),('metal_context.h','private_header_sha256','header','src/backends/metal/metal_context.h'),('interleaved_bf16_body_gpu.mm','harness_generated_sha256','harness','test2026/metal/streamed_direct_mps_body_gpu.mm')]:
        assert (out/file).read_bytes()==factory[operation]((snap/inputfile).read_text()).encode() and digest(out/file)==meta[key]
    assert '-DGARNET_METAL_EXACT_HALF_PREFILL=1' in meta['context_macro_defines'] and '-DGARNET_METAL_GLOBAL_HALF_ALPHA=1' in meta['context_macro_defines']
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==4 and {v['name'] for v in stages}=={'compile'}|{m+'-gpu' for m in meta['assets']}
    command=next(v['command'] for v in stages if v['name']=='compile');binary=Path(command[command.index('-o')+1]).name
    assert proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']==meta['binary_sha256']
    for stage in stages:
        assert stage['exit_code']==0;monitor=json.loads((out/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
    observed=[];checks=[]
    for model,assets in meta['assets'].items():
        assert proof['omitted']['models/'+model+'/model.safetensors']['sha256']==assets['checkpoint_sha256']
        for c in assets['cases']:
            assert proof['omitted']['logs/decoder-codec-gpu-20261013/jobs/'+model+'-assets/'+c['name']+'/raw.bin']['sha256']==c['sha256']['raw.bin']
        values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        safety=[v for v in values if v['kind']=='controls'];assert safety==[dict(kind='controls',profiles=4,passed=True)];controls+=4
        selected=[v for v in values if v['kind']=='check'];assert len(selected)==24
        checks.extend(dict(model=model,**v) for v in selected)
        for batch in (1,2):
            for policy in (1,2,3):
                c=next(v for v in selected if (v['batch'],v['policy'],v['pattern'])==(batch,policy,0))
                relevant=[v for v in selected if (v['batch'],v['policy'])==(batch,policy)];assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
                forced=sum(256<v['rows']<=8192 and v['inner']%4==0 and (policy==1 or policy==2 and '.mlp.' in v['tensor'] or policy==3 and v['rows']*v['inner']>=1024*1024) for v in assets['cases'])
                assert forced>0
                for check in relevant:
                    assert check['projections']==len(assets['cases']) and check['interleaved_projections']==forced and check['values']==batch*sum(v['rows'] for v in assets['cases'])
                    assert check['passed'] and check['finite'] and check['bitwise_equal'] and check['comparison_worst']==0
                    assert all(math.isfinite(check[k]) and 0<=check[k]<=1 for k in ('raw_fp64_worst','fp64_worst'))
                windows=[v for v in values if v['kind']=='samples' and (v['batch'],v['policy'])==(batch,policy)];assert len(windows)==8
                for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
                assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
                means={name:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
                record=next(v for v in summary['profiles'] if (v['model'],v['batch'],v['policy'])==(model,batch,policy));assert record['means']==means and record['check']==c
                for metric in ('gpu','wall'):assert abs(record[metric+'_fraction']-means['candidate'][metric+'_ms']/means['control'][metric+'_ms'])<1e-12
                observed.append(record)
    assert observed==summary['profiles'] and checks==summary['checks']
    profiles.extend(dict(trial=trial,**v) for v in observed);all_checks.extend(dict(trial=trial,**v) for v in checks)
result=dict(verified_files=len(proof['files']),numeric_checks=len(all_checks),control_profiles=controls,timed_executions=len(profiles)*64,profiles=profiles,scope='Isolated native independent first-layer projection sets only, original values and arithmetic; no full-model TPS/TTFT or vLLM claim',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
