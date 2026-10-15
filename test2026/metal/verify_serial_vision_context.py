"""Audit bitwise admission, fixed bounds, original shapes and every raw ABBA mean."""
import hashlib,json,math,runpy,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());profiles=[];checks=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,v in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text());snap=out/'source-snapshot'
    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
    generate=runpy.run_path(str(snap/'test2026/metal/make_serial_vision_context.py'))['generate']
    shader,cpp=generate((snap/'src/backends/metal/metal_context.mm').read_text(),(snap/'test2026/metal/vision_fused_context_gpu.mm').read_text())
    assert (out/'serial-vision.metal').read_bytes()==shader.encode() and digest(out/'serial-vision.metal')==meta['shader_sha256']
    assert (out/'serial_vision_context_gpu.mm').read_bytes()==cpp.encode() and digest(out/'serial_vision_context_gpu.mm')==meta['harness_sha256']
    assert 'strictOptions.fastMathEnabled=NO' in cpp and 'if(!admitted)continue;' in cpp and 'Padding modified' in cpp
    assert proof['omitted']['logs/'+trial+'/jobs/serial_vision_context_gpu']['sha256']==meta['binary_sha256']
    variants=meta.get('selected_variants',list(range(5)));assert variants and set(variants)<=set(range(5)) and len(set(variants))==len(variants)
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==len(variants)+1 and {v['name'] for v in stages}=={'compile'}|{'variant-'+str(i) for i in variants}
    for v in stages:
        assert v['exit_code']==0;monitor=json.loads((out/(v['name']+'-monitor.json')).read_text());assert monitor and all(s['guard_passed'] for s in monitor)
    observed=[];trialchecks=[];key=lambda v:tuple(v[k] for k in ('batch','tokens','heads','dim','variant'))
    for variant in variants:
        values=[json.loads(line) for line in (out/('variant-'+str(variant)+'.log')).read_text().splitlines() if line.startswith('{')]
        selected=[v for v in values if v['kind']=='check'];samples=[v for v in values if v['kind']=='samples']
        assert len(selected)==24 and {key(v) for v in selected}=={(b,t,h,d,variant) for b in (1,2) for t in (4,24,216) for h in (2,12) for d in (8,64)}
        trialchecks.extend(selected)
        for c in selected:
            assert c['generations']==4 and math.isfinite(c['max_abs_error']) and math.isfinite(c['fp64_worst_ratio'])
            assert c['passed']==(c['bits_exact'] and 0<=c['fp64_worst_ratio']<=1)
            windows=[v for v in samples if key(v)==key(c)]
            if not c['passed']:assert not windows;continue
            assert len(windows)==8
            for rnd in (0,1):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
            assert all(len(v[k])==20 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
            means={policy:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for policy,candidate in (('control',False),('candidate',True))}
            record=next(v for v in summary['profiles'] if key(v['check'])==key(c));assert record['check']==c and record['means']==means and record['label']==meta['variants'][variant]
            for metric in ('gpu','wall'):assert abs(record[metric+'_fraction']-means['candidate'][metric+'_ms']/means['control'][metric+'_ms'])<1e-12
            observed.append(record)
    assert observed==summary['profiles'] and trialchecks==summary['checks']
    profiles.extend(dict(trial=trial,**v) for v in observed);checks.extend(dict(trial=trial,**v) for v in trialchecks)
result=dict(verified_files=len(proof['files']),shape_variant_checks=len(checks),failed_admissions=sum(not v['passed'] for v in checks),admitted_profiles=len(profiles),timed_submissions=len(profiles)*160,checks=checks,profiles=profiles,scope='Synthetic original-order vision component only; admission requires full output bit identity and original FP64 bound. No pretrained vision, complete generation or vLLM speed claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','checks')}))
