"""Audit unchanged bounds, source, full shape coverage and raw ABBA observations."""
import hashlib,json,math,runpy,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();manifest=json.loads((root/'evidence/manifest.json').read_text());profiles=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,v in manifest['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
for trial,status in manifest['jobs'].items():
    assert status['exit_code']==0;out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text())
    for name,sha in meta['source_sha256'].items():assert digest(out/'source-snapshot'/name)==sha
    generate=runpy.run_path(str(out/'source-snapshot/test2026/metal/make_parallel_vision_context.py'))['generate']
    shader,harness=generate((out/'source-snapshot/src/backends/metal/metal_context.mm').read_text(),(out/'source-snapshot/test2026/metal/vision_fused_context_gpu.mm').read_text())
    assert (out/'parallel-vision.metal').read_bytes()==shader.encode() and digest(out/'parallel-vision.metal')==meta['shader_sha256']
    assert (out/'parallel_vision_context_gpu.mm').read_bytes()==harness.encode() and digest(out/'parallel_vision_context_gpu.mm')==meta['harness_sha256']
    assert 'Original candidate/control envelope exceeded' in harness and 'if(worst>1||controlWorst>1)' in harness
    assert manifest['omitted_binaries']['logs/'+trial+'/jobs/parallel_vision_context_gpu']['sha256']==meta['binary_sha256']
    stages=json.loads((out/'statuses.json').read_text());assert {v['name'] for v in stages}=={'compile','vision-gpu'} and len(stages)==2
    for v in stages:
        assert v['exit_code']==0;monitor=json.loads((out/(v['name']+'-monitor.json')).read_text());assert monitor and all(p['guard_passed'] for p in monitor)
    records=[json.loads(v) for v in (out/'vision-gpu.log').read_text().splitlines() if v.startswith('{')];checks=[v for v in records if v['kind']=='check'];samples=[v for v in records if v['kind']=='samples']
    key=lambda v:tuple(v[k] for k in ('batch','tokens','heads','dim'))
    assert len(checks)==24 and {key(v) for v in checks}=={(b,t,h,d) for b in (1,2) for t in (4,24,216) for h in (2,12) for d in (8,64)} and len(samples)==192 and len(summary['profiles'])==24
    for c in checks:
        assert c['generations']==4 and c['fp64_worst_ratio']<=1 and c['control_fp64_worst_ratio']<=1 and math.isfinite(c['max_abs_error'])
        windows=[v for v in samples if key(v)==key(c)];assert len(windows)==8
        for rnd in (0,1):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
        assert all(len(v[k])==20 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
        means={policy:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for policy,candidate in (('control',False),('candidate',True))}
        p=next(v for v in summary['profiles'] if key(v['check'])==key(c));assert p['check']==c and p['means']==means
        for metric in ('gpu','wall'):assert abs(p[metric+'_fraction']-means['candidate'][metric+'_ms']/means['control'][metric+'_ms'])<1e-12
        profiles.append(dict(trial=trial,**p))
result=dict(verified_files=len(manifest['files']),trials=len(manifest['jobs']),shape_checks=len(profiles),timed_executions=len(manifest['jobs'])*3840,profiles=profiles,scope='Vision contraction component only; altered FP32 reduction order within original tolerances, no full-image or generated-output validation yet',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
