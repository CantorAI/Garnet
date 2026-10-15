"""Independently audit captured native logits, every rejection, source and raw timings."""
import hashlib,json,math,runpy,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());profiles=[];skipped=[];roundtrips=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,v in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
for trial,status in proof['jobs'].items():
    out=root/'evidence/logs'/trial/'jobs'
    if status['exit_code']:
        assert not out.exists();skipped.append(dict(trial=trial,exit_code=status['exit_code'],executed_queries=0));continue
    meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text());snap=out/'source-snapshot'
    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
    native=(snap/'src/backends/metal/metal_context.mm').read_text();raw=native[native.index('struct BF16Params'):native.index('// BF16 weights are expanded')];argmax=native[native.index('struct ArgmaxParams'):native.index('struct Params { ulong rows')]
    bits=next(b for b in (4,6,8) if 'Derived INT'+str(b) in meta['precision'])
    screen=(snap/'test2026/metal/screened_head.metal').read_text()
    if bits!=8:screen=runpy.run_path(str(snap/'test2026/metal'/('make_screened_head_int'+str(bits)+'.py')))['generate_shader'](screen)
    expected='#include <metal_stdlib>\nusing namespace metal;\n'+raw+argmax+screen
    assert (out/'screened-head.metal').read_bytes()==expected.encode() and digest(out/'screened-head.metal')==meta['shader_sha256']
    module='make_screened_head_int'+str(bits)+'.py' if bits!=8 else 'make_screened_head_harness.py'
    generate=runpy.run_path(str(snap/'test2026/metal'/module))['generate_harness' if bits!=8 else 'generate'];harness=generate((snap/'test2026/metal/pruned_head_gpu.mm').read_text())
    assert (out/'screened_head_gpu.mm').read_bytes()==harness.encode() and digest(out/'screened_head_gpu.mm')==meta['harness_sha256']
    for marker in ('Raw kernel differs from actual native full logits','Retained logit bits changed','Pruned logit can beat the seed','Pruned greedy prediction differs or is invalid'):assert marker in harness
    assert proof['omitted']['logs/'+trial+'/jobs/screened_head_gpu']['sha256']==meta['binary_sha256']
    reused=all((out/(m+'-assets/reuse.json')).exists() for m in meta['assets'])
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==(4 if reused else 7) and all(v['exit_code']==0 for v in stages)
    assert {v['name'] for v in stages}=={'compile'}|{m+suffix for m in meta['assets'] for suffix in (('-gpu',) if reused else ('-prepare','-gpu'))}
    for stage in stages:
        monitor=json.loads((out/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
    count=0
    assert set(meta['assets'])=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
    for model,assets in meta['assets'].items():
        folder=out/(model+'-assets');assert json.loads((folder/'manifest.json').read_text())==assets
        assert assets['inner'] in (896,1024) and len(assets['queries'])==17
        assert [sum(q['batch']==b and q['step']>=0 for q in assets['queries']) for b in (1,2)]==[8,8]
        assert sum(q['batch']==2 and q['step']==-1 for q in assets['queries'])==1
        for name,sha in [('raw.bin',assets['raw_sha256']),('approximate.bin',assets['approximate_sha256']),('remaining.bin',assets['remaining_norm_sha256'])]:assert proof['omitted']['logs/'+trial+'/jobs/'+model+'-assets/'+name]['sha256']==sha
        assert proof['omitted']['logs/'+trial+'/jobs/'+model+'-assets/approximate.bin']['bytes']==assets['rows']*assets['inner']*bits//8
        for q in assets['queries']:
            assert len(q['predictions'])==q['batch']
            for suffix,field in (('.input.bin','input_sha256'),('.reference.bin','reference_sha256')):assert proof['omitted']['logs/'+trial+'/jobs/'+model+'-assets/'+q['name']+suffix]['sha256']==q[field]
        records=[json.loads(v) for v in (out/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')];checks=[v for v in records if v['kind']=='check'];samples=[v for v in records if v['kind']=='samples'];assert len(checks)==17 and len(samples)==136
        if bits==6:
            decoded=[v for v in records if v['kind']=='index_roundtrip'];assert len(decoded)==1 and decoded[0]['all_values_exact'] and decoded[0]['bits']==6 and decoded[0]['elements']==assets['rows']*assets['inner']
            roundtrips.append(dict(trial=trial,model=model,**decoded[0]))
        assert {v['query'] for v in checks}=={v['name'] for v in assets['queries']}
        for c in checks:
            q=next(q for q in assets['queries'] if q['name']==c['query']);assert c['batch']==q['batch']
            assert c['raw_native_bits_exact'] and c['retained_bits_exact'] and c['greedy_exact'] and c['retained_rows']+c['pruned_rows']==c['batch']*assets['rows']
            assert 0<c['counted_weight_byte_fraction']<=1.6
            windows=[v for v in samples if v['query']==c['query']];assert len(windows)==8
            for rnd in (0,1):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
            assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
            means={policy:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for policy,candidate in (('control',False),('candidate',True))}
            p=next(v for v in summary['profiles'] if v['model']==model and v['check']['query']==c['query']);assert p['check']==c and p['means']==means
            for metric in ('gpu','wall'):assert abs(p[metric+'_fraction']-means['candidate'][metric+'_ms']/means['control'][metric+'_ms'])<1e-12
            profiles.append(dict(trial=trial,screen_index_bits=bits,**p));count+=1
    assert count==len(summary['profiles'])==51
result=dict(verified_files=len(proof['files']),successful_trials=len({p['trial'] for p in profiles}),query_checks=len(profiles),row_state_checks=sum(p['check']['batch'] for p in profiles),timed_executions=len(profiles)*64,complete_gpu_index_roundtrips=roundtrips,skipped_dependency_jobs=skipped,profiles=profiles,scope='Captured-query greedy component only; derived integer screening index plus original BF16/FP32 refinement. Every sampled rejected logit is below actual seed, retained bits and greedy result match. Not certified for all future inputs; no full-model or vLLM speed claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
