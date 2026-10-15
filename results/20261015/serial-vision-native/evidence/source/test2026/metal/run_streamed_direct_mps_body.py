"""One-submission real projection sets; original B1/B2, three static policies."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_direct_mps_body.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("paths=[Path(__file__),","paths=[Path(__file__),s/'run_direct_mps_body.py',s/'streamed_direct_mps_body_gpu.mm',")
needle="cpp.write_bytes(generate_harness((s/'padded_mps_body_gpu.mm').read_text()).encode())";assert source.count(needle)==1
source=source.replace(needle,"cpp.write_bytes((s/'streamed_direct_mps_body_gpu.mm').read_bytes())")
source=source.replace("out/'direct_mps_body_gpu","out/'streamed_direct_mps_body_gpu")
source=source.replace('Component only; repeatedly resident individual matrices','Component only; independent real per-layer projections in one submission, not a full model')
start=source.index("    checks=[v for v in values");source=source[:start]+'''    checks=[v for v in values if v['kind']=='check'];assert len(checks)==6
    all_checks.extend(dict(model=model,**v) for v in checks)
    for batch in (1,2):
        for policy in (1,2,3):
            c=next(v for v in checks if (v['batch'],v['policy'])==(batch,policy));assert c['projections']==len(info['cases'])
            windows=[v for v in values if v['kind']=='samples' and (v['batch'],v['policy'])==(batch,policy)]
            if not c['passed']:assert not windows;continue
            assert len(windows)==8
            for rnd in (0,1):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
            assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
            means={name:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
            profiles.append(dict(model=model,batch=batch,policy=policy,check=c,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    (out/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=profiles,checks=all_checks,full_goal_met=False),indent=2)+'\\n').encode());print('Completed streamed native MPS component',model,flush=True)
'''
exec(compile(source,__file__,'exec'))
