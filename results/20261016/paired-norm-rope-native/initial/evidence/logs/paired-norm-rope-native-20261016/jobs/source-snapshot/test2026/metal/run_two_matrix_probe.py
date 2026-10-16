"""Per-projection two-output tuning probe; no complete-model gain claim."""
from pathlib import Path
p=Path(__file__).with_name('run_two_body.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source']
old='from make_two_body import context as generate_context,header as generate_header,harness as generate_harness,shader as generate_shader'
assert source.count(old)==1
source=source.replace(old,old.replace('make_two_body','make_two_matrix_probe'))
old="s/'make_two_body.py'";assert source.count(old)==1
source=source.replace(old,"s/'make_two_matrix_probe.py',s/'run_two_body.py',"+old)
source=source.replace('coalesced_two_body_gpu','two_matrix_probe_gpu')
source=source.replace('Private current four-output control versus two-output SIMD compute with explicit FP32 FMA over the unchanged original four-output compact payload.',
                      'Per-projection current four-output control versus two-output SIMD compute over unchanged compact bytes; every originally admitted matrix, batch1/2 and64/128/256 threads.')
start=source.index("    checks=[v for v in values")
source=source[:start]+'''    checks=[v for v in values if v['kind']=='check']
    selected={v['matrix'] for v in values if v['kind']=='packing' and v['selected']}
    assert selected and len(checks)==24*len(selected)
    all_checks.extend(dict(model=model,**v) for v in checks)
    for case in info['cases']:
        if case['name'] not in selected:continue
        for batch in (1,2):
            for policy in (1,2,3):
                relevant=[v for v in checks if (v['matrix'],v['batch'],v['policy'])==(case['name'],batch,policy)]
                assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
                windows=[v for v in values if v['kind']=='samples' and (v['matrix'],v['batch'],v['policy'])==(case['name'],batch,policy)]
                if not all(v['passed'] for v in relevant):assert not windows;continue
                assert len(windows)==8
                for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
                assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
                means={name:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
                profiles.append(dict(model=model,matrix=case['name'],output_rows=case['rows'],inner=case['inner'],batch=batch,policy=policy,checks=relevant,means=means,
                    gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    (out/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=profiles,checks=all_checks,full_goal_met=False),indent=2)+'\\n').encode())
    print('Completed individual projection probe',model,flush=True)
'''
source=repr(__doc__)+source[source.index('\n'):]
exec(compile(source,__file__,'exec'))
