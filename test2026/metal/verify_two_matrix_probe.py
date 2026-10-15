"""Audit every originally selected projection, exact math and raw timing windows."""
from pathlib import Path
import hashlib,json,math,statistics,sys

p=Path(__file__).with_name('verify_two_body.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
source=ns['source'].replace('make_two_body','make_two_matrix_probe').replace('coalesced_two_body_gpu','two_matrix_probe_gpu')
cut=source.index("    for model,assets in meta['assets'].items():")
prefix=source[:cut]
old="sys.modules.pop('make_prefill_tile_shader',None)";assert prefix.count(old)==1
prefix=prefix.replace(old,"[sys.modules.pop(name,None) for name in ('make_prefill_tile_shader','make_two_matrix_probe','make_two_body','make_coalesced_eight_body')]")
exec(compile(prefix,__file__,'exec'))

observed=[];rejections=[];numeric_count=0;sample_count=0;selected_matrices=0
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0
    out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    actual_profiles=[]
    assert set(meta['assets'])=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
    for model,assets in meta['assets'].items():
        assert len(assets['cases'])==(15 if model=='Qwen3.5-0.8B' else 7)
        assert proof['omitted']['models/'+model+'/model.safetensors']['sha256']==assets['checkpoint_sha256']
        cpu=next(v for v in admission['models'] if v['model']==model)
        assert cpu['checkpoint_sha256']==assets['checkpoint_sha256']
        records={v['name']:v for v in cpu['cases']}
        values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        assert all(v['kind'] in ('controls','packing','check','samples') for v in values)
        assert [v for v in values if v['kind']=='controls']==[dict(kind='controls',profiles=63,passed=True)]
        packing={v['matrix']:v for v in values if v['kind']=='packing'}
        assert set(packing)=={c['name'] for c in assets['cases']}
        for case in assets['cases']:
            path='logs/decoder-codec-gpu-20261013/jobs/'+model+'-assets/'+case['name']+'/raw.bin'
            assert proof['omitted'][path]['sha256']==case['sha256']['raw.bin']
            packed=packing[case['name']]
            if case['name'] not in records:
                assert packed==dict(kind='packing',matrix=case['name'],blocks=0,compact_blocks=0,selected=False)
                continue
            ref=records[case['name']]
            assert ref['raw_sha256']==case['sha256']['raw.bin'] and ref['cpu_original_bits_exact']
            assert (ref['rows'],ref['inner'])==(case['rows'],case['inner'])
            assert packed==dict(kind='packing',matrix=case['name'],blocks=ref['blocks'],compact_blocks=ref['compact_blocks'],selected=True)
        selected={name for name,item in packing.items() if item['selected']};assert selected==set(records)
        checks=[v for v in values if v['kind']=='check']
        assert len(checks)==len(selected)*24 and {v['matrix'] for v in checks}==selected
        selected_matrices+=len(selected);numeric_count+=len(checks)
        for case in assets['cases']:
            if case['name'] not in selected:continue
            for batch in (1,2):
                for policy in (1,2,3):
                    relevant=[v for v in checks if (v['matrix'],v['batch'],v['policy'])==(case['name'],batch,policy)]
                    assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
                    for check in relevant:
                        assert check['rows']==batch and check['projections']==check['selected_projections']==1
                        assert check['values']==batch*case['rows']
                        for key in ('control_fp64_worst','fp64_worst','comparison_worst'):
                            assert math.isfinite(check[key]) and check[key]>=0
                        assert check['control_fp64_worst']<=1
                        assert check['passed']==(check['finite'] and check['bitwise_equal'] and check['fp64_worst']<=1 and check['comparison_worst']<=1)
                        if check['bitwise_equal']:assert check['comparison_worst']==0
                    windows=[v for v in values if v['kind']=='samples' and (v['matrix'],v['batch'],v['policy'])==(case['name'],batch,policy)]
                    if not all(v['passed'] for v in relevant):
                        assert not windows
                        rejections.append(dict(trial=trial,model=model,matrix=case['name'],batch=batch,policy=policy,checks=relevant))
                        continue
                    assert len(windows)==8
                    for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
                    assert all(v['rows']==batch for v in windows)
                    assert all(len(v[k])==8 and all(math.isfinite(t) and t>0 for t in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
                    sample_count+=64
                    means={name:{key:statistics.mean(t for v in windows if v['candidate']==candidate for t in v[key]) for key in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
                    record=dict(model=model,matrix=case['name'],output_rows=case['rows'],inner=case['inner'],batch=batch,policy=policy,checks=relevant,means=means,
                        gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms'])
                    actual_profiles.append(record);observed.append(dict(trial=trial,**record))
    assert actual_profiles==summary['profiles']
result=dict(verified_files=len(proof['files']),selected_original_matrices=selected_matrices,numeric_checks=numeric_count,
    safety_profiles=len(proof['jobs'])*189,timed_executions=sample_count,profiles=observed,rejected_profiles=rejections,
    scope='Per-selected-matrix independent repeated-resident probes; one projection per timed submission, original exact bits/bounds/safety/packing proof. Warm-cache effects and submission overhead differ from full inference. No model gain or vLLM ratio.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','rejected_profiles')}))
