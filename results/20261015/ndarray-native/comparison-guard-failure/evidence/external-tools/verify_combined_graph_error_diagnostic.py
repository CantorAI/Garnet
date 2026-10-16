"""Independently reconcile per-matrix errors with unchanged graph aggregate gates."""
import hashlib,json,math,sys
from pathlib import Path

root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,item in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root)
    assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
checks=[];details=[];first=[];failures=[]
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0
    out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30
    assert 'No performance measurements.' in meta['scope']
    for name,sha in meta['source_sha256'].items():assert digest(out/'source-snapshot'/name)==sha
    assert meta['generated_cpp_sha256']==digest(out/'combined_graph_error_diagnostic_gpu.mm')
    assert meta['reference_static_library_sha256']==proof['omitted']['build-coalesced-bf16-body/libgarnet_metal.a']['sha256']
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==5
    assert {v['name'] for v in stages}=={'harness','compile'}|{m+'-gpu' for m in meta['assets']}
    for stage in stages:
        assert stage['exit_code']==0
        monitor=json.loads((out/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
    command=next(v['command'] for v in stages if v['name']=='compile');binary=Path(command[command.index('-o')+1]).name
    assert 'MetalPerformanceShadersGraph' in command
    assert meta['binary_sha256']==proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']
    tc=[];td=[];tf=[]
    assert set(meta['assets'])=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
    for model,asset in meta['assets'].items():
        assert asset['checkpoint_sha256']==proof['omitted']['models/'+model+'/model.safetensors']['sha256']
        cases=asset['cases'];assert len(cases)==(15 if model=='Qwen3.5-0.8B' else 7)
        for case in cases:
            name='logs/decoder-codec-gpu-20261013/jobs/'+model+'-assets/'+case['name']+'/raw.bin'
            assert case['sha256']['raw.bin']==proof['omitted'][name]['sha256']
        values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        assert all(v['kind'] in ('projection_check','check','first_graph_run') for v in values)
        aggregate=[v for v in values if v['kind']=='check'];part=[v for v in values if v['kind']=='projection_check'];setup=[v for v in values if v['kind']=='first_graph_run']
        assert len(aggregate)==16 and len(part)==16*len(cases) and len(setup)==4
        for batch in (1,2):
            rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch
            for policy in (1,2):
                initial=[v for v in setup if (v['batch'],v['policy'])==(batch,policy)]
                assert len(initial)==1 and initial[0]['rows']==rows and math.isfinite(initial[0]['wall_ms']) and initial[0]['wall_ms']>0
                for pattern in range(4):
                    matches=lambda v:(v['batch'],v['policy'],v['pattern'])==(batch,policy,pattern)
                    a=[v for v in aggregate if matches(v)];p=[v for v in part if matches(v)]
                    assert len(a)==1 and len(p)==len(cases) and {v['matrix'] for v in p}==set(range(len(cases)))
                    a=a[0];assert a['rows']==rows and a['projections']==len(cases) and a['fp32_activation_weight_product_output']
                    assert a['values']==rows*sum(c['rows'] for c in cases)
                    for v in p:
                        case=cases[v['matrix']]
                        assert v['tensor']==case['tensor'] and v['columns']==case['rows'] and v['inner']==case['inner'] and v['rows']==rows and v['values']==rows*case['rows']
                        for key in ('control_fp64_worst','fp64_worst','comparison_worst'):assert math.isfinite(v[key]) and v[key]>=0
                        assert v['control_fp64_worst']<=1
                        assert v['passed']==(v['finite'] and v['fp64_worst']<=1 and v['comparison_worst']<=1)
                        assert 0<=v['worst_index']<v['values']
                        bound=5e-6+5e-5*abs(v['control_at_worst'])
                        assert math.isclose(v['comparison_bound_at_worst'],bound,rel_tol=1e-12,abs_tol=1e-15)
                        ratio=abs(v['candidate_at_worst']-v['control_at_worst'])/bound
                        assert math.isclose(v['comparison_worst'],ratio,rel_tol=1e-9,abs_tol=1e-9)
                        assert math.isfinite(v['fp64_at_worst']) and v['fp64_bound_at_worst']>0
                        if not v['passed']:failures.append(dict(trial=trial,model=model,**v))
                    for key in ('control_fp64_worst','fp64_worst','comparison_worst'):assert math.isclose(a[key],max(v[key] for v in p),rel_tol=1e-12,abs_tol=1e-15)
                    assert a['finite']==all(v['finite'] for v in p) and a['passed']==all(v['passed'] for v in p)
        tc.extend(dict(model=model,**v) for v in aggregate);td.extend(dict(model=model,**v) for v in part);tf.extend(dict(model=model,**v) for v in setup)
    assert summary['checks']==tc and summary['per_projection']==td and summary['first_graph_runs']==tf and not summary['full_goal_met']
    checks.extend(tc);details.extend(td);first.extend(tf)
result=dict(verified_files=len(proof['files']),aggregate_checks=len(checks),per_projection_checks=len(details),first_graph_calls=len(first),failed_projection_checks=failures,timed_executions=0,scope='Unchanged combined-graph numerical diagnosis only; original FP64 and native comparison bounds, no routing/tolerance changes or speed claims.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='failed_projection_checks'}))
