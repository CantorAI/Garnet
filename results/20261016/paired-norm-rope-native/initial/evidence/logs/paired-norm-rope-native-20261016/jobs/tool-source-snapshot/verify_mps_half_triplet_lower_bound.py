"""Audit synthetic exact-plane diagnostics/bounds and complete lower-bound timing records."""
from pathlib import Path
import hashlib,json,math,statistics,sys
base=Path(sys.argv[1]).resolve();proof=json.loads((base/'evidence/manifest.json').read_text());trial='mps-half-triplet-lower-bound-20261016'
assert set(proof['jobs'])=={trial} and proof['jobs'][trial]['exit_code']==0
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
out=base/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text())
for name,sha in meta['source_sha256'].items():assert digest(out/'source-snapshot'/name)==sha
assert proof['omitted_binaries']['logs/'+trial+'/jobs/mps_half_triplet_lower_bound_probe']['sha256']==meta['binary_sha256']
stages=json.loads((out/'statuses.json').read_text());assert [s['name'] for s in stages]==['compile','probe'] and all(s['exit_code']==0 for s in stages)
for stage in stages:
    monitor=json.loads((out/(stage['name']+'-monitor.json')).read_text());assert monitor and all(r['guard_passed'] for r in monitor)
records=[json.loads(line) for line in (out/'probe.log').read_text().splitlines() if line.startswith('{')]
diagnostics=[r for r in records if r['kind']=='precision_diagnostic'];assert len(diagnostics)==3
for index,row in enumerate(diagnostics):
    k=1024 if index==2 else 1;v=2**-14 if index else 1+2**-10;expected=k*v*v
    assert row['fixture']==index and row['inner']==k and row['half_operand']==v
    assert row['expected_float']==row['actual_float']==expected and expected!=row['half_rounded_product_control'] and row['passed'] and row['exact_outputs']==8
scales=[r for r in records if r['kind']=='scale_diagnostic'];assert len(scales)==1
assert scales[0]['passed'] and scales[0]['exact_outputs']==8 and scales[0]['expected_float']==scales[0]['actual_float']==1+2**-11+2**-22
checks=[r for r in records if r['kind']=='check'];assert len(checks)==6 and checks==summary['checks']
assert {(c['model'],c['rows']) for c in checks}=={(m,r) for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for r in (80,160)}
profiles=[];timed=0
for check in checks:
    model=check['model'];rows=check['rows'];shape=meta['models'][model]
    assert rows in (80,160) and (check['inner'],check['columns'])==(shape['inner'],shape['columns'])
    config=out/'configs'/(model+'.json');assert digest(config)==shape['config_sha256'];data=json.loads(config.read_text());data=data.get('text_config',data)
    assert (data['hidden_size'],data['intermediate_size'])==(check['inner'],check['columns'])
    assert check['patterns']==4 and check['reconstructed_inputs']==rows*check['inner']*4 and check['validated_executions']==12
    assert check['source_bits_exact'] and check['padding_preserved'] and check['passed']
    assert all(math.isfinite(check[k]) and 0<=check[k]<=1 for k in ('original_fp64_worst','original_comparison_worst'))
    warm=[r for r in records if r['kind']=='warmup' and (r['model'],r['rows'])==(model,rows)];assert len(warm)==1 and warm[0]['elapsed_ms']>=2000
    assert all(warm[0][key]>0 for key in ('float_input_bytes','half_planes_bytes','half_weight_bytes','float_output_bytes'))
    for candidate in (1,2):
        windows=[r for r in records if r['kind']=='timing' and (r['model'],r['rows'],r['candidate'])==(model,rows,candidate)];assert len(windows)==40
        for rnd in range(10):
            group=sorted([r for r in windows if r['round']==rnd],key=lambda r:r['position']);assert [r['position'] for r in group]==[0,1,2,3] and [r['mode'] for r in group]==[0,candidate,candidate,0]
        assert all(len(r[key])==5 and all(math.isfinite(x) and x>=0 for x in r[key]) for r in windows for key in ('encode_ms','wall_ms','gpu_ms'));timed+=200
        means={policy:{key:statistics.mean(x for r in windows if (r['mode']==candidate)==selected for x in r[key]) for key in ('encode_ms','wall_ms','gpu_ms')} for policy,selected in (('control',False),('candidate',True))}
        p=next(r for r in summary['profiles'] if (r['model'],r['rows'],r['candidate'])==(model,rows,candidate))
        for policy in means:
            for key in means[policy]:assert abs(p['means'][policy][key]-means[policy][key])<1e-12
        assert abs(p['gpu_fraction']-means['candidate']['gpu_ms']/means['control']['gpu_ms'])<1e-12 and abs(p['wall_fraction']-means['candidate']['wall_ms']/means['control']['wall_ms'])<1e-12
        profiles.append(p)
assert len(profiles)==12 and timed==2400
result=dict(verified_files=len(proof['files']),precision_diagnostic_outputs=32,synthetic_shape_checks=6,source_reconstructed_inputs=sum(c['reconstructed_inputs'] for c in checks),timed_executions=timed,profiles=profiles,
 scope='Synthetic precomputed-plane numeric feasibility and timing only. No GPU decomposition cost, learned/actual model activation coverage or general internal accumulator dtype proof; no inference or vLLM claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
