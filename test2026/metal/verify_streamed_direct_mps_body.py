"""Audit original-sized streamed projections, policies, precision and raw ABBA."""
import hashlib,json,math,runpy,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());profiles=[];all_checks=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,v in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
for name,sha in json.loads((root/'evidence/source-manifest.json').read_text()).items():assert digest(root/'evidence/source'/name)==sha
reference=json.loads((root/'evidence/reference-direct-summary.json').read_text());reference_meta=json.loads((root/'evidence/reference-direct-metadata.json').read_text())
omitted={(v['model'],v['matrix']) for v in reference['skipped']}
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text());snap=out/'source-snapshot'
    assert meta['assets']==reference_meta['assets']
    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
    sys.path.insert(0,str(snap/'test2026/metal'))
    factory='make_mps_fastencode_body.py' if 'test2026/metal/make_mps_fastencode_body.py' in meta['source_sha256'] else 'make_direct_mps_body.py'
    module=runpy.run_path(str(snap/'test2026/metal'/factory));expected=module['context']((snap/'src/backends/metal/metal_context.mm').read_text()).encode()
    assert (out/'metal_context_direct.mm').read_bytes()==expected and digest(out/'metal_context_direct.mm')==meta['context_generated_sha256']
    assert '-DGARNET_METAL_EXACT_HALF_PREFILL=1' in meta['context_macro_defines'] and '-DGARNET_METAL_GLOBAL_HALF_ALPHA=1' in meta['context_macro_defines']
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==4 and {v['name'] for v in stages}=={'compile'}|{m+'-gpu' for m in meta['assets']}
    command=next(v['command'] for v in stages if v['name']=='compile');binary=Path(command[command.index('-o')+1]).name
    cpp=next(out/Path(v).name for v in command if v.endswith('_gpu.mm'))
    assert cpp.read_bytes()==(snap/'test2026/metal/streamed_direct_mps_body_gpu.mm').read_bytes() and digest(cpp)==meta['harness_generated_sha256']
    assert proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']==meta['binary_sha256'] and all(v in command for v in meta['context_macro_defines'])
    for v in stages:
        assert v['exit_code']==0;monitor=json.loads((out/(v['name']+'-monitor.json')).read_text());assert monitor and all(s['guard_passed'] for s in monitor)
    observed=[];checks=[]
    for model,assets in meta['assets'].items():
        values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        selected=[v for v in values if v['kind']=='check'];assert len(selected)==6
        checks.extend(dict(model=model,**v) for v in selected)
        for batch in (1,2):
            for policy in (1,2,3):
                c=next(v for v in selected if (v['batch'],v['policy'])==(batch,policy))
                forced=sum((model,v['name']) not in omitted and (policy==1 or policy==2 and '.mlp.' in v['tensor'] or policy==3 and v['rows']*v['inner']>=1024*1024) for v in assets['cases'])
                assert c['projections']==len(assets['cases']) and c['forced_mps_projections']==forced and c['values']==batch*sum(v['rows'] for v in assets['cases'])
                assert math.isfinite(c['raw_fp64_worst']) and 0<=c['raw_fp64_worst']<=1
                assert c['passed']==(c['finite'] and math.isfinite(c['fp64_worst']) and 0<=c['fp64_worst']<=1 and math.isfinite(c['comparison_worst']) and 0<=c['comparison_worst']<=1)
                windows=[v for v in values if v['kind']=='samples' and (v['batch'],v['policy'])==(batch,policy)]
                if not c['passed']:assert not windows;continue
                assert len(windows)==8
                for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
                assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
                means={name:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
                record=next(v for v in summary['profiles'] if (v['model'],v['batch'],v['policy'])==(model,batch,policy));assert record['means']==means and record['check']==c
                for metric in ('gpu','wall'):assert abs(record[metric+'_fraction']-means['candidate'][metric+'_ms']/means['control'][metric+'_ms'])<1e-12
                observed.append(record)
    assert observed==summary['profiles'] and checks==summary['checks']
    profiles.extend(dict(trial=trial,**v) for v in observed);all_checks.extend(dict(trial=trial,**v) for v in checks)
result=dict(verified_files=len(proof['files']),numeric_checks=len(all_checks),failed_numeric_checks=sum(not v['passed'] for v in all_checks),admitted_profiles=len(profiles),timed_executions=len(profiles)*64,profiles=profiles,scope='Independent trained per-layer projection sets in one command submission. Original logical B1/B2, no prefix/answer reuse; not full-model generation or vLLM performance.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
