"""Audit the unchanged math, current large-body control and smaller admission."""
import hashlib,json,math,runpy,statistics,sys
from pathlib import Path

root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());profiles=[];checks=[];controls=0
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,item in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root)
    assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0
    out=root/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text());snap=out/'source-snapshot'
    assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30
    for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
    factory=runpy.run_path(str(snap/'test2026/metal/make_small_body_threshold.py'))
    for name,key,op,original in [('metal_context_small_body_threshold.mm','context_generated_sha256','context','src/backends/metal/metal_context.mm'),('metal_context.h','private_header_sha256','header','src/backends/metal/metal_context.h'),('small_body_threshold_gpu.mm','harness_generated_sha256','harness','test2026/metal/coalesced_body_block_gpu.mm')]:
        assert (out/name).read_bytes()==factory[op]((snap/original).read_text()).encode()
        assert digest(out/name)==meta[key]
    for name in ('EXACT_HALF_PREFILL','GLOBAL_HALF_ALPHA','INTERLEAVED_BF16_BODY','COALESCED_BF16_BODY'):assert '-DGARNET_METAL_'+name+'=1' in meta['context_macro_defines']
    admission=json.loads((out/'admission/report.json').read_text());assert digest(out/'admission/source.py')==admission['source_sha256']==digest(snap/'test2026/metal/audit_small_body_threshold.py')
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==5 and {v['name'] for v in stages}=={'cpu-admission','compile'}|{m+'-gpu' for m in meta['assets']}
    command=next(v['command'] for v in stages if v['name']=='compile');binary=Path(command[command.index('-o')+1]).name
    assert meta['binary_sha256']==proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']
    for v in stages:
        assert v['exit_code']==0;monitor=json.loads((out/(v['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
    observed=[];quality=[]
    assert set(meta['assets'])=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
    for model,assets in meta['assets'].items():
        assert assets['checkpoint_sha256']==proof['omitted']['models/'+model+'/model.safetensors']['sha256']
        cases=assets['cases'];assert len(cases)==(15 if model=='Qwen3.5-0.8B' else 7)
        cpu=next(v for v in admission['models'] if v['model']==model);assert cpu['checkpoint_sha256']==assets['checkpoint_sha256'];cpu_cases={v['name']:v for v in cpu['cases']}
        values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        assert all(v['kind'] in ('packing','controls','check','samples') for v in values)
        assert [v for v in values if v['kind']=='controls']==[dict(kind='controls',profiles=30,passed=True)];controls+=30
        packed=[v for v in values if v['kind']=='packing'];assert len(packed)==len(cases)
        for case in cases:
            raw='logs/decoder-codec-gpu-20261013/jobs/'+model+'-assets/'+case['name']+'/raw.bin';assert case['sha256']['raw.bin']==proof['omitted'][raw]['sha256']
            p=next(v for v in packed if v['matrix']==case['name']);special=case['tensor'].startswith('model.layers.') and case['tensor'].endswith('.self_attn.k_proj.weight')
            admitted=not special and 256<case['rows']<=8192 and case['inner']%128==0 and case['rows']*case['inner']>=524288
            assert (case['name'] in cpu_cases)==admitted
            if admitted:
                c=cpu_cases[case['name']];assert c['cpu_original_bits_exact'] and c['raw_sha256']==case['sha256']['raw.bin']
                assert c['rows']==case['rows'] and c['inner']==case['inner'] and c['blocks']==case['rows']//4*(case['inner']//128)
                assert 0<=c['compact_blocks']<=c['blocks'] and c['compact_fraction']==c['compact_blocks']/c['blocks']
                assert c['fixed_payload_bytes']==c['blocks']*772
                ready=c['compact_blocks']>0
                expected=dict(kind='packing',matrix=case['name'],blocks=c['blocks'],compact_blocks=c['compact_blocks'],selected=ready,baseline_selected=ready and case['rows']*case['inner']>=1048576)
            else:expected=dict(kind='packing',matrix=case['name'],blocks=0,compact_blocks=0,selected=False,baseline_selected=False)
            assert p==expected
        q=[v for v in values if v['kind']=='check'];assert len(q)==8
        for batch in (1,2):
            relevant=[v for v in q if v['batch']==batch];assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
            for c in relevant:
                assert c['rows']==batch and c['policy']==1 and c['projections']==len(cases) and c['selected_projections']==sum(v['compact_blocks']>0 for v in cpu_cases.values())
                assert c['values']==batch*sum(v['rows'] for v in cases)
                for key in ('control_fp64_worst','fp64_worst','comparison_worst'):assert math.isfinite(c[key]) and 0<=c[key]<=1
                assert c['finite'] and c['bitwise_equal'] and c['comparison_worst']==0 and c['passed']
            windows=[v for v in values if v['kind']=='samples' and v['batch']==batch];assert len(windows)==8
            for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
            assert all(v['rows']==batch and v['policy']==1 and all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for k in ('gpu_ms','wall_ms')) for v in windows)
            means={name:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
            record=next(v for v in summary['profiles'] if (v['model'],v['batch'],v['policy'])==(model,batch,1));assert record['means']==means and record['checks']==relevant and record['physical_rows']==batch
            for metric in ('gpu','wall'):assert math.isclose(record[metric+'_fraction'],means['candidate'][metric+'_ms']/means['control'][metric+'_ms'],rel_tol=1e-12)
            observed.append(record)
        quality.extend(dict(model=model,**v) for v in q)
    assert observed==summary['profiles'] and quality==summary['checks'] and not summary['full_goal_met']
    profiles.extend(dict(trial=trial,**v) for v in observed);checks.extend(quality)
result=dict(verified_files=len(proof['files']),numeric_checks=len(checks),safety_profiles=controls,timed_executions=len(profiles)*64,profiles=profiles,scope='Private smaller body admission only; control retains current coalesced large weights. Fixed64-thread original arithmetic, original bits/FP64/ownership/recovery checks; extra allocation/packing excluded. No full-model or vLLM claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
