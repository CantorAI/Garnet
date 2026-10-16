"""Audit exact shape specialization, canonical bytes and both ABBA comparisons."""
import hashlib,json,math,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for e in proof['files']:
    p=root/e['path'];assert p.resolve().is_relative_to(root) and p.stat().st_size==e['bytes'] and digest(p)==e['sha256']
profiles=[];validated=0;timed=0;configurations=0;worst=0
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;out=root/'logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text())
    for name,sha in meta['source_sha256'].items():assert digest(out/'source-snapshot'/name)==sha
    assert digest(out/'combined.metal')==meta['shader_sha256']
    native=(out/'source-snapshot/src/backends/metal/metal_context.mm').read_text()
    original=('#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+native[native.index('kernel void bf16_decode('):native.index('kernel void bf16_gemm(')])
    assert (out/'combined.metal').read_text().startswith(original)
    cpp=(out/'shape_specialized_bf16_gpu.mm').read_text();assert 'bool noTail=mode&&columns%tile==0;' in cpp and 'MTLFunctionConstantValues' in cpp
    assert next(e['sha256'] for e in proof['omitted'] if e['path']=='logs/'+trial+'/jobs/shape_specialized_bf16_gpu')==meta['binary_sha256']
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==5 and {v['name'] for v in stages}=={'generate','compile'}|{m+'-gpu' for m in meta['assets']}
    for s in stages:
        assert s['exit_code']==0;monitor=json.loads((out/(s['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
    count=0
    for model,assets in meta['assets'].items():
        lookup={c['name']:c for c in assets['cases']}
        for c in assets['cases']:
            e=proof['omitted_matrices'][model+':'+c['name']];assert e['sha256']==c['sha256'] and e['bytes']==c['inner']*c['columns']*2 and e['checkpoint_sha256']==assets['checkpoint_sha256']
        data=[json.loads(v) for v in (out/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')];checks=[v for v in data if v['kind']=='check']
        assert len(checks)==2*len(lookup) and {(v['matrix'],v['rows']) for v in checks}=={(n,b) for n in lookup for b in (1,2)}
        terminal=next(v for v in data if v['kind']=='completed');assert terminal['profiles']==len(checks) and terminal['executions']==sum(v['validated_executions']+4 for v in checks)
        for c in checks:
            assert c['bits_exact'] and c['padding_preserved'] and c['input_variants']==4 and c['validated_executions']>240 and 0<=c['fp64_worst_ratio']<=1
            warm=[v for v in data if v['kind']=='warmup' and v['matrix']==c['matrix'] and v['rows']==c['rows']];assert len(warm)==1 and warm[0]['elapsed_ms']>=250
            setup=[v for v in data if v['kind']=='setup' and v['matrix']==c['matrix'] and v['rows']==c['rows']];assert len(setup)==1
            case=lookup[c['matrix']];assert setup[0]['tail_checks_removed']==(case['columns']%(4 if case['columns']<=256 else 16)==0)
            validated+=c['validated_executions'];configurations+=1;worst=max(worst,c['fp64_worst_ratio'])
            for pair in ('native','dimensions'):
                windows=[v for v in data if v['kind']=='timing' and v['matrix']==c['matrix'] and v['rows']==c['rows'] and v['pair']==pair];assert len(windows)==8
                for rnd in (0,1):assert [v['policy'] for v in windows if v['round']==rnd]==['control','candidate','candidate','control']
                assert all(v['bits_exact'] for v in windows)
                assert all(len(v[k])==12 and all(math.isfinite(x) and x>=0 for x in v[k]) for v in windows for k in ('encode_ms','wall_ms','gpu_ms'));timed+=96
                means={policy:{k:statistics.mean(x for v in windows if v['policy']==policy for x in v[k]) for k in ('encode_ms','wall_ms','gpu_ms')} for policy in ('control','candidate')}
                p=next(v for v in summary['profiles'] if v['model']==model and v['matrix']==c['matrix'] and v['batch']==c['rows'] and v['pair']==pair);assert p['checks']==c;count+=1
                for policy in means:
                    for k in means[policy]:assert abs(means[policy][k]-p['means'][policy][k])<1e-12
                assert abs(p['gpu_fraction']-means['candidate']['gpu_ms']/means['control']['gpu_ms'])<1e-12 and abs(p['wall_fraction']-means['candidate']['wall_ms']/means['control']['wall_ms'])<1e-12
                profiles.append(dict(trial=trial,tensor=case['tensor'],inner=case['inner'],columns=case['columns'],pipeline_specialization_ms=setup[0]['pipeline_specialization_ms'],**p))
    assert count==len(summary['profiles'])==136
result=dict(verified_files=len(proof['files']),trials=len(proof['jobs']),matrix_batch_checks=configurations,bit_exact_validated_executions=validated,timed_executions=timed,fp64_worst_ratio=worst,profiles=profiles,scope='Component only: original dimensions/strides specialized and safe full-tile bounds eliminated; pipeline setup excluded from projection timing; no full inference or vLLM claim',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
