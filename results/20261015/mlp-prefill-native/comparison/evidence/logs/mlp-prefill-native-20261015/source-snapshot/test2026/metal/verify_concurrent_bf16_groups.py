"""Independent original-shader, full-bit-parity and grouped timing audit."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
profiles=[];validated=0;timed=0;configurations=0;worst=0
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;out=root/'evidence/logs'/trial/'jobs'
    meta=json.loads((out/'metadata.json').read_text());summary=json.loads((out/'summary.json').read_text())
    for name,sha in meta['source_sha256'].items():assert digest(out/'source-snapshot'/name)==sha
    native=(out/'source-snapshot/src/backends/metal/metal_context.mm').read_text()
    shader=('#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+native[native.index('kernel void bf16_decode('):native.index('kernel void bf16_gemm(')]).encode()
    assert (out/'native-decode.metal').read_bytes()==shader and digest(out/'native-decode.metal')==meta['native_shader_sha256']
    harness=(out/'source-snapshot/test2026/metal/concurrent_bf16_group_gpu.mm').read_text()
    assert 'computeCommandEncoderWithDispatchType:mode==2?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial' in harness
    assert 'Projection or padding changed bits' in harness and 'Original FP64 envelope exceeded' in harness
    assert proof['omitted_binaries']['logs/'+trial+'/jobs/concurrent_bf16_group_gpu']['sha256']==meta['binary_sha256']
    stages=json.loads((out/'statuses.json').read_text());assert len(stages)==4 and {s['name'] for s in stages}=={'compile'}|{m+'-gpu' for m in meta['assets']}
    for s in stages:
        assert s['exit_code']==0
        monitor=json.loads((out/(s['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
    count=0
    for model,assets in meta['assets'].items():
        assert json.loads((out/(model+'-assets/groups.json')).read_text())==assets
        assert {g['name'] for g in assets['groups']}==({'mlp-gate-up','irregular-readonly-pair'}|({'attention-qkv'} if model!='Qwen3.5-0.8B' else set()))
        for g in assets['groups']:
            assert len({c['inner'] for c in g['cases']})==1 and len(g['cases'])==(3 if g['name']=='attention-qkv' else 2)
            if g['name']=='mlp-gate-up':assert [c['tensor'].rsplit('.',3)[-2] for c in g['cases']]==['gate_proj','up_proj']
            if g['name']=='attention-qkv':assert [c['tensor'].rsplit('.',3)[-2] for c in g['cases']]==['q_proj','k_proj','v_proj']
        records=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
        checks=[v for v in records if v['kind']=='check'];assert {(v['group'],v['rows']) for v in checks}=={(g['name'],b) for g in assets['groups'] for b in (1,2)} and len(checks)==2*len(assets['groups'])
        for c in checks:
            assert c['bits_exact'] and c['padding_preserved'] and c['input_variants']==4 and c['fp64_worst_ratio']<=1 and c['validated_executions']>=252
            assert c['projections']==len(next(g for g in assets['groups'] if g['name']==c['group'])['cases'])
            warm=[v for v in records if v['kind']=='warmup' and v['group']==c['group'] and v['rows']==c['rows']];assert len(warm)==1 and warm[0]['elapsed_ms']>=250
            configurations+=1;validated+=c['validated_executions'];worst=max(worst,c['fp64_worst_ratio'])
            for mode in (1,2):
                windows=[v for v in records if v['kind']=='timing' and v['group']==c['group'] and v['rows']==c['rows'] and v['candidate_mode']==mode];assert len(windows)==8
                for rnd in (0,1):
                    group=sorted([v for v in windows if v['round']==rnd],key=lambda v:v['window']);assert [v['window'] for v in group]==[0,1,2,3] and [v['mode'] for v in group]==[0,mode,mode,0]
                assert all(len(v[k])==12 and all(x>=0 for x in v[k]) for v in windows for k in ('encode_ms','wall_ms','gpu_ms'));timed+=96
                means={policy:{k:statistics.mean(x for v in windows if (v['mode']==mode)==candidate for x in v[k]) for k in ('encode_ms','wall_ms','gpu_ms')} for policy,candidate in (('control',False),('candidate',True))}
                p=next(v for v in summary['profiles'] if v['model']==model and v['group']==c['group'] and v['batch']==c['rows'] and v['candidate_mode']==mode);assert p['checks']==c;count+=1
                for policy in means:
                    for k in means[policy]:assert abs(p['means'][policy][k]-means[policy][k])<1e-12
                assert abs(p['wall_fraction']-means['candidate']['wall_ms']/means['control']['wall_ms'])<1e-12 and abs(p['gpu_fraction']-means['candidate']['gpu_ms']/means['control']['gpu_ms'])<1e-12
                profiles.append(dict(trial=trial,**p))
    assert count==len(summary['profiles'])==32
result=dict(verified_files=len(proof['files']),trials=len(proof['jobs']),group_batch_checks=configurations,validated_executions=validated,timed_executions=timed,fp64_worst_ratio=worst,profiles=profiles,scope='Independent projection component only, original shader bits; concurrent API request does not prove hardware overlap; no full inference or vLLM gain',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
