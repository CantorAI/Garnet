"""Synthetic precomputed three-half product feasibility, exact source bits; exclusive GPU."""
from pathlib import Path
import statistics
p=Path(__file__).with_name('run_constant_bf16_probe.py');source=p.read_text();ns={'__file__':__file__}
exec(compile(source[:source.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
probe=s/'mps_half_triplet_lower_bound.mm';paths=[probe,Path(__file__),s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py']
metadata=dict(scope=__doc__,source_sha256={p.relative_to(root/'source').as_posix():digest(p) for p in paths},
 hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},
 versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),
 lower_bound_exclusions=['GPU source decomposition','learned checkpoint weights','actual model activations','ownership/native integration','general MPS internal accumulator dtype proof'],
 precision=dict(storage_planes='three precomputed FP16 matrices',source='bounded normal FP32 patterns, exact bit reconstruction',result='FP32 MPS result matrix',diagnostics='finite distinguishing product and alpha/beta fixtures, not an internal-instruction dtype audit'),models={})
for p in paths:
    dest=out/'source-snapshot'/p.relative_to(root/'source');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
for model,expected in [('Qwen3-0.6B',(1024,3072)),('Qwen2.5-Coder-0.5B-Instruct',(896,4864)),('Qwen3.5-0.8B',(1024,3584))]:
    path=root/'models'/model/'config.json';c=json.loads(path.read_text());c=c.get('text_config',c);assert (c['hidden_size'],c['intermediate_size'])==expected
    metadata['models'][model]=dict(inner=expected[0],columns=expected[1],config_sha256=digest(path));dest=out/'configs'/(model+'.json');dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes())
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');binary=out/'mps_half_triplet_lower_bound_probe'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(probe),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
run('probe',[str(binary)])
records=[json.loads(line) for line in (out/'probe.log').read_text().splitlines() if line.startswith('{')]
checks=[r for r in records if r['kind']=='check'];assert len(checks)==6;profiles=[]
assert len([r for r in records if r['kind']=='precision_diagnostic'])==3 and len([r for r in records if r['kind']=='scale_diagnostic'])==1
for check in checks:
    assert check['passed'] and check['source_bits_exact'] and check['padding_preserved'] and check['patterns']==4 and check['validated_executions']==12
    assert check['reconstructed_inputs']==4*check['rows']*check['inner']
    for candidate in (1,2):
        windows=[r for r in records if r['kind']=='timing' and (r['model'],r['rows'],r['candidate'])==(check['model'],check['rows'],candidate)];assert len(windows)==40
        for rnd in range(10):assert [r['mode'] for r in sorted([r for r in windows if r['round']==rnd],key=lambda r:r['position'])]==[0,candidate,candidate,0]
        assert all(len(r[k])==5 for r in windows for k in ('encode_ms','wall_ms','gpu_ms'))
        means={policy:{key:statistics.mean(x for r in windows if (r['mode']==candidate)==selected for x in r[key]) for key in ('encode_ms','wall_ms','gpu_ms')} for policy,selected in (('control',False),('candidate',True))}
        profiles.append(dict(model=check['model'],rows=check['rows'],candidate=candidate,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
(out/'summary.json').write_text(json.dumps(dict(checks=checks,profiles=profiles,scope=__doc__,full_goal_met=False),indent=2)+'\n')
print('Synthetic MPS three-half lower-bound feasibility completed; no model precision/performance admission',flush=True)
