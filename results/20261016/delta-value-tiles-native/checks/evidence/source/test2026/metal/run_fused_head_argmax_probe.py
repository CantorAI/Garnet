"""Exclusive full-head projection plus greedy reduction component trial."""
from pathlib import Path
base=Path(__file__).with_name('run_decode_subsimd_probe.py').read_text()
exec(compile(base[:base.index('native=r/')],__file__,'exec'))
native=r/'source/src/backends/metal/metal_context.mm';origin=r/'logs/pruned-head-gpu-20261013/jobs'
paths=[native,s/'fused_head_argmax_gpu.mm',s/'make_fused_head_argmax_shader.py',Path(__file__),s/'run_decode_subsimd_probe.py',s/'compare_vllm_metal.py']
metadata=dict(scope='Projection plus greedy reduction only; no full inference or fresh vLLM claim',source_sha256={str(p.relative_to(r/'source')):digest(p) for p in paths},assets={},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},math='Original native BF16 dot products, FP32 sums and original greedy/nonfinite semantics; no pruning')
for p in paths:
    target=output/'source-snapshot'/p.relative_to(r/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
shader=output/'combined.metal';binary=output/'fused_head_argmax_gpu'
run('shader',[sys.executable,str(s/'make_fused_head_argmax_shader.py'),str(native),str(shader)])
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(s/'fused_head_argmax_gpu.mm'),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);metadata['shader_sha256']=digest(shader)
summary=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    guard();assets=origin/(model+'-assets');manifest=json.loads((assets/'manifest.json').read_text());assert digest(assets/'raw.bin')==manifest['raw_sha256']
    for query in manifest['queries']:
        assert digest(assets/(query['name']+'.input.bin'))==query['input_sha256'] and digest(assets/(query['name']+'.reference.bin'))==query['reference_sha256']
    metadata['assets'][model]=manifest;(output/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
    run(model+'-gpu',[str(binary),str(shader),str(assets)])
    records=[json.loads(line) for line in (output/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    checks=[v for v in records if v['kind']=='check'];assert len(checks)==len(manifest['queries'])+32
    for check in checks:
        assert all(check[k] for k in ('projection_bits_exact','greedy_exact','padding_preserved'))
        if not check['timed']:continue
        assert check['captured_reference_checked']
        windows=[v for v in records if v['kind']=='timing' and v['query']==check['query']];assert len(windows)==8
        means={name:{metric:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[metric]) for metric in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
        summary.append(dict(model=model,query=check['query'],batch=check['batch'],means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
(output/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=summary,full_goal_met=False),indent=2)+'\n').encode());guard()
print(json.dumps(dict(captured_profiles=len(summary),edge_checks=96,full_goal_met=False)),flush=True)
