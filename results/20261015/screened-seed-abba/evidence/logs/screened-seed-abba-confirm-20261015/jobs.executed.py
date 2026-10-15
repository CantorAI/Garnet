"""Exclusive private greedy-head screening experiment; no serving or vLLM claim."""
from pathlib import Path
from make_screened_seed_abba import generate_harness as generate,generate_shader
import statistics
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
native=root/'source/src/backends/metal/metal_context.mm';text=native.read_text()
paths=[native,Path(__file__),s/'make_screened_seed_abba.py',s/'make_screened_seed_probe.py',s/'run_guarded_screened_head.py',s/'run_screened_head_repeat.py',s/'make_guarded_screened_head.py',s/'run_screened_head.py',s/'make_screened_head_harness.py',s/'screened_head.metal',s/'prepare_screened_head.py',s/'pruned_head_gpu.mm',s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py']
metadata=dict(control_seed_rows=8192,candidate_seed_rows=1024,strict_screen_math=True,query_admission='finite abs<=1e6 and norm in [1e-10,1e10]; otherwise full original refinement',scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},precision='Derived INT8 filter index with FP32 screening/bounds; every retained logit uses unchanged BF16 bits and original FP32 projection; private greedy output only',timed_operations=['exact 8192-row control versus exact 1024-row candidate seed','seed max and query norm','INT8 screening plus original BF16/FP32 retained-row projection','original full argmax'],limitations='Sampled captured queries only; error allowances not certified for all future input magnitudes; not a quantized serving-model result',hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
for p in paths:
    destination=out/'source-snapshot'/p.relative_to(root/'source');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(p.read_bytes())
raw=text[text.index('struct BF16Params'):text.index('// BF16 weights are expanded')];argmax=text[text.index('struct ArgmaxParams'):text.index('struct Params { ulong rows')]
shader=out/'screened-head.metal';shader.write_bytes(('#include <metal_stdlib>\nusing namespace metal;\n'+raw+argmax+generate_shader((s/'screened_head.metal').read_text())).encode())
cpp=out/'screened_head_gpu.mm';cpp.write_bytes(generate((s/'pruned_head_gpu.mm').read_text()).encode());binary=out/'screened_head_gpu'
metadata.update(shader_sha256=digest(shader),harness_sha256=digest(cpp))
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(cpp),'-o',str(binary)])
metadata['binary_sha256']=digest(binary);profiles=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    if shutil.disk_usage(root).free<3*2**30:raise OSError('Screen-index preparation requires at least 3 GiB free')
    assets=out/(model+'-assets');assets.mkdir()
    origin=root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets')
    original=json.loads((origin/'manifest.json').read_text())
    for filename in ('raw.bin','approximate.bin','remaining.bin'):
        import os
        os.link(origin/filename,assets/filename)
    for query in original['queries']:
        for suffix in ('.input.bin','.reference.bin'):os.link(origin/(query['name']+suffix),assets/(query['name']+suffix))
    (assets/'manifest.json').write_bytes((origin/'manifest.json').read_bytes())
    (assets/'reuse.json').write_text(json.dumps(dict(origin=str(origin),scope='All bytes verified below before GPU use'))+'\n')
    info=json.loads((assets/'manifest.json').read_text());assert len(info['queries'])==17
    for name,sha in [('raw.bin',info['raw_sha256']),('approximate.bin',info['approximate_sha256']),('remaining.bin',info['remaining_norm_sha256'])]:assert digest(assets/name)==sha
    for q in info['queries']:
        for suffix,field in (('.input.bin','input_sha256'),('.reference.bin','reference_sha256')):assert digest(assets/(q['name']+suffix))==q[field]
    metadata['assets'][model]=info;(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    run(model+'-gpu',[str(binary),str(shader),str(assets)])
    records=[json.loads(v) for v in (out/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')];checks=[v for v in records if v['kind']=='check'];samples=[v for v in records if v['kind']=='samples']
    controls=[v for v in records if v['kind']=='control_check'];assert len(controls)==17 and {v['query'] for v in controls}=={q['name'] for q in info['queries']} and all(v['all_values_checked'] and v['greedy_exact'] for v in controls)
    assert len(checks)==17 and {v['query'] for v in checks}=={q['name'] for q in info['queries']} and len(samples)==136
    for c in checks:
        assert c['raw_native_bits_exact'] and c['retained_bits_exact'] and c['greedy_exact']
        assert c['retained_rows']+c['pruned_rows']==c['batch']*info['rows'] and c['counted_weight_byte_fraction']>0
        windows=[v for v in samples if v['query']==c['query']];assert len(windows)==8
        for rnd in (0,1):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
        assert all(len(v[k])==8 and all(x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
        means={policy:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for policy,candidate in (('control',False),('candidate',True))}
        profiles.append(dict(model=model,check=c,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    (out/'summary.json').write_text(json.dumps(dict(profiles=profiles,scope=__doc__,full_goal_met=False),indent=2)+'\n');print('Completed sampled exact-greedy screening component',model,flush=True)
