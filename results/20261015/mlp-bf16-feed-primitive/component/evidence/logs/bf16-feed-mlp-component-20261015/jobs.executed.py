"""Native common-alpha MPS versus MPSGraph FP32 prefill; wall clock only."""
from pathlib import Path
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
import math,statistics
build=root/'build-coalesced-bf16-body';library=build/'libgarnet_metal.a'
assert digest(build/'libgarnet.dylib')==json.loads((root/'logs/coalesced-body-native-checks-20261015/jobs/metadata.json').read_text())['candidate_sha256']
probe=out/'bf16_feed_mlp_graph_gpu.mm'
run('harness',[sys.executable,str(s/'make_bf16_feed_mlp_component.py'),str(probe)])
paths=[Path(__file__),s/'run_stage_rounded_mlp_graph.py',s/'make_bf16_feed_mlp_component.py',s/'run_dependent_mlp_graph.py',s/'reference_dense.py',s/'run_combined_graph_prefill.py',s/'run_mpsgraph_prefill.py',s/'run_constant_bf16_probe.py',s/'dependent_mlp_graph_gpu.mm',s/'compare_vllm_metal.py',root/'source/src/backends/metal/metal_context.h',root/'source/src/backends/metal/exact_half_codec.h']
metadata=dict(scope='Combined MPSGraph prefill, wall latency only. Original FP32 activation, weight values, matmul and output. No input deduplication, graph-visible model prefixes, complete-layer dependency graph or whole-model result.',limitations='Complete independent resident first-layer projection sets, all matrices included; native single command submission versus one synchronous multi-result graph. Policy1 has FP32 weight placeholders, policy2 has original FP32 weight constants. Initialization/uploads/first graph compile outside warm windows. No graph GPU span, whole-model TTFT or vLLM ratio is inferred.',numerical_bounds='Original FP64 gamma(ceil(K/32)+32)*sumabs+1e-7; 5e-6+5e-5*abs(native)',reference_static_library_sha256=digest(library),source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},versions=dict(python=sys.version,compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
metadata.update(scope='Complete dependent MLP fixtures: native gate/up, SiLU multiplication, down projection versus corresponding FP32 graph. Text/coder layer0 and VL layers0/3; original checkpoint values.',limitations='One native submission or one synchronous graph call PER MLP. VL layers0/3 run sequentially through separate graphs, never fused across layers. True dependencies retained within each MLP. Placeholder/constant policies, first setup separate. Independent CPU FP64 complete MLP reference; no full-model TTFT or vLLM claim.',numerical_bounds='Original trained-layer envelope5e-6+5e-5*abs(full FP64 MLP reference), and unchanged5e-6+5e-5*abs(native). Failed native or candidate profiles have no timed windows.')
metadata.update(scope='Dependent MLP, existing FP32-valued reference semantics: FP64 dots rounded toFP32 at each projection, FP32 activation, finalFP32 projection. Graph/native operations unchanged from prior stress test.',limitations='One native submission or synchronous graph call perMLP; VL blocks remain separate and sequential. Existing FP32 stage reference uses wider CPU dot accumulation. First/setup costs separate; no full-model or vLLM result.',numerical_bounds='Unchanged5e-6+5e-5*abs(stage-rounded reference), and unchanged5e-6+5e-5*abs(native). Prior whole-chain FP64 stress results remain separately archived.',generated_cpp_sha256=digest(probe))
metadata.update(scope='Complete dependent MLP with original BF16 placeholder/constant storage, exact FP32 casts and unchanged FP32 arithmetic. Mandatory GPU roundtrip of every learned weight bit before timing.',limitations='Original stage-rounded reference and native comparison bounds; all failed profiles excluded without changed coefficients. One native submission or synchronous graph call per MLP; no native command-buffer integration, full-model TTFT, memory saving or vLLM result.')
(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
for path in paths:
    target=out/'source-snapshot'/path.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
binary=out/'bf16_feed_mlp_graph_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(root/'source/src/backends/metal'),str(probe),str(library),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-framework','MetalPerformanceShadersGraph','-o',str(binary)])
metadata['binary_sha256']=digest(binary);profiles=[];all_checks=[];skipped=[];first=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assets=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets');info=json.loads((assets/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==info['checkpoint_sha256']
    with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        header_bytes=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+header_bytes]);offset=8+header_bytes
        for case in info['cases']:
            assert digest(assets/case['name']/'raw.bin')==case['sha256']['raw.bin'];tensor=header[case['tensor']];assert tensor['dtype']=='BF16' and tensor['shape']==[case['rows'],case['inner']]
            lo,hi=tensor['data_offsets'];assert hashlib.sha256(mapping[offset+lo:offset+hi]).hexdigest()==case['sha256']['raw.bin']
    metadata['assets'][model]=info;(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
    run(model+'-gpu',[str(binary),str(assets),model])
    values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    casts=[v for v in values if v['kind']=='weight_cast']
    learned={c['tensor']:c['rows']*c['inner'] for c in info['cases'] if '.mlp.' in c['tensor']}
    assert len(casts)==len(learned) and {v['tensor'] for v in casts}==set(learned)
    assert all(v['bitwise_equal'] and v['values']==learned[v['tensor']] for v in casts)
    checks=[v for v in values if v['kind']=='check'];assert len(checks)==16
    all_checks.extend(dict(model=model,**v) for v in checks);first.extend(dict(model=model,**v) for v in values if v['kind']=='first_graph_run')
    for batch in (1,2):
        rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch
        for policy in (1,2):
            relevant=[v for v in checks if (v['batch'],v['policy'])==(batch,policy)]
            assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4)) and all(v['rows']==rows and v['projections']==sum('.mlp.' in c['tensor'] for c in info['cases']) for v in relevant)
            windows=[v for v in values if v['kind']=='samples' and (v['batch'],v['policy'])==(batch,policy)]
            if not all(v['passed'] for v in relevant):assert not windows;continue
            assert len(windows)==8
            for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
            assert all(len(v['wall_ms'])==8 and all(math.isfinite(x) and x>0 for x in v['wall_ms']) for v in windows)
            means={name:statistics.mean(x for v in windows if v['candidate']==candidate for x in v['wall_ms']) for name,candidate in (('control',False),('candidate',True))}
            profiles.append(dict(model=model,batch=batch,rows=rows,policy=policy,checks=relevant,wall_ms=means,wall_fraction=means['candidate']/means['control']))
    (out/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=profiles,checks=all_checks,first_graph_runs=first,full_goal_met=False),indent=2)+'\n').encode());print('Completed combined graph prefill component',model,flush=True)
