'Original physical prefill shapes: selected global-alpha MPS versus packed FP32 tiles.'
from pathlib import Path
base=Path(__file__).with_name('run_constant_bf16_probe.py').read_text();ns={'__file__':__file__}
exec(compile(base[:base.index("native=root/")],__file__,'exec'),ns);globals().update(ns)
import math,statistics,re,shlex
from make_ndarray_prefill import context as generate_context,header as generate_header,harness as generate_harness
build=root/'build-coalesced-bf16-body';library=build/'libgarnet_metal.a'
assert digest(build/'libgarnet.dylib')==json.loads((root/'logs/coalesced-body-native-checks-20261015/jobs/metadata.json').read_text())['candidate_sha256']
for trial in ('screened-prefill-reset-native-20261015','screened-prefill-reset-confirm-native-20261015'):assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
paths=[Path(__file__),s/'make_ndarray_prefill.py',s/'run_packed_body_prefill.py',s/'make_packed_body_prefill.py',s/'make_prefill_tile_shader.py',s/'packed_body_prefill_gpu.mm',s/'run_streamed_direct_mps_body.py',root/'source/src/backends/metal/interleaved_bf16_body_shader.h',root/'source/src/backends/metal/screened_head_index.h',root/'source/src/backends/metal/screened_head_shader.h',root/'source/src/backends/metal/screened_head_native_impl.h',s/'run_direct_mps_body.py',s/'streamed_direct_mps_body_gpu.mm',s/'make_direct_mps_body.py',s/'run_padded_mps_body.py',s/'padded_mps_body_gpu.mm',s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py',root/'source/src/backends/metal/metal_context.h',root/'source/src/backends/metal/metal_context.mm',root/'source/src/backends/metal/exact_half_codec.h']
paths=list(dict.fromkeys(paths+[p for p in (root/'source/src/backends/metal').iterdir() if p.suffix in ('.h','.mm')]))
metadata=dict(scope=__doc__,limitations='Component only; independent real per-layer projections in one submission, not a full model, Frozen physical prefill rows24/44,26/52,80/160; FP32 activation and accumulation, original BF16 values in four-output storage; current global-alpha MPS control. Every projection set uses one submission; no complete-model or vLLM ratio, no full-model or vLLM claim',numerical_bounds='Unchanged FP64 gamma(ceil(K/32)+32)*sumabs+1e-7 and original-control 5e-6+5e-5*abs(control)',reference_static_library_sha256=digest(library),source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},versions=dict(python=sys.version,compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip())/'System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks'
sdk_paths={
 'MPSNDArray.h':sdk/'MPSCore.framework/Versions/A/Headers/MPSNDArray.h',
 'MPSCoreTypes.h':sdk/'MPSCore.framework/Versions/A/Headers/MPSCoreTypes.h',
 'MPSNDArrayKernel.h':sdk/'MPSNDArray.framework/Versions/A/Headers/MPSNDArrayKernel.h',
 'MPSNDArrayMatrixMultiplication.h':sdk/'MPSNDArray.framework/Versions/A/Headers/MPSNDArrayMatrixMultiplication.h'}
for name,path in sdk_paths.items():(out/name).write_bytes(path.read_bytes())
metadata['sdk_headers']={name:digest(out/name) for name in sdk_paths}
metadata['versions']['macos']=subprocess.check_output(['sw_vers'],text=True)
metadata.update(scope='Streamed original trained projection sets in Garnet MetalContext. Policy1 exact-half alias with three-source alpha; policy2 exact expanded FP32 cache with two-source unit alpha; policy3 native null. All original values/FP32 activations/output and bounds unchanged.',limitations='Stable input/output slots reused across patterns, bounded256-entry per-context view cache. F32 policy adds host-created weight storage; mixed policy aliases existing half storage. No synthetic ownership suite, memory saving, complete-model or vLLM claim. Null calibration included; failed profiles have no timings.')
for p in paths:
    target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
native=root/'source/src/backends/metal/metal_context.mm'
context_cpp=out/'metal_context_ndarray_prefill.mm';context_cpp.write_bytes(generate_context(native.read_text()).encode())
cpp=out/'ndarray_prefill_gpu.mm';cpp.write_bytes(generate_harness((s/'packed_body_prefill_gpu.mm').read_text()).encode())
header_cpp=out/'metal_context.h';header_cpp.write_bytes(generate_header((root/'source/src/backends/metal/metal_context.h').read_text()).encode())
metadata['private_header_sha256']=digest(header_cpp)
flags=(build/'CMakeFiles/garnet_metal.dir/flags.make').read_text()
match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags,re.M);assert match
defines=[v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')]
assert '-DGARNET_METAL_INTERLEAVED_BF16_BODY=1' in defines
assert '-DGARNET_METAL_EXACT_HALF_PREFILL=1' in defines and '-DGARNET_METAL_GLOBAL_HALF_ALPHA=1' in defines
metadata.update(context_generated_sha256=digest(context_cpp),harness_generated_sha256=digest(cpp),context_macro_defines=defines)
binary=out/'ndarray_prefill_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(root/'source/src/backends/metal'),str(cpp),str(context_cpp),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
metadata['binary_sha256']=digest(binary);profiles=[];all_checks=[];skipped=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assets=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets');info=json.loads((assets/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors'
    assert digest(checkpoint)==info['checkpoint_sha256']
    with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        header_bytes=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+header_bytes]);start=8+header_bytes
        for c in info['cases']:
            raw=assets/c['name']/'raw.bin';assert digest(raw)==c['sha256']['raw.bin']
            tensor=header[c['tensor']];assert tensor['dtype']=='BF16' and tensor['shape']==[c['rows'],c['inner']]
            lo,hi=tensor['data_offsets'];assert hashlib.sha256(mapping[start+lo:start+hi]).hexdigest()==c['sha256']['raw.bin']
    metadata['assets'][model]=info;(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
    run(model+'-gpu',[str(binary),str(assets),model])
    values=[json.loads(line) for line in (out/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    checks=[v for v in values if v['kind']=='check'];assert len(checks)==24
    all_checks.extend(dict(model=model,**v) for v in checks)
    for batch in (1,2):
        rows=80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch
        for policy in (1,2,3):
            relevant=[v for v in checks if (v['batch'],v['policy'])==(batch,policy)];assert len(relevant)==4 and {v['pattern'] for v in relevant}==set(range(4))
            assert all(v['rows']==rows and v['projections']==len(info['cases']) for v in relevant)
            windows=[v for v in values if v['kind']=='samples' and (v['batch'],v['policy'])==(batch,policy)]
            if not all(v['passed'] for v in relevant):assert not windows;continue
            assert len(windows)==8 and all(v['rows']==rows for v in windows)
            for rnd in (0,1):assert [(v['position'],v['candidate']) for v in windows if v['round']==rnd]==list(enumerate((False,True,True,False)))
            assert all(len(v[k])==8 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
            means={name:{k:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[k]) for k in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))}
            profiles.append(dict(model=model,batch=batch,physical_rows=rows,policy=policy,checks=relevant,means=means,gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms']))
    (out/'summary.json').write_bytes((json.dumps(dict(scope=metadata['scope'],profiles=profiles,checks=all_checks,full_goal_met=False),indent=2)+'\n').encode());print('Completed native NDArray prefill component',model,flush=True)
