"""Compile private paired norm-RoPE context and execute original-bit/ownership controls."""
from pathlib import Path
import re,shlex
from make_paired_norm_rope import context,header
p=Path(__file__).with_name('run_constant_bf16_probe.py');text=p.read_text();ns={'__file__':__file__}
exec(compile(text[:text.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
selected=root/'vector-vision-native-investigation-20261016';base=selected/'source/src/backends/metal';build=selected/'build-vector-vision-width4';assert digest(build/'libgarnet.dylib')=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
generated=out/'generated';generated.mkdir();(generated/'metal_context.h').write_bytes(header((base/'metal_context.h').read_text()).encode());(generated/'metal_context.mm').write_bytes(context((base/'metal_context.mm').read_text()).encode())
flags_path=build/'CMakeFiles/garnet_metal.dir/flags.make';flags=flags_path.read_text();match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags,re.M);assert match
defines=[v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')];assert '-DGARNET_METAL_GLOBAL_HALF_ALPHA=1' in defines
paths=[Path(__file__),s/'make_paired_norm_rope.py',s/'paired_norm_rope_primitive.mm',s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py',s/'launch_paired_norm_rope.py',s/'launch_quiet_prefill_residual_norm_probe.py',flags_path,*[p for p in base.iterdir() if p.suffix in ('.h','.mm')]]
meta=dict(source_sha256={p.as_posix():digest(p) for p in paths},generated_sha256={p.name:digest(p) for p in generated.iterdir()},defines=defines,reference_library_sha256=digest(build/'libgarnet.dylib'),scope=__doc__,limitations='Synthetic FP32 primitive only; no learned weights, captured activations, timing or native admission. Original arithmetic body and independent outputs retained; no joined weight or output buffers.',full_goal_met=False)
for i,p in enumerate(paths):
 target=out/'source-snapshot'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');binary=out/'paired_norm_rope_primitive_probe'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(generated),'-I'+str(base),str(s/'paired_norm_rope_primitive.mm'),str(generated/'metal_context.mm'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
meta['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');run('primitive',[str(binary)])
for name,sha in meta['source_sha256'].items():assert digest(Path(name))==sha
print('Private paired norm-RoPE primitive completed; no learned/model speed claim.',flush=True)
