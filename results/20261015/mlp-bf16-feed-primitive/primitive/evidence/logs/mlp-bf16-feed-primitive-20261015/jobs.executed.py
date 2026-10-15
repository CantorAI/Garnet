'Original64 plus64 chained cases using BF16 feed storage and FP32 arithmetic.'
from pathlib import Path
import json,re,shlex,subprocess,sys

p=Path(__file__).with_name('run_constant_bf16_probe.py')
source=p.read_text();exec(compile(source[:source.index("native=root/")],str(p),'exec'))
from make_mlp_bf16_feed_graph import context,graph,primitive

native=root/'source/src/backends/metal'
paths=[Path(__file__),p,s/'make_mlp_bf16_feed_graph.py',s/'run_mlp_command_buffer_primitive.py',s/'make_mlp_command_buffer_context.py',s/'mlp_prefill_graph_test.cpp',s/'compare_vllm_metal.py']
paths.extend(p for p in native.iterdir() if p.suffix in ('.h','.mm'))
manifest={p.relative_to(root/'source').as_posix():digest(p) for p in paths}
for path in paths:
    dest=out/'source-snapshot'/path.relative_to(root/'source')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
chained=True
generated={
    'metal_context.mm':context((native/'metal_context.mm').read_text()),
    'metal_context.h':(native/'metal_context.h').read_text(),
    'mlp_prefill_graph_impl.h':graph((native/'mlp_prefill_graph_impl.h').read_text()),
    'mlp_command_buffer_primitive.cpp':primitive((s/'mlp_prefill_graph_test.cpp').read_text(),chained=chained)}
for name,value in generated.items():(out/name).write_bytes(value.encode())
sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip())
for name,rel in {
    'MPSGraph.h':'MetalPerformanceShadersGraph.framework/Headers/MPSGraph.h',
    'MPSCommandBuffer.h':'MetalPerformanceShaders.framework/Frameworks/MPSCore.framework/Headers/MPSCommandBuffer.h',
    'MPSCoreTypes.h':'MetalPerformanceShaders.framework/Versions/A/Frameworks/MPSCore.framework/Versions/A/Headers/MPSCoreTypes.h'}.items():
    (out/name).write_bytes((sdk/'System/Library/Frameworks'/rel).read_bytes())
flags=root/'build-mlp-prefill-graph-guardfix/CMakeFiles/garnet_metal.dir/flags.make'
defines=shlex.split(re.search(r'^OBJCXX_DEFINES = (.*)$',flags.read_text(),re.M)[1])
defines=[v for v in defines if v.startswith('-DGARNET_METAL_')]
assert '-DGARNET_METAL_MLP_PREFILL_GRAPH=1' in defines
metadata=dict(scope=__doc__,weight_storage='original BF16 buffers',arithmetic='FP32 after exact graph cast; no activation precision change',source_sha256=manifest,defines=defines,chained=chained,
    generated_sha256={name:digest(out/name) for name in generated},
    sdk_headers={name:digest(out/name) for name in ('MPSGraph.h','MPSCommandBuffer.h','MPSCoreTypes.h')},
    hardware={key:subprocess.check_output(['sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},
    compiler=subprocess.check_output(['xcrun','clang++','--version'],text=True),
    limitations='Primitive correctness/ordering only. No trained checkpoint validation, timing, model generation, speed gain or vLLM claim.',full_goal_met=False)
(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
binary=out/'mlp_bf16_feed_gpu'
run('compile',['xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(out),'-I'+str(native),str(out/'metal_context.mm'),str(out/'mlp_command_buffer_primitive.cpp'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-framework','MetalPerformanceShadersGraph','-o',str(binary)])
metadata['binary_sha256']=digest(binary)
(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
run('primitive',[str(binary)])
assert all(digest(root/'source'/name)==sha for name,sha in manifest.items())
print('Private MPSCommandBuffer primitive finished; full original trained validation and model performance remain required.',flush=True)
