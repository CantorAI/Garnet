'Deferred RHS transpose compatibility with original synthetic numerical/guard gates.'
from pathlib import Path
import json,subprocess,sys
p=Path(__file__).with_name('run_constant_bf16_probe.py');base=p.read_text()
exec(compile(base[:base.index('native=root/')],str(p),'exec'))
run('harness',[sys.executable,str(s/'make_ndarray_view_probe.py'),str(out/'ndarray_view_gpu.mm')])
paths=[Path(__file__),p,s/'run_ndarray_compat.py',s/'make_ndarray_view_probe.py',s/'make_ndarray_alpha_probe.py',s/'ndarray_compat_gpu.mm',s/'compare_vllm_metal.py',root/'source/src/backends/metal/exact_half_codec.h']
manifest={p.relative_to(root/'source').as_posix():digest(p) for p in paths}
for p in paths:
    target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip())/'System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks'
headers={
 'MPSNDArray.h':sdk/'MPSCore.framework/Versions/A/Headers/MPSNDArray.h',
 'MPSCoreTypes.h':sdk/'MPSCore.framework/Versions/A/Headers/MPSCoreTypes.h',
 'MPSNDArrayKernel.h':sdk/'MPSNDArray.framework/Versions/A/Headers/MPSNDArrayKernel.h',
 'MPSNDArrayMatrixMultiplication.h':sdk/'MPSNDArray.framework/Versions/A/Headers/MPSNDArrayMatrixMultiplication.h'}
for name,p in headers.items():(out/name).write_bytes(p.read_bytes())
metadata=dict(scope=__doc__,generated_cpp_sha256=digest(out/'ndarray_view_gpu.mm'),source_sha256=manifest,sdk_headers={name:digest(out/name) for name in headers},
 hardware={key:subprocess.check_output(['sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},
 compiler=subprocess.check_output(['xcrun','clang++','--version'],text=True),macos=subprocess.check_output(['sw_vers'],text=True),
 limitations='Synthetic API compatibility only. No learned checkpoint, Garnet backend integration, latency, memory-saving or vLLM result.',full_goal_met=False)
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
binary=out/'ndarray_view_gpu'
run('compile',['xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(root/'source/src/backends/metal'),str(out/'ndarray_view_gpu.mm'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
results=[]
for storage,variant in [('mixed','unit'),('mixed','three'),('f32','unit'),('f32','three')]:
    mode=storage+'-'+variant
    try:run(mode,[str(binary),storage,variant])
    except AssertionError:
        assert statuses[-1]['name']==mode and statuses[-1]['exit_code']!=0
        check()
    records=[json.loads(line) for line in (out/(mode+'.log')).read_text().splitlines() if line.startswith('{')]
    passed=statuses[-1]['exit_code']==0
    if passed:assert records[-1]==dict(kind='complete',checks=16)
    results.append(dict(mode=mode,exit_code=statuses[-1]['exit_code'],complete_checks=sum(v['kind']=='check' for v in records),compatible=passed))
(out/'summary.json').write_text(json.dumps(dict(modes=results,all_compatible=all(v['compatible'] for v in results),full_goal_met=False),indent=2))
assert all(digest(root/'source'/name)==sha for name,sha in manifest.items())
sys.exit(0 if all(v['compatible'] for v in results) else 1)
