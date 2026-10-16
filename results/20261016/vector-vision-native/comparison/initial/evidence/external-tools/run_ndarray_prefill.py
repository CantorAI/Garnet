"""Actual Garnet context, streamed projection sets, unchanged native/FP64 gates."""
from pathlib import Path
import sys
p=Path(__file__).with_name('run_packed_body_prefill.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns);source=ns['source']
old='from make_packed_body_prefill import context as generate_context,header as generate_header';assert source.count(old)==1
source=source.replace(old,'from make_ndarray_prefill import context as generate_context,header as generate_header,harness as generate_harness')
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body').replace('interleaved-body-native-checks-20261015','coalesced-body-native-checks-20261015')
source=source.replace("s/'make_packed_body_prefill.py'","s/'make_ndarray_prefill.py',s/'run_packed_body_prefill.py',s/'make_packed_body_prefill.py'")
old="cpp.write_bytes((s/'packed_body_prefill_gpu.mm').read_bytes())";assert source.count(old)==1
source=source.replace(old,"cpp.write_bytes(generate_harness((s/'packed_body_prefill_gpu.mm').read_text()).encode())")
source=source.replace('packed_body_prefill_gpu','ndarray_prefill_gpu').replace('metal_context_packed_prefill','metal_context_ndarray_prefill')
source=source.replace("s/'ndarray_prefill_gpu.mm'","s/'packed_body_prefill_gpu.mm'")
needle='metadata=dict(';at=source.index(needle)
source=source[:at]+"paths=list(dict.fromkeys(paths+[p for p in (root/'source/src/backends/metal').iterdir() if p.suffix in ('.h','.mm')]))\n"+source[at:]
needle='for p in paths:';at=source.index(needle)
source=source[:at]+'''sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip())/'System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks'
sdk_paths={
 'MPSNDArray.h':sdk/'MPSCore.framework/Versions/A/Headers/MPSNDArray.h',
 'MPSCoreTypes.h':sdk/'MPSCore.framework/Versions/A/Headers/MPSCoreTypes.h',
 'MPSNDArrayKernel.h':sdk/'MPSNDArray.framework/Versions/A/Headers/MPSNDArrayKernel.h',
 'MPSNDArrayMatrixMultiplication.h':sdk/'MPSNDArray.framework/Versions/A/Headers/MPSNDArrayMatrixMultiplication.h'}
for name,path in sdk_paths.items():(out/name).write_bytes(path.read_bytes())
metadata['sdk_headers']={name:digest(out/name) for name in sdk_paths}
metadata['versions']['macos']=subprocess.check_output(['sw_vers'],text=True)
'''+source[at:]
needle='for p in paths:';at=source.index(needle)
source=source[:at]+"metadata.update(scope='Streamed original trained projection sets in Garnet MetalContext. Policy1 exact-half alias with three-source alpha; policy2 exact expanded FP32 cache with two-source unit alpha; policy3 native null. All original values/FP32 activations/output and bounds unchanged.',limitations='Stable input/output slots reused across patterns, bounded256-entry per-context view cache. F32 policy adds host-created weight storage; mixed policy aliases existing half storage. No synthetic ownership suite, memory saving, complete-model or vLLM claim. Null calibration included; failed profiles have no timings.')\n"+source[at:]
source=source.replace('Completed packed prefill component','Completed native NDArray prefill component')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
