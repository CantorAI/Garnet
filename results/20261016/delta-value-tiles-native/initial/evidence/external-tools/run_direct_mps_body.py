"""Native context-copy MPS GEMM at original logical B1/B2, no padded rows."""
from pathlib import Path
source=Path(__file__).with_name('run_padded_mps_body.py').read_text()
source=source.replace('import math,statistics','import math,statistics,re,shlex\nfrom make_direct_mps_body import context as generate_context,harness as generate_harness')
source=source.replace("paths=[Path(__file__),","paths=[Path(__file__),s/'make_direct_mps_body.py',s/'run_padded_mps_body.py',")
source=source.replace('padding/copy into resident inputs outside timing','original-sized inputs; raw BF16 control and forced mixed-storage MPS GEMM in the same isolated context copy')
source=source.replace('library_sha256=digest(library)','reference_static_library_sha256=digest(library)')
start=source.index("binary=out/'padded_mps_body_gpu'");end=source.index("metadata['binary_sha256']",start)
source=source[:start]+'''native=root/'source/src/backends/metal/metal_context.mm'
context_cpp=out/'metal_context_direct.mm';context_cpp.write_bytes(generate_context(native.read_text()).encode())
cpp=out/'direct_mps_body_gpu.mm';cpp.write_bytes(generate_harness((s/'padded_mps_body_gpu.mm').read_text()).encode())
flags=(build/'CMakeFiles/garnet_metal.dir/flags.make').read_text()
match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags,re.M);assert match
defines=[v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')]
assert '-DGARNET_METAL_EXACT_HALF_PREFILL=1' in defines and '-DGARNET_METAL_GLOBAL_HALF_ALPHA=1' in defines
metadata.update(context_generated_sha256=digest(context_cpp),harness_generated_sha256=digest(cpp),context_macro_defines=defines)
binary=out/'direct_mps_body_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(root/'source/src/backends/metal'),str(cpp),str(context_cpp),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
'''+source[end:]
source=source.replace('for physical in (4,8,16):','for physical in (batch,):').replace('Completed native padded-row component','Completed native direct-row MPS component')
exec(compile(source,__file__,'exec'))
