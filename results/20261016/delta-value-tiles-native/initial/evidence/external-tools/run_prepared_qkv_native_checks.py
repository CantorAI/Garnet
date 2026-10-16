"""Fresh original30 plus static-library QKV primitive and trained graph/session rows/fallbacks."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('run_vector_vision_native_checks.py');ns={'__file__':str(p)};saved=sys.argv
sys.argv=[saved[0],saved[1],saved[2],'--generate-only']
try:exec(compile(p.read_text(),str(p),'exec'),ns)
finally:sys.argv=saved
source=ns['source']
def replace(old,new,count=1):
 global source
 assert source.count(old)==count,(old,source.count(old),count);source=source.replace(old,new)
replace('build-vector-vision-width4','build-prepared-qkv-text')
replace("'METAL_COALESCED_BF16_BODY'","'METAL_COALESCED_BF16_BODY','METAL_PREPARED_QKV_TEXT'")
replace("fields[-1]==str(owned_vector_binary)","fields[-1] in (str(owned_vector_binary),str(owned_qkv_binary))")
needle="owned_vector_binary=Path(sys.argv[2]).resolve()/'vector_vision_native_gpu'"
replace(needle,needle+"\nowned_qkv_binary=Path(sys.argv[2]).resolve()/'prepared_qkv_native_probe'")
needle="run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])"
replace(needle,"""run('prepared-qkv-compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(r/'source/src/backends/metal'),str(s/'prepared_union_qkv_primitive.mm'),str(build/'libgarnet_metal.a'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(output/'prepared_qkv_native_probe')])
run('prepared-qkv-primitive',[str(output/'prepared_qkv_native_probe')])
"""+needle)
needle="run('wide-batch-trained-reference',"
replace(needle,"run('prepared-qkv-trained-reference',[py,str(s/'test_prepared_qkv_dense_reference.py'),runtime,str(build),str(r/'source'),str(models/'Qwen3-0.6B'),'bf16',str(output/'trained-qkv-reference')])\n"+needle)
replace('assert len(statuses)==30','assert len(statuses)==33')
needle="external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_vector_vision_native_checks.py')]"
replace(needle,"external=[Path(__file__),Path(__file__).with_name('run_vector_vision_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_prepared_qkv_native_checks.py')]")
ast.parse(source);destination=Path(sys.argv[2]).with_suffix('.prepared_qkv.complete33.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
