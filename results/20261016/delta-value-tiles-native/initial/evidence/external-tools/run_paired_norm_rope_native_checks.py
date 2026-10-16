"""Fresh original30 plus linked paired primitive and two retained trained decode/fallback suites."""
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
replace('build-vector-vision-width4','build-paired-norm-rope-decode')
replace("'METAL_COALESCED_BF16_BODY'","'METAL_COALESCED_BF16_BODY','METAL_PAIRED_NORM_ROPE'")
replace("fields[-1]==str(owned_vector_binary)","fields[-1] in (str(owned_vector_binary),str(owned_paired_binary))")
needle="owned_vector_binary=Path(sys.argv[2]).resolve()/'vector_vision_native_gpu'";replace(needle,needle+"\nowned_paired_binary=Path(sys.argv[2]).resolve()/'paired_norm_rope_native_probe'")
needle="run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])"
replace(needle,"""run('paired-rope-compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(r/'source/src/backends/metal'),str(s/'paired_norm_rope_primitive.mm'),str(build/'libgarnet_metal.a'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(output/'paired_norm_rope_native_probe')])
run('paired-rope-primitive',[str(output/'paired_norm_rope_native_probe')])
"""+needle)
needle="run('wide-batch-trained-reference',"
replace(needle,"""run('paired-rope-text-trained-reference',[py,str(s/'test_paired_norm_rope_dense_reference.py'),runtime,str(build),str(r/'source'),str(models/'Qwen3-0.6B'),'bf16',str(output/'trained-paired-rope-text')])
run('paired-rope-coder-trained-reference',[py,str(s/'test_paired_norm_rope_dense_reference.py'),runtime,str(build),str(r/'source'),str(models/'Qwen2.5-Coder-0.5B-Instruct'),'bf16',str(output/'trained-paired-rope-coder')])
"""+needle)
# Account for both early added stages at every intermediate checkpoint, before execution.
replace('assert len(statuses)==23','assert len(statuses)==25')
replace('assert len(statuses)==29','assert len(statuses)==31')
replace('assert len(statuses)==30','assert len(statuses)==34')
needle="external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_vector_vision_native_checks.py')]"
replace(needle,"external=[Path(__file__),Path(__file__).with_name('run_vector_vision_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_paired_norm_rope_native_checks.py')]")
ast.parse(source);destination=Path(sys.argv[2]).with_suffix('.paired_rope.complete34.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
