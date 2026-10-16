"""Fresh original30 plus real-library bit/state/ownership/fallback primitive compile and execution."""
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

replace('build-vector-vision-width4','build-delta-value-tile64-prefill')
replace("'METAL_COALESCED_BF16_BODY'","'METAL_COALESCED_BF16_BODY','METAL_DELTA_VALUE_TILE64'")
replace('fields[-1]==str(owned_vector_binary)','fields[-1] in (str(owned_vector_binary),str(owned_delta_binary))')
needle="owned_vector_binary=Path(sys.argv[2]).resolve()/'vector_vision_native_gpu'";replace(needle,needle+"\nowned_delta_binary=Path(sys.argv[2]).resolve()/'delta_value_tiles_native_probe'")
needle="run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])"
replace(needle,"""run('delta-value64-compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(r/'source/src/backends/metal'),str(s/'delta_value_tiles_native.mm'),str(build/'libgarnet_metal.a'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(output/'delta_value_tiles_native_probe')])
run('delta-value64-primitive',[str(output/'delta_value_tiles_native_probe'),str(r/'selected-component/original-recurrence.metal')])
"""+needle)
replace('assert len(statuses)==23','assert len(statuses)==25');replace('assert len(statuses)==29','assert len(statuses)==31');replace('assert len(statuses)==30','assert len(statuses)==32')
needle="external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_vector_vision_native_checks.py')]"
replace(needle,"external=[Path(__file__),Path(__file__).with_name('run_vector_vision_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_delta_value_tiles_native_checks.py')]")
ast.parse(source);destination=Path(sys.argv[2]).with_suffix('.delta_value64.complete32.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
