"""Fresh original28 stages plus actual native ordered-vector primitive."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('run_ndarray_native_checks_repaired.py');ns={'__file__':str(p)}
saved=sys.argv;sys.argv=[saved[0],saved[1],saved[2],'--generate-only']
try:exec(compile(p.read_text(),str(p),'exec'),ns)
finally:sys.argv=saved
source=ns['source'].replace('build-ndarray-prefill-shape80-guardrepair','build-vector-vision-width4')
source=source.replace('ndarray_prefill_cache_gpu','vector_vision_native_gpu').replace('ndarray-cache','vector-vision-primitive').replace('owned_ndarray_binary','owned_vector_binary')
old="external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('launch_ndarray_native_checks_repaired.py')]"
assert source.count(old)==1
source=source.replace(old,"external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_vector_vision_native_checks.py')]")
ast.parse(source)
destination=Path(sys.argv[2]).with_suffix('.vector.complete30.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
