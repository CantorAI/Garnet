"""Fresh original28 plus selective actual-library cache/route/abort tests, repaired owned guard."""
from pathlib import Path
import ast,sys
p=Path(__file__).with_name('run_ndarray_native_checks_repaired.py');ns={'__file__':str(p)}
saved=sys.argv;sys.argv=[saved[0],saved[1],saved[2],'--generate-only']
try:exec(compile(p.read_text(),str(p),'exec'),ns)
finally:sys.argv=saved
source=ns['source']
source=source.replace('build-ndarray-prefill-shape80-guardrepair','build-ndarray-selective-prefill')
source=source.replace('ndarray_prefill_cache_gpu','ndarray_selective_prefill_cache_gpu')
old="external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('launch_ndarray_native_checks_repaired.py')]"
assert source.count(old)==1
source=source.replace(old,"external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_ndarray_selective_native_checks.py')]")
ast.parse(source)
destination=Path(sys.argv[2]).with_suffix('.selective.complete30.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
