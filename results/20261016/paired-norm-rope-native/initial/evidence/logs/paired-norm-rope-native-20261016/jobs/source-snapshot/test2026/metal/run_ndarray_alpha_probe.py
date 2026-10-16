"""Unit coefficient and explicit zero-addend diagnostics, sequential per format."""
from pathlib import Path
import sys
p=Path(__file__).with_name('run_ndarray_compat.py');source=p.read_text()
source=source.replace("paths=[Path(__file__),p,","run('harness',[sys.executable,str(s/'make_ndarray_alpha_probe.py'),str(out/'ndarray_alpha_gpu.mm')])\npaths=[Path(__file__),p,s/'run_ndarray_compat.py',s/'make_ndarray_alpha_probe.py',")
source=source.replace("str(s/'ndarray_compat_gpu.mm'),'-framework'","str(out/'ndarray_alpha_gpu.mm'),'-framework'")
source=source.replace("binary=out/'ndarray_compat_gpu'","binary=out/'ndarray_alpha_gpu'")
source=source.replace("metadata=dict(scope=__doc__,source_sha256=manifest,", "metadata=dict(scope=__doc__,generated_cpp_sha256=digest(out/'ndarray_alpha_gpu.mm'),source_sha256=manifest,")
old="for mode in ('mixed','f32'):\n    try:run(mode,[str(binary),mode])"
new="for storage,variant in [('mixed','unit'),('mixed','three'),('f32','unit'),('f32','three')]:\n    mode=storage+'-'+variant\n    try:run(mode,[str(binary),storage,variant])"
assert source.count(old)==1;source=source.replace(old,new)
source=repr(__doc__)+source[source.index('\n'):]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
