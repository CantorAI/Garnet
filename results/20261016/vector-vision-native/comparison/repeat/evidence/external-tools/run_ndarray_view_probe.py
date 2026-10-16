"""Deferred RHS transpose compatibility with original synthetic numerical/guard gates."""
from pathlib import Path
import sys
p=Path(__file__).with_name('run_ndarray_alpha_probe.py');ns={'__file__':str(p)}
saved=sys.argv;sys.argv=[saved[0],saved[1],saved[2],'--generate-only']
exec(compile(p.read_text(),str(p),'exec'),ns);sys.argv=saved
source=ns['source'].replace("s/'make_ndarray_alpha_probe.py'","s/'make_ndarray_view_probe.py',s/'make_ndarray_alpha_probe.py'")
# The harness command has a single path, not a snapshot list.
source=source.replace("str(s/'make_ndarray_view_probe.py',s/'make_ndarray_alpha_probe.py')","str(s/'make_ndarray_view_probe.py')")
source=source.replace('ndarray_alpha_gpu','ndarray_view_gpu')
source=source.replace('Unit coefficient and explicit zero-addend diagnostics, sequential per format.',__doc__)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
