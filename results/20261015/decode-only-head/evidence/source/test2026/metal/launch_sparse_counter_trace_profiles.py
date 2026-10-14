"""Begin sparse profiling only after full diagnostic is terminal and archived."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/counter-trace-file-profiles-20261015/exit.json').read_text())['exit_code']==0
assert (root/'counter-trace-file-evidence.tar.gz').is_file()
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_sparse_counter_trace_profiles.py')
exec(compile(source,__file__,'exec'))
