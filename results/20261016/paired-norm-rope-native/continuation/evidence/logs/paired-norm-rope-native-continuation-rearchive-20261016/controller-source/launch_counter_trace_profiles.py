"""Launch a guarded isolated diagnostic after the hardware capability gate."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/counter-trace-profiles-20261015/exit.json').read_text())['exit_code']==1
assert 'JSONDecodeError' in (root/'logs/counter-trace-profiles-20261015/output.log').read_text()
assert (root/'counter-trace-failed-evidence.tar.gz').is_file()
folder=root/'logs/metal-counter-capabilities-20261015'
assert json.loads((folder/'exit.json').read_text())['exit_code']==0
cap=json.loads((folder/'jobs/capabilities.log').read_text());assert cap['stage_boundary'] and cap['sampled_compute_passes']==2 and cap['exact_dependency_values']==4096
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_counter_trace_profiles.py')
exec(compile(source,__file__,'exec'))
