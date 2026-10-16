"""Launch learned-matrix experiment only after isolated capability is terminal."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve();cap=root/'logs/mixed-accumulator-capability-20261015'
assert json.loads((cap/'exit.json').read_text())['exit_code']==0
result=json.loads((cap/'jobs/capabilities.log').read_text());assert result['supported'] and result['exact_outputs']==128
assert json.loads((root/'logs/counter-trace-sparse-profiles-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_coalesced_activation_decode.py')
exec(compile(source,__file__,'exec'))
