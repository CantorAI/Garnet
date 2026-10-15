"""Launch prefill study only after the full retained-source trial terminated."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/reused-zero-source-confirm-native-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_coalesced_activation_prefill.py')
exec(compile(source,__file__,'exec'))
