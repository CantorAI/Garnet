"""Durable exclusive component launch after the full deferred-clear trials ended."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/deferred-state-zero-confirm-native-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_reused_zero_source_probe.py')
exec(compile(source,__file__,'exec'))
