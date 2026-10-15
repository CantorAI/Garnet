"""Run one guarded joint-clear component only after all inference jobs ended."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/decode-only-head-confirm-native-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_joint_state_zero_probe.py')
exec(compile(source,__file__,'exec'))
