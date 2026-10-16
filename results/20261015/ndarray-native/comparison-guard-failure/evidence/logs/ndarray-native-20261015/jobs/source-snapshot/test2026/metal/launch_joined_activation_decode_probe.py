"""Run only after both native reset measurement jobs are terminal; keep selection."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('dirty-prefix-reset-repair-native-20261014','dirty-prefix-reset-repair-confirm-native-20261014'):
 assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_single_row_bf16_probe.py').read_text().replace('run_single_row_bf16_probe.py','run_joined_activation_decode_probe.py')
exec(compile(source,__file__,'exec'))
