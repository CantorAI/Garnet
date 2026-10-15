"""Launch independent dispatch study after the owned prefill probe is terminal."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/coalesced-activation-prefill-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_concurrent_bf16_groups.py')
exec(compile(source,__file__,'exec'))
