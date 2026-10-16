"""Durable exclusive head-component launcher after the previous job ended."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/coalesced-activation-decode-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_fused_head_argmax_probe.py')
exec(compile(source,__file__,'exec'))
