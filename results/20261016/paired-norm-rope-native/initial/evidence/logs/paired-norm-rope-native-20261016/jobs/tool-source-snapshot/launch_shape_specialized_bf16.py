"""Launch fixed-shape study after the owned grouped-pass repeat ended."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/concurrent-bf16-groups-confirm-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_shape_specialized_bf16.py')
exec(compile(source,__file__,'exec'))
