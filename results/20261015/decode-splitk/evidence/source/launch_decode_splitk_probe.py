"""Launch only after earlier owned jobs reach terminal status; guards remain intact."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
for name in ('combined-prefill-reset-20261015','combined-prefill-reset-confirm-20261015','decode-eight-lane-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_decode_matrix_probe.py').read_text().replace('run_decode_matrix_probe.py','run_decode_splitk_probe.py')
exec(compile(source,__file__,'exec'))
