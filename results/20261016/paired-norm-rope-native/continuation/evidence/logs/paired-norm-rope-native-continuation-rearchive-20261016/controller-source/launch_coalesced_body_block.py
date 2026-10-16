"""Exclusive exact block compression probe after terminal wide-body experiment."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/wide-body-decode-repaired-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text()
source=source.replace('run_guarded_global_half_alpha_checks.py','run_coalesced_body_block.py')
exec(compile(source,__file__,'exec'))
