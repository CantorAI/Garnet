"""Run only after the sequential width sweep is terminal and independently audited."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/screened-seed-sweep-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/screened-seed-sweep-verification.json').read_text());assert audit['query_checks']==204 and audit['row_state_checks']==312
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_screened_seed_abba.py')
exec(compile(source,__file__,'exec'))
