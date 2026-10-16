"""Require terminal audited prior head trials before testing the middle precision index."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('screened-head-after-vision-20261015','screened-head-confirm-20261015','screened-head-int4-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/screened-head-int4-verification.json').read_text())
assert audit['successful_trials']==3 and audit['query_checks']==153 and audit['row_state_checks']==234
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_screened_head_int6.py')
exec(compile(source,__file__,'exec'))
