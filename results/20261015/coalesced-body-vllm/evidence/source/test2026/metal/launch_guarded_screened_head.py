"""Require audited terminal width experiments before bounded-query refinement checks."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
for name in ('screened-head-after-vision-20261015','screened-head-confirm-20261015','screened-head-int4-20261015','screened-head-int6-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/screened-head-all-widths-verification.json').read_text());assert audit['successful_trials']==4 and audit['query_checks']==204
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_screened_head.py')
exec(compile(source,__file__,'exec'))
