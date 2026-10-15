"""Require two actual component trials, independent audit and unchanged stage."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('screened-seed-abba-20261015','screened-seed-abba-confirm-20261015'):assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/screened-seed-abba-verification.json').read_text());assert audit['successful_trials']==2 and audit['query_checks']==102 and audit['row_state_checks']==156
manifest=json.loads((root/'small-seed-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/n).read_bytes()).hexdigest()==sha for n,sha in manifest.items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_small_seed_checks.py')
exec(compile(source,__file__,'exec'))
