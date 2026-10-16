"""Require independent initial output audit and unchanged executed source."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();prior=root/'logs/screened-head-after-vision-20261015'
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/screened-head-verification.json').read_text());assert audit['successful_trials']==1 and audit['query_checks']==51 and audit['row_state_checks']==78
meta=json.loads((prior/'jobs/metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_screened_head_repeat.py')
exec(compile(source,__file__,'exec'))
