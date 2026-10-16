"""Repeat the full exact compression grid after independent raw-window audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/coalesced-body-block-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/coalesced-body-block-verification.json').read_text())
assert audit['numeric_checks']==72 and audit['control_profiles']==189 and audit['timed_executions']==1152
meta=json.loads((root/'logs/coalesced-body-block-20261015/jobs/metadata.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in meta['source_sha256'].items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text()
source=source.replace('run_guarded_global_half_alpha_checks.py','run_coalesced_body_block.py')
exec(compile(source,__file__,'exec'))
