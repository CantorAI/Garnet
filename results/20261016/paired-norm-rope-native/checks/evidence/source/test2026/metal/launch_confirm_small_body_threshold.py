"""Repeat the smaller-body component only after its original-bit audit."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve();prior=root/'logs/small-body-threshold-20261015'
assert json.loads((prior/'pid.json').read_text())['pid']==46404
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/small-body-threshold-verification.json').read_text())
assert audit['numeric_checks']==24 and audit['safety_profiles']==90 and audit['timed_executions']==384
meta=json.loads((prior/'jobs/metadata.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in meta['source_sha256'].items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_small_body_threshold.py')
exec(compile(source,__file__,'exec'))
