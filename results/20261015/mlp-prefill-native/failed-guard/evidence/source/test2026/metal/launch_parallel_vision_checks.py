"""Require audited terminal component trials, pin installed sources, wait for quiet GPU."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('parallel-vision-context-20261015','parallel-vision-context-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/parallel-vision-context-verification.json').read_text());assert audit['trials']==2 and audit['shape_checks']==48
manifest=json.loads((root/'parallel-vision-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_parallel_vision_checks.py')
exec(compile(source,__file__,'exec'))
