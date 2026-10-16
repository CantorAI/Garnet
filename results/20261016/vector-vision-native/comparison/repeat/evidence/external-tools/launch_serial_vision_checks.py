"""Require the independent repeated component audit and unchanged staged sources."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('serial-vision-context-20261015','serial-vision-context-confirm-20261015'):assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/serial-vision-context-confirm-verification.json').read_text());assert audit['shape_variant_checks']==144 and audit['failed_admissions']==0
manifest=json.loads((root/'serial-vision-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/n).read_bytes()).hexdigest()==sha for n,sha in manifest.items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_serial_vision_checks.py')
exec(compile(source,__file__,'exec'))
