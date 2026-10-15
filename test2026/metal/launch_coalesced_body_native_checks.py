"""Require two archived component trials and installed pinned native sources."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name in ('coalesced-body-block-20261015','coalesced-body-block-confirm-20261015','interleaved-body-native-confirm-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
audit=json.loads((root/'coalesced-body-native-staging/component-verification.json').read_text())
assert audit['numeric_checks']==144 and audit['control_profiles']==378 and audit['timed_executions']==2304
manifest=json.loads((root/'coalesced-body-native-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text()
source=source.replace('run_guarded_global_half_alpha_checks.py','run_guarded_coalesced_body_native_checks.py')
exec(compile(source,__file__,'exec'))
