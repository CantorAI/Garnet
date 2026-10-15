"""Repeat all six deferred-clear profiles after independent full-output audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/deferred-state-zero-native-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/deferred-state-zero-native-verification.json').read_text())
assert audit['original_validation_stages']==19 and audit['trained_recovery_cases']==6 and audit['full_output_requests']==72 and len(audit['profiles'])==6
assert hashlib.sha256((root/'build-deferred-state-zero/libgarnet.dylib').read_bytes()).hexdigest()=='6cc869948b05a5d44708120fca3a7dfd13268cede7dc5233bd270011bcedcb31'
manifest=json.loads((root/'deferred-state-zero-staging/manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_deferred_state_zero_profiles.py')
exec(compile(source,__file__,'exec'))
