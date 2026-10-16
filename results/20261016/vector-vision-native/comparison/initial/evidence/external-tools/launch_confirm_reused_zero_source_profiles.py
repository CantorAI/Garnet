"""Repeat all six profiles only after the independent initial-output audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/reused-zero-source-native-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/reused-zero-source-native-verification.json').read_text())
assert audit['original_validation_stages']==19 and audit['trained_recovery_cases']==6 and audit['full_output_requests']==72 and len(audit['profiles'])==6
assert hashlib.sha256((root/'build-reused-zero-source/libgarnet.dylib').read_bytes()).hexdigest()=='c54ff5a158ce48bce8dd36e8eec6c2e59049013d13fbdf4db22233fb4476d6ac'
manifest=json.loads((root/'reused-zero-source-staging/manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_reused_zero_source_profiles.py')
exec(compile(source,__file__,'exec'))
