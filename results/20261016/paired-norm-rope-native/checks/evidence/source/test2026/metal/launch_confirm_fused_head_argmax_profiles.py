"""Repeat all six profiles only after independent full-gate/output verification."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/fused-head-argmax-native-repair-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/fused-head-argmax-native-verification.json').read_text())
assert audit['original_validation_stages']==19 and audit['trained_recovery_cases']==6 and audit['full_output_requests']==72 and len(audit['profiles'])==6
assert hashlib.sha256((root/'build-fused-head-argmax/libgarnet.dylib').read_bytes()).hexdigest()=='8cfcfb392d19fad68695ad7d80c8c3b5989f01ca956fa50370c7b2f960230177'
manifest=json.loads((root/'fused-head-argmax-staging/manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_fused_head_argmax_profiles.py')
exec(compile(source,__file__,'exec'))
