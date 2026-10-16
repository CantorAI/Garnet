"""Repeat all six route-policy profiles after independent complete-output audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/decode-only-head-native-20261015/exit.json').read_text())['exit_code']==0
audit=json.loads((root/'work/decode-only-head-native-verification.json').read_text())
assert audit['original_validation_stages']==19 and audit['trained_recovery_cases']==6 and audit['full_output_requests']==72 and len(audit['profiles'])==6
assert hashlib.sha256((root/'build-decode-only-head/libgarnet.dylib').read_bytes()).hexdigest()=='5f40fdfb1921a0024bb1119fe6c2b2925498ef3e3256eab6133431c66f49795c'
manifest=json.loads((root/'decode-only-head-staging/manifest.json').read_text())
for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
source=Path(__file__).with_name('launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py','run_decode_only_head_profiles.py')
exec(compile(source,__file__,'exec'))
